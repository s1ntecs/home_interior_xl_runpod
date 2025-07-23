import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict, Tuple, Union, List, Optional
from PIL import Image, ImageFilter

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL, DDIMScheduler
)
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from controlnet_aux import MidasDetector

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

from colors import ade_palette
from utils import map_colors_rgb

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250
TARGET_RES = 1024  # SDXL рекомендует 1024×1024

logger = RunPodLogger()


DEFAULT_CONTROL_ITEMS = [
    "windowpane;window",
    "column;pillar",
    "door;double;door",
]


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side
    scale = min(max_side / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # выравниваем до кратных 8
    new_w = round_to_multiple(new_w, 8)
    new_h = round_to_multiple(new_h, 8)
    return max(new_w, 8), max(new_h, 8)


def normalize_control_items(raw) -> List[str]:
    """
    Приводит control_items к списку строк.
    Поддерживает: None, строку с запятыми/новыми строками, список.
    """
    if raw is None:
        return DEFAULT_CONTROL_ITEMS[:]

    if isinstance(raw, str):
        # разделяем по запятым или переносам
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        return parts or DEFAULT_CONTROL_ITEMS[:]

    if isinstance(raw, (list, tuple)):
        cleaned = [str(x).strip() for x in raw if str(x).strip()]
        return cleaned or DEFAULT_CONTROL_ITEMS[:]

    # на всякий случай
    return DEFAULT_CONTROL_ITEMS[:]


def filter_items(
    colors_list: Union[List, np.ndarray],
    items_list: Union[List, np.ndarray],
    items_to_remove: Union[List, np.ndarray],
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)

    return filtered_colors, filtered_items


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True
)

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler"
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)


PIPELINE = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    # "RunDiffusion/Juggernaut-XL-v9",
    # "SG161222/RealVisXL_V5.0",
    "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    # controlnet=[cn_depth, cn_seg],
    controlnet=controlnet,
    torch_dtype=DTYPE,
    # variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
    requires_safety_checker=False,
    add_watermarker=False,
    use_safetensors=True,
    resume_download=True,
    scheduler=eulera_scheduler,
    vae=vae,
)
PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
    PIPELINE.scheduler.config)
PIPELINE.enable_xformers_memory_efficient_attention()
PIPELINE.to(DEVICE)

REFINER = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=DTYPE,
    variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
)
REFINER.scheduler = DDIMScheduler.from_config(REFINER.scheduler.config)
REFINER.to(DEVICE)

midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")

seg_image_processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)


@torch.inference_mode()
@torch.autocast(DEVICE)
def segment_image(image):
    """
    Segments an image using a semantic segmentation model.

    Args:
        image (PIL.Image): The input image to be segmented.
        image_processor (AutoImageProcessor): The processor to prepare the
            image for segmentation.
        image_segmentor (SegformerForSemanticSegmentation): The semantic
            segmentation model used to identify different segments in the image.

    Returns:
        Image: The segmented image with each segment colored differently based
            on its identified class.
    """
    pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = seg_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg).convert("RGB")

    return seg_image


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required"}

        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        negative_prompt = payload.get(
            "negative_prompt", "")
        img_strength = payload.get(
            "img_strength", 0.5)
        guidance_scale = float(payload.get(
            "guidance_scale", 7.5))
        steps = min(int(payload.get(
            "steps", MAX_STEPS)),
                    MAX_STEPS)

        seed = int(payload.get(
            "seed",
            random.randint(0, MAX_SEED)))
        generator = torch.Generator(
            device=DEVICE).manual_seed(seed)

        # refiner
        refiner_strength = float(payload.get(
            "refiner_strength", 0.2))
        refiner_steps = int(payload.get(
            "refiner_steps", 15))
        refiner_scale = float(payload.get(
            "refiner_scale", 7.5))

        # control scales
        depth_scale = float(payload.get(
            "depth_conditioning_scale", 0.8))
        depth_guidance_start = float(payload.get(
            "depth_guidance_start", 0.0))
        depth_guidance_end = float(payload.get(
            "depth_guidance_end", 1.0))
        # ---------- препроцессинг входа ------------

        # mask
        control_items = normalize_control_items(payload.get("control_items"))
        mask_blur_radius = float(payload.get("mask_blur_radius", 3))

        image_pil = url_to_pil(image_url)

        control_image = midas(image_pil)

        orig_w, orig_h = image_pil.size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)

        # resize *both* the init image and the control image to the same, /8-aligned size
        image_pil = image_pil.resize((work_w, work_h),
                                     Image.Resampling.LANCZOS)
        depth_cond = control_image.resize((work_w, work_h),
                                          Image.Resampling.LANCZOS)

        real_seg = np.array(
            segment_image(image_pil)
        )
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]),
                                  axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]

        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=control_items,
        )

        logger.log(f"SEGMENTED ITEMS {segment_items}")

        mask = np.zeros_like(real_seg)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1
        mask_image = Image.fromarray(
            (mask * 255).astype(np.uint8)).convert("RGB")
        mask_image = mask_image.filter(
            ImageFilter.GaussianBlur(radius=mask_blur_radius))

        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            control_image=depth_cond,
            controlnet_conditioning_scale=depth_scale,
            control_guidance_start=depth_guidance_start,
            control_guidance_end=depth_guidance_end,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            strength=img_strength,
            mask_image=mask_image
        ).images

        final = []
        for im in images:
            im = im.resize((orig_w, orig_h),
                           Image.Resampling.LANCZOS).convert("RGB")
            ref = REFINER(
                prompt=prompt, image=im, strength=refiner_strength,
                num_inference_steps=refiner_steps, guidance_scale=refiner_scale
            ).images[0]
            final.append(ref)

        mask_vis = mask_image.resize((orig_w, orig_h),
                                     resample=Image.Resampling.NEAREST).convert("RGB")
        final.append(mask_vis)
        torch.cuda.empty_cache()

        return {
            "images_base64": [pil_to_b64(i) for i in final],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps, "seed": seed
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
