import gc
import os
import torch

from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionXLControlNetInpaintPipeline,
    AutoencoderKL,
    StableDiffusionXLImg2ImgPipeline
)
from controlnet_aux import MidasDetector
from transformers import (AutoImageProcessor,
                          SegformerForSemanticSegmentation)

from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

LORA_NAMES = [
    # "ArtDeco_XL.safetensors",
    # "Authoritarian_XL.safetensors",
    # "Bathroom_XL.safetensors",
    # "FictionInterior_XL.safetensors",
    # "FuturismStyle_Interior_XL.safetensors",
    # "JapaneseInterior_XL.safetensors",
    # "KidsRoom_XL.safetensors",
    # "LivingRoom_XL.safetensors",
    # "LuxuryBedroomXL.safetensors",
    # "ModernStyle_XL.safetensors",
    # "NordicStyle_XL.safetensors",
    # "PublicSpace_XL.safetensors"
]


# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и все внешние зависимости."""
    hf_hub_download(
        repo_id="sintecs/interior",
        filename="interiorSceneXL_v1.safetensors",
        local_dir="checkpoints"
    )

    for fname in LORA_NAMES:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="loras"
        )


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16,
                                        use_safetensors=True)
    print("LOADED VAE")
    # PIPELINE = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    #     # "RunDiffusion/Juggernaut-XL-v9",
    #     # "SG161222/RealVisXL_V5.0",
    #     # "misri/cyberrealisticPony_v90Alt1",
    #     "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    #     torch_dtype=torch.float16,
    #     add_watermarker=False,
    #     controlnet=controlnet,
    #     vae=vae,
    #     # variant="fp16",
    #     use_safetensors=True,
    #     resume_download=True,
    # ).to(DEVICE)
    PIPELINE = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
        "checkpoints/interiorSceneXL_v1.safetensors",
        torch_dtype=torch.float16,
        add_watermarker=False,
        vae=vae,
        controlnet=controlnet,
    )

    print("LOADED PIPELINE")
    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
        PIPELINE.scheduler.config)

    PIPELINE.enable_model_cpu_offload

    del PIPELINE

    gc.collect()

    StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    )
    # print("LOADED REFINER")
    MidasDetector.from_pretrained(
        "lllyasviel/ControlNet"
    )

    AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
