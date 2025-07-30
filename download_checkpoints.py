import gc
import os
import torch

from diffusers import (
    StableDiffusionXLInpaintPipeline
)

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
    StableDiffusionXLInpaintPipeline.from_pretrained(
        # "RunDiffusion/Juggernaut-XL-v9",
        # "SG161222/RealVisXL_V5.0",
        # "misri/cyberrealisticPony_v90Alt1",
        # "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        add_watermarker=False,
        variant="fp16",
    )



if __name__ == "__main__":
    # fetch_checkpoints()
    get_pipeline()
