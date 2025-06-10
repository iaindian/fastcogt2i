#!/usr/bin/env python3
"""
Post-install script for downloading custom model weights into ComfyUI/models.

Run this after the main installer. It skips files that already exist.
Supports:
- Hugging Face Hub (via `huggingface_hub`)
- Direct URLs
- Google Drive links (via `gdown`)
"""
import sys
import logging
from pathlib import Path
import requests
import urllib.request
from tqdm import tqdm
import zipfile

# Optional imports
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base models directory
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "ComfyUI" / "models"

# Model entries
# Each entry must include exactly one of:
#   'repo_id' + 'filename' for HF
#   'url' + 'filename' for direct HTTP
#   'url' + 'filename' (and gdrive ID) for Google Drive
# and may include 'subdir'.
MODELS = [
    # HF example:
    # {"repo_id": "Gourieff/ReActor", "filename": "models/facerestore_models/codeformer-v0.1.0.pth", "subdir": "facerestore_models"},
    # URL example:
    # {"url": "https://example.com/model.bin", "filename": "model.bin", "subdir": "custom_models"},
    # Google Drive example:
    # {"url": "https://drive.google.com/uc?id=FILE_ID", "filename": "model_drive.ckpt", "subdir": "custom_models"},
    
  {
    "url": "https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/resolve/main/TTPLANET_Controlnet_Tile_realistic_v2_rank256.safetensors",
    "filename": "TTPLANET_Controlnet_Tile_realistic_v2_rank256.safetensors",
    "subdir": "controlnet"
  },

  {
    "url": "https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth?download=true",
    "filename": "4x_NMKD-Superscale-SP_178000_G.pth",
    "subdir": "upscale_models"
  },
  {
    "url": "https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth",
    "filename": "4x_NMKD-Siax_200k.pth",
    "subdir": "upscale_models"
  },

  {
    "url": "https://huggingface.co/Hishambarakat/checkpoint/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors",
    "filename": "ip-adapter_pulid_sdxl_fp16.safetensors",
    "subdir": "pulid"
  },

  {
    "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
    "filename": "inswapper_128.onnx",
    "subdir": "insightface"
  },

  {
    "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
    "filename": "GFPGANv1.3.pth",
    "subdir": "facerestore_models"
  },
  {
    "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "filename": "parsing_parsenet.pth",
    "subdir": "facedetection"
  },
  {
    "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "filename": "detection_Resnet50_Final.pth",
    "subdir": "facedetection"
  },

  {
    "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
    "filename": "GFPGANv1.4.pth",
    "subdir": "facerestore_models"
  },
  {
    "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
    "filename": "codeformer-v0.1.0.pth",
    "subdir": "facerestore_models"
  },
  {
    "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
    "filename": "GPEN-BFR-512.onnx",
    "subdir": "facerestore_models"
  },
  {
    "url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx",
    "filename": "1k3d68.onnx",
    "subdir": "insightface/models/antelopev2"
  },
  {
    "url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx",
    "filename": "2d106det.onnx",
    "subdir": "insightface/models/antelopev2"
  },
  {
    "url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx",
    "filename": "genderage.onnx",
    "subdir": "insightface/models/antelopev2"
  },
  {
    "url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx",
    "filename": "glintr100.onnx",
    "subdir": "insightface/models/antelopev2"
  },
  {
    "url": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx",
    "filename": "scrfd_10g_bnkps.onnx",
    "subdir": "insightface/models/antelopev2"
  },
  {
       "url":  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
       "filename": "buffalo_l.zip",
       "subdir":   "insightface/models/buffalo_l",
   },
  {
       "url":  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
       "filename": "clip_l.safetensors",
       "subdir":   "clip",
   },
  {
       "url":  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
       "filename": "t5xxl_fp8_e4m3fn.safetensors",
       "subdir":   "clip",
   },
  {
       "url":  "https://huggingface.co/jagat334433/beru_custom/resolve/main/ae.safetensors",
       "filename": "ae.safetensors",
       "subdir":   "vae",
   },
  # loras
  {
     "url": "https://huggingface.co/jagat334433/beru_custom/resolve/main/FameGrid_Bold_SDXL_V1.safetensors",
    "filename": "FameGrid_Bold_SDXL_V1.safetensors",
    "subdir": "loras"
    },
 
   
       {
    "url": "https://huggingface.co/jagat334433/beru_custom/resolve/main/beru_custom_merge.safetensors",
    "filename": "beru_custom_merge.safetensors",
    "subdir": "checkpoints"
  },
       {
    "url": "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf",
    "filename": "flux1-dev-Q4_K_S.gguf",
    "subdir": "unet"
  },
      #  dfixmodels
    {
    "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
    "filename": "GroundingDINO_SwinT_OGC.cfg.py",
    "subdir": "grounding-dino"
  },

  {
    "url": " https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    "filename": "groundingdino_swint_ogc.pth",
    "subdir": "grounding-dino"
  },
  {
    "url": "https://huggingface.co/jagat334433/beru_custom/resolve/main/beru_custom_2.safetensors",
    "filename": "beru_custom_2.safetensors",
    "subdir": "checkpoints"
  },
     
  {
    "url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
    "filename": "fooocus_inpaint_head.pth",
    "subdir": "inpaint"
  },

  {
    "url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
    "filename": "inpaint_v26.fooocus.patch",
    "subdir": "inpaint"
  },
  #     {
  #      "url":  "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
  #      "filename": "flux1-dev.safetensors",
  #      "subdir":   "diffusion_models",
  #  },
   
    
]


def download_hf(entry):
    repo_id = entry['repo_id']
    filename = entry['filename']
    subdir = entry.get('subdir', '')
    dest = MODELS_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(filename).name
    if target.exists():
        logger.info(f"Skipping existing HF model: {target}")
        return
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not installed; cannot download HF model")
        return
    logger.info(f"Downloading from HF: {repo_id}/{filename}")
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest), local_dir_use_symlinks=False)
    Path(downloaded).rename(target)
    logger.info(f"Saved HF model to: {target}")


# def download_url(entry):
#     url = entry['url']
#     filename = entry['filename']
#     subdir = entry.get('subdir', '')
#     dest = MODELS_DIR / subdir
#     dest.mkdir(parents=True, exist_ok=True)
#     target = dest / filename
#     if target.exists():
#         logger.info(f"Skipping existing URL model: {target}")
#         return
#     # Google Drive detection
#     if GDOWN_AVAILABLE and 'drive.google.com' in url:
#         logger.info(f"Downloading Google Drive file: {url}")
#         gdown.download(url, str(target), quiet=False, fuzzy=True)
#     else:
#         logger.info(f"Downloading URL with progress: {url}")
#         # Download using urllib and tqdm for progress
#         with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=f"Downloading {filename}") as pbar:
#             def _hook(count, block_size, total_size):
#                 pbar.total = total_size
#                 pbar.update(block_size)
#             urllib.request.urlretrieve(url, str(target), reporthook=_hook)
#     logger.info(f"Saved URL model to: {target}")
    
#         # --- new: if it's a zip, extract and remove archive ---
#     if target.suffix == '.zip':
#         logger.info(f"Unzipping {target} into {target.parent}...")
#         with zipfile.ZipFile(str(target), 'r') as zf:
#             zf.extractall(str(target.parent))
#         target.unlink()  # delete the .zip
#         logger.info(f"Removed archive {target}")

def download_url(entry):
    url = entry['url']
    filename = entry['filename']
    subdir = entry.get('subdir', '')
    dest = MODELS_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / filename

    if target.exists():
        logger.info(f"Skipping existing URL model: {target}")
        return

    # Google Drive detection
    if GDOWN_AVAILABLE and 'drive.google.com' in url:
        logger.info(f"Downloading Google Drive file: {url}")
        gdown.download(url, str(target), quiet=False, fuzzy=True)
        return

    logger.info(f"Downloading (streamed) URL: {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(target, 'wb') as f, tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if target.exists():
            target.unlink()  # Delete partial file if failed
        return

    logger.info(f"Saved model to: {target}")

    # --- unzip if needed ---
    if target.suffix == '.zip':
        logger.info(f"Unzipping {target} into {target.parent}...")
        with zipfile.ZipFile(str(target), 'r') as zf:
            zf.extractall(str(target.parent))
        target.unlink()
        logger.info(f"Removed archive {target}")


def main():
    if not MODELS_DIR.exists():
        logger.error(f"Models directory not found: {MODELS_DIR}")
        sys.exit(1)
    for entry in MODELS:
        try:
            if 'repo_id' in entry and 'filename' in entry:
                download_hf(entry)
            elif 'url' in entry and 'filename' in entry:
                download_url(entry)
            else:
                logger.error(f"Invalid entry, must include 'repo_id' or 'url' + 'filename': {entry}")
        except Exception as e:
            logger.error(f"Failed to download for entry {entry}: {e}")
    logger.info("All model downloads complete.")

if __name__ == '__main__':
    main()
