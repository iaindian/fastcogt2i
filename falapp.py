from tempfile import TemporaryDirectory
import textwrap
import fal

from fal import ContainerImage
from fal.toolkit import download_file, Image, clone_repository
from pydantic import BaseModel, Field


from typing import List, Union
import subprocess
import time
import logging
import requests
import copy
import shutil
import json
import hashlib
import uuid
from io import BytesIO
import tarfile
from pathlib import Path
import os



#default_workflow_path = (Path(__file__).parent / "workflow_api/face-match-4-12-api.json").resolve()


class Input(BaseModel):
    images: str = Field(
        default='[{"name": "", "inputs": {"prompt": "((Close-up headshot)) ((cjw woman)) with a confident smile, wearing a tailored blazer, in front of a neutral background", "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream", "width": "640", "height": "960", "num_steps": 25}}]',
        description="JSON list of image entries with 'name' and 'inputs'",
    )
    weights: str = Field(
        default="",
        description="Load LoRA weights (tar.gz URL)"
    )
    image1: str = Field(description="First input image")
    image2: str = Field(description="Second input image")
    image3: str = Field(description="Third input image")
    # bypass_reactor: bool = Input(default=False, description="Skip ReActor nodes"),
    # bypass_upscale_node: bool = Input(default=False, description="Skip upscaling/TTP nodes"),
    bypass_dfix_node: bool = Field(default=True, description="Skip Dfix nodes by default")
    poll_interval: float = Field(default=1.0, description="Seconds between polls")
    timeout: float = Field(default=300.0, description="Completion timeout (s)")
    log_level: str = Field(default="INFO", description="Logging level")
    do_settings: str = Field(default="", description="DigitalOcean Spaces JSON settings")


class Output(BaseModel):
    results: List[Union[Image, dict]]

class FastCogt2i(
    fal.App,
    kind="container",
    image=ContainerImage.from_dockerfile_str(
        textwrap.dedent(
            """
            #FROM python:3.10
            FROM r8.im/thumbnailai/sdxl-comfyui@sha256:09ad66b49304259b69136c5ebc423229d6ec7d316dcd24cd113253a2bfe0324e
            #RUN apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 imagemagick gifsicle

            #RUN curl -o /usr/local/bin/pget -L \
            #    "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)" \
            #    && chmod +x /usr/local/bin/pget
            RUN pip install --upgrade pip
            RUN pip uninstall -y onnxruntime onnxruntime-gpu
            RUN pip cache purge
            RUN pip install onnxruntime-gpu pydantic_settings git+https://github.com/facebookresearch/sam2 alembic
            RUN export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
            #ADD https://github.com/fal-ai-community/fastcogt2i.git#e22b59d132edbec8a1601bcbe34fc91addc9db28 /app/fastcogt2i
            #RUN pget https://dev.mymoodai.app/v1/models/fastcogt2i.tar /app/fastcogt2i -x
            WORKDIR /src/
            RUN git clone https://github.com/infinigence/ComfyUI_Model_Cache.git /src/ComfyUI/custom_nodes/ComfyUI_Model_Cache
            #RUN python setup_final.py
            #RUN python download_models.py
            #RUN python patchNSFW.py
            """
        )
    )
):
    app_name = "fastcogt2i"
    app_auth = "private"
    machine_type = "GPU-H100"
    local_python_modules = ["comfyrunbatch"]
    requirements = [
        "torch",
        "torchvision",
        "torchaudio",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu121",
        "accelerate",
        "aiohappyeyeballs",
        "aiohttp",
        "aiosignal",
        "albucore",
        "albumentations",
        "annotated-types",
        "anyio",
        "async-timeout",
        "attrs",
        "av",
        "beautifulsoup4",
        "certifi",
        "cffi",
        "charset-normalizer",
        "clip-interrogator",
        "color-matcher",
        "coloredlogs",
        "colour-science",
        "comfyui_frontend_package",
        "comfyui_workflow_templates",
        "contourpy",
        "cycler",
        "Cython",
        "ddt",
        "diffusers",
        "dill",
        "docutils",
        "easydict",
        "einops",
        "eval_type_backport",
        "exceptiongroup",
        "facexlib",
        "filelock",
        "filterpy",
        "flatbuffers",
        "flet",
        "fonttools",
        "frozenlist",
        "fsspec",
        "ftfy",
        "gdown",
        "gguf",
        "h11",
        "httpcore",
        "httpx",
        "huggingface-hub",
        "humanfriendly",
        "idna",
        "imageio",
        "importlib_metadata",
        "insightface",
        "Jinja2",
        "joblib",
        "jsonschema",
        "jsonschema-specifications",
        "kiwisolver",
        "kornia",
        "kornia_rs",
        "lark",
        "lazy_loader",
        "llvmlite",
        "MarkupSafe",
        "matplotlib",
        "mpmath",
        "mss",
        "multidict",
        "networkx",
        "numba",
        "numpy",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cusparselt-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvtx-cu12",
        "oauthlib",
        "onnx",
        #"onnxruntime",
        "onnxruntime-gpu",
        "open_clip_torch",
        "opencv-python",
        "opencv-python-headless",
        "packaging",
        "pandas",
        "peft",
        "piexif",
        "pillow",
        "pixeloe",
        "platformdirs",
        "pooch",
        "prettytable",
        "propcache",
        "protobuf",
        "psutil",
        "py-cpuinfo",
        "pycparser",
        "pydantic",
        "pydantic_core",
        "pydantic_settings",
        "PyMatting",
        "pyparsing",
        "PySocks",
        "python-dateutil",
        "pytz",
        "PyYAML",
        "referencing",
        "regex",
        "rembg",
        "repath",
        "replicate",
        "requests",
        "rpds-py",
        "safetensors",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "seaborn",
        "segment-anything",
        "sentencepiece",
        "simsimd",
        "six",
        "sniffio",
        "soundfile",
        "soupsieve",
        "spandrel",
        "stringzilla",
        "sympy",
        "threadpoolctl",
        "tifffile",
        "timm",
        "tokenizers",
        "torchsde",
        "tqdm",
        "trampoline",
        "transformers",
        "transparent-background",
        "triton",
        "typing-inspection",
        "typing_extensions",
        "tzdata",
        "ultralytics",
        "ultralytics-thop",
        "urllib3",
        "wcwidth",
        "wget",
        "yarl",
        "zipp",
        "boto3",
        "addict",
        "yapf",
        "alembic",
    ]

    def setup(self):
        # Launch ComfyUI server
        self.server = subprocess.Popen([
            "python", "ComfyUI/main.py",
            "--listen", "0.0.0.0",
            "--port", "8188"
        ])
        host = "http://127.0.0.1:8188"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(host, timeout=5)
                if r.status_code == 200:
                    logging.info("ComfyUI server ready at %s", host)
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("ComfyUI failed to start within 60s")

    def get_bucket_name(self, filename: str, prefix: str) -> str:
        # Original lookup_map for bucket suffixes
        lookup_map = {
            "a": "a1", "b": "b1", "c": "c1",
            "d": "d1", "e": "e1", "f": "f1",
            "A": "a2", "B": "b2", "C": "c2",
            "D": "d2", "E": "e2", "F": "f2",
        }
        # Determine the character to lookup based on filename
        if "_" not in filename:
            lookup_char = filename[0]
        else:
            subfilename = filename[filename.rfind("_") + 1:]
            lookup_char = subfilename[0]
        # Fetch mapped bucket suffix or fallback
        bucket_suffix = lookup_map.get(lookup_char, lookup_char)
        return prefix + bucket_suffix

    def upload_to_digitalocean_spaces(
        self,
        file_path: str,
        bucket_prefix: str,
        region: str,
        access_key: str,
        secret_key: str,
    ) -> str:
        import boto3

        # Upload file to DigitalOcean Spaces
        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=region,
            endpoint_url=f'https://{region}.digitaloceanspaces.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        fname = uuid.uuid4().hex
        obj = hashlib.md5(fname.encode()).hexdigest() + Path(file_path).suffix
        bucket = self.get_bucket_name(obj, bucket_prefix)
        try:
            client.upload_file(file_path, bucket, obj, ExtraArgs={'ACL': 'public-read'})
            return f'https://{bucket}.{region}.cdn.digitaloceanspaces.com/{obj}'
        except Exception as e:
            logging.error(f"DO upload failed: {e}")
            return ''

    @fal.endpoint("/")
    def predict(
        self,
        input: Input,
    ) -> Output:
        """
        Download input images, optionally load LoRA weights, run ComfyUI, and optionally upload to DO Spaces.
        Returns list of local file paths or dicts with 'name' and 'url'.
        """
        import sys, os
        # Insert the directory of this script into sys.path
        sys.path.insert(
            0,
            os.path.dirname(os.path.abspath(__file__))
        )
        import random
        from comfyrunbatch import (
            load_prompts,
            find_input_images,
            upload_images,
            load_workflow,
            inject_prompts_and_images,
            inject_parameters,
            strip_reactor_nodes,
            bypass_upscale,
            queue_workflow,
            await_completion,
            download_outputs,
            bypass_dfix,
            clear_and_interrupt
        )

        images = input.images
        weights = input.weights

        image1 = Path(download_file(input.image1, "/tmp/"))
        image2 = Path(download_file(input.image2, "/tmp/"))
        image3 = Path(download_file(input.image3, "/tmp/"))

        bypass_dfix_node = input.bypass_dfix_node
        poll_interval = input.poll_interval
        timeout = input.timeout
        log_level = input.log_level
        do_settings = input.do_settings

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s"
        )
        host = "http://127.0.0.1:8188"
        clear_and_interrupt(host)
        for directory in ["ComfyUI/input", "ComfyUI/output", "input_tmp"]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        # Load LoRA weights if provided
        if weights.strip():
            r = requests.get(weights, stream=True, timeout=30)
            r.raise_for_status()
            buf = BytesIO(r.content)
            with tarfile.open(fileobj=buf, mode='r:gz') as tar:
                tar.extractall(path="ComfyUI/models/Lora")

        # Parse inputs
        entries = json.loads(images)
        do_cfg = json.loads(do_settings) if do_settings.strip() else {}
        if not entries:
            raise RuntimeError("No image entries provided")

         # Use the three Input() Paths instead of URLs in JSON

        temp_dir = Path("input_tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        input_images = []
        for idx, img in enumerate([image1, image2, image3], start=1):
            ext = img.suffix or ".png"
            out = temp_dir / f"img{idx}{ext}"
            shutil.copy(img, out)
            input_images.append(out.name)
        upload_images(input_images, str(temp_dir), host)

        # Load workflow
        default_workflow_path = Path("/src/workflow_api/face-match-4-14-api.json").resolve()
        wf_path = default_workflow_path
        logging.info(f"Using default workflow file: {wf_path}")
        logging.debug(f"Exists? {wf_path.exists()}, CWD={Path().resolve()}")
        if not wf_path.is_file():
            logging.error(f"Default workflow file not found: {wf_path}")
            raise FileNotFoundError(f"Default workflow file not found: {wf_path}")
        wf_base = load_workflow(wf_path)
        results = []

        for entry in entries:
            run_id = entry.get("name") or str(uuid.uuid4())
            inp = entry.get("inputs", {})
            pos = inp.get("prompt", "")
            neg = inp.get("negative_prompt", "")
            seed = int(inp.get("seed", random.randrange(0, 2**32)))
            guidance = float(inp.get("guidance_scale", 3.7))
            steps = int(inp.get("num_inference_steps", inp.get("num_steps", 33)))
            width = int(inp.get("width", 896))
            height = int(inp.get("height", 1152))
            strength = float(inp.get("strength", 1.0))
            scheduler = inp.get("scheduler")

            logging.info(f"=== run {run_id} ===")
            wf = copy.deepcopy(wf_base)
            wf = inject_prompts_and_images(wf, pos, neg, images=input_images)
            wf = inject_parameters(
                wf,
                seed=seed,
                guidance_scale=guidance,
                num_steps=steps,
                width=width,
                height=height,
                strength=strength,
                scheduler=scheduler,
            )
            # if bypass_reactor:
            wf = strip_reactor_nodes(wf)
            # if bypass_upscale_node:
            #     wf = bypass_upscale(wf)
            #     out_node = "230"
            if bypass_dfix_node:
                wf = bypass_dfix(wf)
                out_node = "230"
            else:
                out_node = "230"

            pid = queue_workflow(wf, host, [out_node])
            imgs_info = await_completion(pid, host, poll_interval, timeout)

            dest = Path("ComfyUI/output") / run_id
            download_outputs(imgs_info, host, str(dest))

            # Collect outputs
            for img_path in sorted(dest.glob("*.png")):
                fp = str(img_path)
                if do_cfg:
                    url = self.upload_to_digitalocean_spaces(
                        fp,
                        do_cfg.get("bucket_prefix", ""),
                        do_cfg.get("region", "sfo3"),
                        do_cfg.get("access_key_id", ""),
                        do_cfg.get("secret_access_key", ""),
                    )
                    if url:
                        results.append({"name": run_id, "url": url})
                    else:
                        results.append(Image.from_path(fp))
                else:
                    results.append(Image.from_path(fp))

        if not results:
            raise RuntimeError("No outputs generated")

        logging.info(f"Returning {len(results)} results")
        return Output(results=results)
