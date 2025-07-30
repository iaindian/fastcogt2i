#!/usr/bin/env python3
import argparse
import json
import logging
import time
import copy
from pathlib import Path
from urllib.parse import urlencode
import requests


def load_prompts(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_input_images(input_dir):
    p = Path(input_dir)
    imgs = sorted(x.name for x in p.iterdir()
                  if x.is_file() and x.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    if not imgs:
        raise FileNotFoundError(f"No images in {input_dir}")
    return imgs


def upload_images(images, input_dir, host):
    url = f"{host.rstrip('/')}/upload/image"
    for name in images:
        fp = Path(input_dir) / name
        with open(fp, 'rb') as f:
            files = {'image': (name, f, 'application/octet-stream')}
            data = {'type': 'input', 'overwrite': 'true'}
            resp = requests.post(url, files=files, data=data)
        resp.raise_for_status()
        logging.info(f"Uploaded {name}")


def load_workflow(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def inject_prompts_and_images(workflow, pos, neg, images):
    img_idx = 0
    for node_id, node in workflow.items():
        inputs = node.setdefault('inputs', {})
        if node_id == "364":
            logging.info(f"using positive prompt: {pos}")
            inputs['wildcard_text'] = pos
        elif node_id == "366":
            inputs['wildcard_text'] = neg
            logging.info(f"using negative prompt: {neg}")
        elif node.get('class_type') == 'LoadImage' and img_idx < len(images):
            inputs['image'] = images[img_idx]
            logging.debug(f"→ LoadImage node {node_id}: path ← {images[img_idx]}")
            img_idx += 1
    return workflow


def inject_parameters(workflow, seed=None, guidance_scale=None, num_steps=None,
                      width=None, height=None, strength=None, scheduler=None):
    for node in workflow.values():
        inputs = node.setdefault('inputs', {})
        ctype = node.get('class_type')
        if seed is not None and ctype == 'RandomNoise':
            inputs['noise_seed'] = seed
        if width is not None and height is not None and ctype == 'EmptyLatentImage':
            inputs['width'] = width
            inputs['height'] = height
        if strength is not None and ctype == 'ControlNetApplyAdvanced':
            inputs['strength'] = strength
    return workflow


def strip_reactor_nodes(workflow, reactor_id="271"):
    if reactor_id in workflow:
        old = workflow[reactor_id]["inputs"].get("enabled")
        workflow[reactor_id]["inputs"]["enabled"] = False
        logging.info(f"Bypassed reactor node {reactor_id}")
        logging.info(f"Reactor {reactor_id} enabled was {old}, now set to False")
    else:
        logging.warning(f"Reactor node {reactor_id} not found")
    return workflow

def enable_anal_boost(workflow, anal_id="375"):
    if anal_id in workflow:
        old = workflow[anal_id]["inputs"].get("switch_2")
        workflow[anal_id]["inputs"]["switch_2"] = "On"
        logging.info(f"Enabled anal booster node {anal_id}")
        logging.info(f"{anal_id} enabled was {old}, now set to On")
    else:
        logging.warning(f"node {anal_id} not found")
    return workflow

    
def remove_Image_WithoutDfix(workflow: dict) -> dict:
    """
    Remove any nodes whose IDs are listed in the internal bypass_list, this method removes the output image by flux
    """
    bypass_list = ["230"]
    # Create a new dict without those keys
    return {nid: node for nid, node in workflow.items() if nid not in bypass_list}
    


def strip_ttp_nodes(workflow, cutoff_node="230"):
    """
    Prune any nodes that depend on the given cutoff_node, effectively bypassing
    TTP/upscaling steps downstream of the flux output node.
    """
    deps = {}
    for nid, node in workflow.items():
        for inp in node.get('inputs', {}).values():
            if isinstance(inp, list) and inp and isinstance(inp[0], str):
                deps.setdefault(inp[0], []).append(nid)
    to_remove = set()
    queue = [cutoff_node]
    while queue:
        curr = queue.pop(0)
        for child in deps.get(curr, []):
            if child not in to_remove:
                to_remove.add(child)
                queue.append(child)
    for nid in to_remove:
        workflow.pop(nid, None)
    logging.info(f"Removed TTP nodes downstream of {cutoff_node}: {sorted(to_remove)}")
    return workflow

def clear_and_interrupt(host: str):
    """
    Tell ComfyUI to drop any pending jobs and kill any in-flight run.
    """
    # 1) clear the queue
    resp = requests.post(f"{host.rstrip('/')}/queue", json={"clear": True})
    resp.raise_for_status()
    logging.info("Cleared ComfyUI queue")

    # 2) interrupt any running workflow
    resp = requests.post(f"{host.rstrip('/')}/interrupt")
    resp.raise_for_status()
    logging.info("Interrupted ComfyUI execution")

def bypass_upscale(workflow: dict) -> dict:
    """
    Remove any nodes whose IDs are listed in the internal bypass_list.
    """
    bypass_list = ["245", "247", "264"]
    # Create a new dict without those keys
    return {nid: node for nid, node in workflow.items() if nid not in bypass_list}

def bypass_dfix(workflow: dict) -> dict:
    """
    Remove any nodes whose IDs are listed in the internal bypass_list.
    """
    bypass_list = ["307","308","309","310","311","312","313","314","315","316","317","318","320","321","322","323","324","325","326","327","328","329","330","332","333","334","337"]
    # Create a new dict without those keys
    return {nid: node for nid, node in workflow.items() if nid not in bypass_list}

def queue_workflow(workflow, host, outputs):
    url = f"{host.rstrip('/')}/prompt"
    payload = {'prompt': workflow, 'outputs': outputs}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    pid = data.get('prompt_id') or data.get('id')
    if not pid:
        raise RuntimeError(f"No prompt_id returned: {resp.text}")
    logging.info(f"Queued, prompt_id={pid}")
    return pid


def await_completion(prompt_id, host, interval, timeout):
    url = f"{host.rstrip('/')}/history/{prompt_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(url)
        resp.raise_for_status()
        outputs = resp.json().get(str(prompt_id), {}).get('outputs', {})
        images = [img for node in outputs.values() for img in node.get('images', [])]
        if images:
            logging.info(f"Got {len(images)} output images")
            return images
        time.sleep(interval)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def download_outputs(images, host, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for img in images:
        params = {
            'filename': img['filename'],
            'subfolder': img.get('subfolder', ''),
            'type': img.get('type', 'output'),
        }
        url = f"{host.rstrip('/')}/view?{urlencode(params)}"
        r = requests.get(url)
        r.raise_for_status()
        target = Path(out_dir) / img['filename']
        target.write_bytes(r.content)
        logging.info(f"Saved {target}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-json",    required=False,
                   default="workflow_api/face-match-4-17-api.json",
                   help="Path to ComfyUI workflow JSON (defaults to ./workflow_api/face-match-4-17-api.json)")
    p.add_argument("--prompt-file", required=True,
                   help="JSON array of {id,positive,negative} objects")
    p.add_argument("--input-dir",   default="input",
                   help="Directory of input images to upload")
    p.add_argument("--output-dir",  default="comfyui/output",
                   help="Base directory for output images")
    p.add_argument("--host",        default="http://127.0.0.1:8188",
                   help="ComfyUI server URL")
    p.add_argument("--bypass-reactor", action="store_true",
                   help="Skip any ReActor nodes in the workflow")
    p.add_argument("--bypass-upscale", action="store_true",
                   help="Bypass upscaling/TTP steps after flux output node")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="Seconds between history polls")
    p.add_argument("--timeout",       type=float, default=300.0,
                   help="Timeout for prompt completion (s)")
    p.add_argument("--log-level",     default="INFO",
                   help="Logging level")
    # Parameter injections
    p.add_argument("--seed",           type=int,   help="Seed for random noise")
    p.add_argument("--guidance-scale", dest="guidance_scale", type=float,
                   help="CFG guidance scale")
    p.add_argument("--num-steps",      dest="num_steps",      type=int,
                   help="Number of steps for BasicScheduler nodes")
    p.add_argument("--width",          type=int,   help="Width for EmptyLatentImage")
    p.add_argument("--height",         type=int,   help="Height for EmptyLatentImage")
    p.add_argument("--strength",       type=float, help="ControlNetApplyAdvanced strength")
    p.add_argument("--scheduler",      type=str,   help="Scheduler for BasicScheduler nodes")
    p.add_argument("--output-node",    choices=["230","264"], default="230",
                   help="Node ID to collect output from")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")

    prompts = load_prompts(args.prompt_file)
    if not prompts:
        raise RuntimeError("No prompts found")

    imgs = find_input_images(args.input_dir)
    upload_images(imgs, args.input_dir, args.host)

    base_wf = load_workflow(args.api_json)

    for entry in prompts:
        run_id = entry['id']
        pos, neg = entry['positive'], entry['negative']
        logging.info(f"=== Running prompt id={run_id} ===")
        logging.info(f"=== Running prompt id={pos} ===")

        wf = copy.deepcopy(base_wf)
        wf = inject_prompts_and_images(wf, pos, neg, imgs)
        wf = inject_parameters(
            wf,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            width=args.width,
            height=args.height,
            strength=args.strength,
            scheduler=args.scheduler
        )
        if args.bypass_reactor:
            wf = strip_reactor_nodes(wf)
        if args.bypass_upscale:
            wf = bypass_upscale(wf)
            out_node = "230"
        else:
            out_node = args.output_node

        pid = queue_workflow(wf, args.host, [out_node])
        images_info = await_completion(pid, args.host,
                                       args.poll_interval, args.timeout)

        out_dir = Path(args.output_dir) / run_id
        download_outputs(images_info, args.host, str(out_dir))

    logging.info("All runs complete.")


if __name__ == "__main__":
    main()
