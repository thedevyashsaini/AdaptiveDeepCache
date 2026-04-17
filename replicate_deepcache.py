"""
DeepCache Paper Replication Script
===================================
Paper: "DeepCache: Accelerating Diffusion Models for Free" (CVPR 2024)
       Ma, Fang & Wang - National University of Singapore

This script replicates the core result from the paper:
  - Stable Diffusion v1.5 speedup via DeepCache
  - Side-by-side visual comparison (baseline vs DeepCache)
  - Timing measurements showing ~2x speedup
  - Multiple prompts to demonstrate generality

Hardware target: NVIDIA RTX 3050 Laptop GPU (4GB VRAM)
  -> Uses FP16 + attention slicing to fit in VRAM

Usage:
  .venv\\Scripts\\python.exe replicate_deepcache.py
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torchvision.utils import save_image

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("replication_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ── HuggingFace auth ──────────────────────────────────────────────────────────
def setup_hf_auth():
    """Load HF_TOKEN from .env file if present."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()
    token = os.environ.get("HF_TOKEN")
    if token and token.startswith("hf_"):
        logger.info("HF_TOKEN loaded successfully.")
        return token
    else:
        logger.warning(
            "No HF_TOKEN found. Model download may fail if the model is gated.\n"
            "  -> Copy .env.example to .env and paste your token.\n"
            "  -> Get a token at https://huggingface.co/settings/tokens"
        )
        return None


# ── Seed utility ───────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ── System info ────────────────────────────────────────────────────────────────
def log_system_info():
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"PyTorch version : {torch.__version__}")
    logger.info(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device     : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"VRAM            : {props.total_memory / 1e9:.2f} GB")
        logger.info(f"Compute cap.    : {props.major}.{props.minor}")
    logger.info("=" * 60)


# ── VRAM-safe pipeline loader ─────────────────────────────────────────────────
def load_pipeline(model_id: str, token=None):
    """Load SD 1.5 pipeline with memory optimizations for 4GB GPU."""
    from diffusers import StableDiffusionPipeline

    logger.info(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # saves ~300MB VRAM
        requires_safety_checker=False,
        token=token,
    )
    pipe = pipe.to("cuda:0")

    # Memory optimizations for 4GB VRAM
    pipe.enable_attention_slicing()
    logger.info("Attention slicing enabled (VRAM optimization).")

    return pipe


# ── Run baseline (standard SD 1.5) ────────────────────────────────────────────
def run_baseline(pipe, prompt: str, seed: int, num_steps: int = 50):
    """Run the standard Stable Diffusion pipeline."""
    set_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = pipe(
        prompt,
        num_inference_steps=num_steps,
        output_type="pt",
    ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output, elapsed


# ── Run DeepCache ──────────────────────────────────────────────────────────────
def run_deepcache(
    pipe,
    prompt: str,
    seed: int,
    num_steps: int = 50,
    cache_interval: int = 3,
    cache_branch_id: int = 0,
):
    """Run SD pipeline with DeepCache acceleration."""
    from DeepCache import DeepCacheSDHelper

    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
        cache_interval=cache_interval,
        cache_branch_id=cache_branch_id,
    )
    helper.enable()

    set_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = pipe(
        prompt,
        num_inference_steps=num_steps,
        output_type="pt",
    ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    helper.disable()
    return output, elapsed


# ── Main replication ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DeepCache Paper Replication")
    parser.add_argument(
        "--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--cache_interval",
        type=int,
        default=3,
        help="Cache interval N (paper uses 3 and 5)",
    )
    parser.add_argument(
        "--cache_branch_id",
        type=int,
        default=0,
        help="Skip branch (0 = shallowest, paper default)",
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_warmup", action="store_true")
    args = parser.parse_args()

    # Setup
    setup_hf_auth()
    log_system_info()
    os.makedirs(args.output_dir, exist_ok=True)

    # Prompts from the paper / README demos
    prompts = [
        "a photo of an astronaut on a moon",
        "a beautiful castle beside a waterfall in spring, painting",
        "a golden retriever playing in the snow, professional photo",
        "a futuristic city skyline at sunset, digital art",
    ]

    model_id = args.model
    token = os.environ.get("HF_TOKEN")

    # Load pipeline
    pipe = load_pipeline(model_id, token=token)

    # ── Warmup ─────────────────────────────────────────────────────────────
    if not args.skip_warmup:
        logger.info("Warming up GPU (1 inference pass)...")
        set_seed(0)
        _ = pipe("warmup", num_inference_steps=5, output_type="pt")
        torch.cuda.empty_cache()
        logger.info("Warmup complete.")

    # ── Run experiments ────────────────────────────────────────────────────
    results = []
    all_images = []

    logger.info("")
    logger.info("=" * 60)
    logger.info("REPLICATION EXPERIMENT: Stable Diffusion v1.5 + DeepCache")
    logger.info(
        f"  Steps: {args.steps}  |  Cache interval: {args.cache_interval}  |  Branch: {args.cache_branch_id}"
    )
    logger.info("=" * 60)

    for i, prompt in enumerate(prompts):
        logger.info("")
        logger.info(f'─── Prompt {i + 1}/{len(prompts)}: "{prompt}" ───')

        # Baseline
        logger.info("  Running baseline (standard SD 1.5)...")
        torch.cuda.empty_cache()
        baseline_img, baseline_time = run_baseline(pipe, prompt, args.seed, args.steps)
        logger.info(f"  Baseline: {baseline_time:.2f}s")

        # DeepCache
        logger.info(
            f"  Running DeepCache (interval={args.cache_interval}, branch={args.cache_branch_id})..."
        )
        torch.cuda.empty_cache()
        dc_img, dc_time = run_deepcache(
            pipe,
            prompt,
            args.seed,
            args.steps,
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        logger.info(f"  DeepCache: {dc_time:.2f}s")

        speedup = baseline_time / dc_time
        logger.info(f"  >>> Speedup: {speedup:.2f}x <<<")

        # Save individual side-by-side
        save_image(
            [baseline_img, dc_img],
            os.path.join(args.output_dir, f"comparison_{i + 1}.png"),
            nrow=2,
            padding=4,
            pad_value=1.0,
        )

        # Save individual images
        save_image(
            [baseline_img], os.path.join(args.output_dir, f"baseline_{i + 1}.png")
        )
        save_image([dc_img], os.path.join(args.output_dir, f"deepcache_{i + 1}.png"))

        all_images.extend([baseline_img, dc_img])
        results.append(
            {
                "prompt": prompt,
                "baseline_time_s": round(baseline_time, 3),
                "deepcache_time_s": round(dc_time, 3),
                "speedup": round(speedup, 3),
                "steps": args.steps,
                "cache_interval": args.cache_interval,
                "cache_branch_id": args.cache_branch_id,
                "seed": args.seed,
            }
        )

    # ── Save combined grid ─────────────────────────────────────────────────
    save_image(
        all_images,
        os.path.join(args.output_dir, "all_comparisons_grid.png"),
        nrow=2,
        padding=6,
        pad_value=1.0,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    avg_baseline = np.mean([r["baseline_time_s"] for r in results])
    avg_dc = np.mean([r["deepcache_time_s"] for r in results])
    avg_speedup = avg_baseline / avg_dc

    logger.info("")
    logger.info("=" * 60)
    logger.info("REPLICATION RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model          : {model_id}")
    logger.info(f"Inference steps: {args.steps}")
    logger.info(f"Cache interval : {args.cache_interval}")
    logger.info(f"Cache branch   : {args.cache_branch_id}")
    logger.info(f"Seed           : {args.seed}")
    logger.info(f"Num prompts    : {len(prompts)}")
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"  [{r['speedup']:.2f}x] {r['baseline_time_s']:.2f}s -> {r['deepcache_time_s']:.2f}s  "
            f'| "{r["prompt"][:50]}..."'
        )
    logger.info("-" * 60)
    logger.info(f"  Average baseline  : {avg_baseline:.2f}s")
    logger.info(f"  Average DeepCache : {avg_dc:.2f}s")
    logger.info(f"  Average speedup   : {avg_speedup:.2f}x")
    logger.info("")
    logger.info(f"  Paper reports ~2.3x speedup for SD 1.5 with PLMS, 50 steps.")
    logger.info(
        f"  (Our config: PNDM scheduler, {args.steps} steps, cache_interval={args.cache_interval})"
    )
    logger.info("=" * 60)

    # Save JSON results for documentation
    report = {
        "experiment": "DeepCache Replication - Stable Diffusion v1.5",
        "paper": "DeepCache: Accelerating Diffusion Models for Free (CVPR 2024)",
        "date": datetime.now().isoformat(),
        "system": {
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if torch.cuda.is_available()
            else 0,
            "pytorch": torch.__version__,
        },
        "config": {
            "model": model_id,
            "steps": args.steps,
            "cache_interval": args.cache_interval,
            "cache_branch_id": args.cache_branch_id,
            "seed": args.seed,
        },
        "results": results,
        "summary": {
            "avg_baseline_s": round(avg_baseline, 3),
            "avg_deepcache_s": round(avg_dc, 3),
            "avg_speedup": round(avg_speedup, 3),
            "paper_reported_speedup": "~2.3x (PLMS, 50 steps)",
        },
    }
    json_path = os.path.join(args.output_dir, "replication_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full report saved to: {json_path}")
    logger.info(f"Images saved to: {args.output_dir}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
