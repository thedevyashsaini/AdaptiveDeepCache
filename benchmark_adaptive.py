import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from DeepCache import DeepCacheSDHelper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_hf_auth() -> None:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_dirs(base_dir: Path) -> dict:
    paths = {
        "root": base_dir,
        "raw": base_dir / "raw",
        "tables": base_dir / "tables",
        "plots": base_dir / "plots",
        "images": base_dir / "images",
        "step_logs": base_dir / "step_logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).reshape(-1)
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def tensor_psnr(a: torch.Tensor, b: torch.Tensor, max_value: float = 1.0) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def load_pipeline(model_id: str):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda:0")
    pipe.enable_attention_slicing()
    return pipe


def run_pipe(pipe, prompt: str, steps: int) -> tuple[torch.Tensor, float]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    img = pipe(prompt, num_inference_steps=steps, output_type="pt").images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return img.detach().cpu(), elapsed


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_with_helper(
    pipe,
    prompt: str,
    seed: int,
    steps: int,
    helper_params: dict,
) -> tuple[torch.Tensor, float, list[dict]]:
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(**helper_params)
    helper.enable()
    try:
        set_seed(seed)
        image, elapsed = run_pipe(pipe, prompt, steps)
        step_logs = helper.get_step_logs()
    finally:
        helper.disable()
    return image, elapsed, step_logs


def compute_step_summary(step_logs: list[dict]) -> dict:
    refresh_count = sum(1 for x in step_logs if x.get("refresh"))
    reuse_count = sum(1 for x in step_logs if x.get("reuse"))
    deltas = [x["delta_latent"] for x in step_logs if x.get("delta_latent") is not None]
    delta_scores = [
        x["delta_score"] for x in step_logs if x.get("delta_score") is not None
    ]
    unet_times = [
        x["step_unet_time_s"]
        for x in step_logs
        if x.get("step_unet_time_s") is not None
    ]
    return {
        "refresh_count": refresh_count,
        "reuse_count": reuse_count,
        "refresh_ratio": float(refresh_count / max(refresh_count + reuse_count, 1)),
        "avg_delta_latent": float(np.mean(deltas)) if deltas else None,
        "avg_delta_score": float(np.mean(delta_scores)) if delta_scores else None,
        "avg_step_unet_time_s": float(np.mean(unet_times)) if unet_times else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Adaptive DeepCache")
    parser.add_argument("--config", type=str, default="benchmark_config_adaptive.json")
    parser.add_argument(
        "--output_root", type=str, default="results/benchmarks/adaptive"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    setup_hf_auth()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths = ensure_dirs(Path(args.output_root))

    model_id = config["model_id"]
    prompts = config["prompts"]
    seeds = config["seeds"]
    steps = int(config["steps"])
    repeats = int(config.get("repeats", 1))
    warmup_steps = int(config.get("warmup_steps", 5))
    fixed_compare = config["fixed_compare"]
    adaptive_policies = config["adaptive_policies"]
    layer_sweep_branch_ids = config["layer_sweep_branch_ids"]

    logger.info("Loading pipeline: %s", model_id)
    pipe = load_pipeline(model_id)

    logger.info("Warmup run...")
    set_seed(0)
    _ = pipe("warmup", num_inference_steps=warmup_steps, output_type="pt")

    run_meta = {
        "experiment": config.get("experiment_name", "DeepCache Adaptive Benchmark"),
        "date": datetime.now().isoformat(),
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "pytorch": torch.__version__,
        },
        "config": config,
    }

    rows = []
    baseline_cache = {}

    logger.info("Running baseline...")
    for seed in seeds:
        for prompt_i, prompt in enumerate(prompts):
            for repeat_i in range(repeats):
                set_seed(seed)
                img, elapsed = run_pipe(pipe, prompt, steps)
                key = (prompt, seed, repeat_i)
                baseline_path = (
                    paths["images"]
                    / f"baseline_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.png"
                )
                save_image([img], baseline_path.as_posix())
                baseline_cache[key] = {"img": img, "runtime": elapsed}
                rows.append(
                    {
                        "mode": "baseline",
                        "scope": "default",
                        "policy_name": "baseline",
                        "prompt_index": prompt_i + 1,
                        "prompt": prompt,
                        "seed": seed,
                        "repeat": repeat_i + 1,
                        "steps": steps,
                        "cache_interval": 0,
                        "cache_branch_id": -1,
                        "runtime_s": round(elapsed, 6),
                        "speedup_vs_baseline": 1.0,
                        "l2_vs_baseline": 0.0,
                        "psnr_vs_baseline": 99.0,
                        "refresh_count": 0,
                        "reuse_count": 0,
                        "refresh_ratio": 0.0,
                        "avg_delta_latent": None,
                        "avg_delta_score": None,
                        "avg_step_unet_time_s": None,
                        "threshold_early": None,
                        "threshold_mid": None,
                        "threshold_late": None,
                        "early_ratio": None,
                        "mid_ratio": None,
                        "force_refresh_every": None,
                        "min_refresh_interval": None,
                        "ema_alpha": None,
                        "use_relative_delta": None,
                        "step_log_path": None,
                        "image_path": baseline_path.as_posix(),
                    }
                )

    logger.info("Running fixed comparison config...")
    fixed_params = {
        "cache_interval": int(fixed_compare["cache_interval"]),
        "cache_branch_id": int(fixed_compare["cache_branch_id"]),
        "adaptive": False,
    }
    for seed in seeds:
        for prompt_i, prompt in enumerate(prompts):
            for repeat_i in range(repeats):
                key = (prompt, seed, repeat_i)
                base = baseline_cache[key]
                image, elapsed, step_logs = run_with_helper(
                    pipe, prompt, seed, steps, fixed_params
                )
                speedup = base["runtime"] / elapsed
                l2 = tensor_l2(image, base["img"])
                psnr = tensor_psnr(image, base["img"])
                step_summary = compute_step_summary(step_logs)
                log_path = (
                    paths["step_logs"]
                    / f"fixed_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.json"
                )
                log_path.write_text(json.dumps(step_logs, indent=2), encoding="utf-8")

                image_path = (
                    paths["images"]
                    / f"fixed_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.png"
                )
                save_image([image], image_path.as_posix())

                rows.append(
                    {
                        "mode": "fixed",
                        "scope": "default",
                        "policy_name": "fixed_k3",
                        "prompt_index": prompt_i + 1,
                        "prompt": prompt,
                        "seed": seed,
                        "repeat": repeat_i + 1,
                        "steps": steps,
                        "cache_interval": fixed_params["cache_interval"],
                        "cache_branch_id": fixed_params["cache_branch_id"],
                        "runtime_s": round(elapsed, 6),
                        "speedup_vs_baseline": round(speedup, 6),
                        "l2_vs_baseline": round(l2, 8),
                        "psnr_vs_baseline": round(psnr, 6),
                        "refresh_count": step_summary["refresh_count"],
                        "reuse_count": step_summary["reuse_count"],
                        "refresh_ratio": round(step_summary["refresh_ratio"], 6),
                        "avg_delta_latent": None,
                        "avg_delta_score": None,
                        "avg_step_unet_time_s": round(
                            step_summary["avg_step_unet_time_s"], 8
                        )
                        if step_summary["avg_step_unet_time_s"] is not None
                        else None,
                        "threshold_early": None,
                        "threshold_mid": None,
                        "threshold_late": None,
                        "early_ratio": None,
                        "mid_ratio": None,
                        "force_refresh_every": None,
                        "min_refresh_interval": None,
                        "ema_alpha": None,
                        "use_relative_delta": None,
                        "step_log_path": log_path.as_posix(),
                        "image_path": image_path.as_posix(),
                    }
                )

    logger.info("Running adaptive policy default branch...")
    for policy in adaptive_policies:
        policy_name = policy["name"]
        params = {
            "cache_interval": fixed_params["cache_interval"],
            "cache_branch_id": fixed_params["cache_branch_id"],
            "adaptive": True,
            "threshold_early": policy["threshold_early"],
            "threshold_mid": policy["threshold_mid"],
            "threshold_late": policy["threshold_late"],
            "early_ratio": policy["early_ratio"],
            "mid_ratio": policy["mid_ratio"],
            "force_refresh_every": policy["force_refresh_every"],
            "min_refresh_interval": policy.get("min_refresh_interval", 1),
            "ema_alpha": policy.get("ema_alpha", 0.30),
            "use_relative_delta": policy.get("use_relative_delta", True),
        }
        for seed in seeds:
            for prompt_i, prompt in enumerate(prompts):
                for repeat_i in range(repeats):
                    key = (prompt, seed, repeat_i)
                    base = baseline_cache[key]
                    image, elapsed, step_logs = run_with_helper(
                        pipe, prompt, seed, steps, params
                    )
                    speedup = base["runtime"] / elapsed
                    l2 = tensor_l2(image, base["img"])
                    psnr = tensor_psnr(image, base["img"])
                    step_summary = compute_step_summary(step_logs)
                    log_path = (
                        paths["step_logs"]
                        / f"adaptive_{policy_name}_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.json"
                    )
                    log_path.write_text(
                        json.dumps(step_logs, indent=2), encoding="utf-8"
                    )

                    image_path = (
                        paths["images"]
                        / f"adaptive_{policy_name}_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.png"
                    )
                    save_image([image], image_path.as_posix())

                    rows.append(
                        {
                            "mode": "adaptive",
                            "scope": "default",
                            "policy_name": policy_name,
                            "prompt_index": prompt_i + 1,
                            "prompt": prompt,
                            "seed": seed,
                            "repeat": repeat_i + 1,
                            "steps": steps,
                            "cache_interval": fixed_params["cache_interval"],
                            "cache_branch_id": fixed_params["cache_branch_id"],
                            "runtime_s": round(elapsed, 6),
                            "speedup_vs_baseline": round(speedup, 6),
                            "l2_vs_baseline": round(l2, 8),
                            "psnr_vs_baseline": round(psnr, 6),
                            "refresh_count": step_summary["refresh_count"],
                            "reuse_count": step_summary["reuse_count"],
                            "refresh_ratio": round(step_summary["refresh_ratio"], 6),
                            "avg_delta_latent": round(
                                step_summary["avg_delta_latent"], 8
                            )
                            if step_summary["avg_delta_latent"] is not None
                            else None,
                            "avg_delta_score": round(step_summary["avg_delta_score"], 8)
                            if step_summary["avg_delta_score"] is not None
                            else None,
                            "avg_step_unet_time_s": round(
                                step_summary["avg_step_unet_time_s"], 8
                            )
                            if step_summary["avg_step_unet_time_s"] is not None
                            else None,
                            "threshold_early": policy["threshold_early"],
                            "threshold_mid": policy["threshold_mid"],
                            "threshold_late": policy["threshold_late"],
                            "early_ratio": policy["early_ratio"],
                            "mid_ratio": policy["mid_ratio"],
                            "force_refresh_every": policy["force_refresh_every"],
                            "min_refresh_interval": policy.get(
                                "min_refresh_interval", 1
                            ),
                            "ema_alpha": policy.get("ema_alpha", 0.30),
                            "use_relative_delta": policy.get(
                                "use_relative_delta", True
                            ),
                            "step_log_path": log_path.as_posix(),
                            "image_path": image_path.as_posix(),
                        }
                    )

    logger.info("Running layer sensitivity sweep for first adaptive policy...")
    layer_policy = adaptive_policies[0]
    for branch in layer_sweep_branch_ids:
        params = {
            "cache_interval": fixed_params["cache_interval"],
            "cache_branch_id": int(branch),
            "adaptive": True,
            "threshold_early": layer_policy["threshold_early"],
            "threshold_mid": layer_policy["threshold_mid"],
            "threshold_late": layer_policy["threshold_late"],
            "early_ratio": layer_policy["early_ratio"],
            "mid_ratio": layer_policy["mid_ratio"],
            "force_refresh_every": layer_policy["force_refresh_every"],
            "min_refresh_interval": layer_policy.get("min_refresh_interval", 1),
            "ema_alpha": layer_policy.get("ema_alpha", 0.30),
            "use_relative_delta": layer_policy.get("use_relative_delta", True),
        }
        for seed in seeds:
            for prompt_i, prompt in enumerate(prompts):
                for repeat_i in range(repeats):
                    key = (prompt, seed, repeat_i)
                    base = baseline_cache[key]
                    image, elapsed, step_logs = run_with_helper(
                        pipe, prompt, seed, steps, params
                    )
                    speedup = base["runtime"] / elapsed
                    l2 = tensor_l2(image, base["img"])
                    psnr = tensor_psnr(image, base["img"])
                    step_summary = compute_step_summary(step_logs)
                    log_path = paths["step_logs"] / (
                        f"adaptive_layer_{layer_policy['name']}_b{branch}_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.json"
                    )
                    log_path.write_text(
                        json.dumps(step_logs, indent=2), encoding="utf-8"
                    )

                    image_path = paths["images"] / (
                        f"adaptive_layer_{layer_policy['name']}_b{branch}_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.png"
                    )
                    save_image([image], image_path.as_posix())

                    rows.append(
                        {
                            "mode": "adaptive",
                            "scope": "layer_sweep",
                            "policy_name": layer_policy["name"],
                            "prompt_index": prompt_i + 1,
                            "prompt": prompt,
                            "seed": seed,
                            "repeat": repeat_i + 1,
                            "steps": steps,
                            "cache_interval": fixed_params["cache_interval"],
                            "cache_branch_id": branch,
                            "runtime_s": round(elapsed, 6),
                            "speedup_vs_baseline": round(speedup, 6),
                            "l2_vs_baseline": round(l2, 8),
                            "psnr_vs_baseline": round(psnr, 6),
                            "refresh_count": step_summary["refresh_count"],
                            "reuse_count": step_summary["reuse_count"],
                            "refresh_ratio": round(step_summary["refresh_ratio"], 6),
                            "avg_delta_latent": round(
                                step_summary["avg_delta_latent"], 8
                            )
                            if step_summary["avg_delta_latent"] is not None
                            else None,
                            "avg_delta_score": round(step_summary["avg_delta_score"], 8)
                            if step_summary["avg_delta_score"] is not None
                            else None,
                            "avg_step_unet_time_s": round(
                                step_summary["avg_step_unet_time_s"], 8
                            )
                            if step_summary["avg_step_unet_time_s"] is not None
                            else None,
                            "threshold_early": layer_policy["threshold_early"],
                            "threshold_mid": layer_policy["threshold_mid"],
                            "threshold_late": layer_policy["threshold_late"],
                            "early_ratio": layer_policy["early_ratio"],
                            "mid_ratio": layer_policy["mid_ratio"],
                            "force_refresh_every": layer_policy["force_refresh_every"],
                            "min_refresh_interval": layer_policy.get(
                                "min_refresh_interval", 1
                            ),
                            "ema_alpha": layer_policy.get("ema_alpha", 0.30),
                            "use_relative_delta": layer_policy.get(
                                "use_relative_delta", True
                            ),
                            "step_log_path": log_path.as_posix(),
                            "image_path": image_path.as_posix(),
                        }
                    )

    grouped = {}
    for row in rows:
        key = (
            row["mode"],
            row["scope"],
            row["policy_name"],
            row["cache_interval"],
            row["cache_branch_id"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (mode, scope, policy_name, interval, branch), group_rows in grouped.items():
        runtimes = np.array(
            [float(r["runtime_s"]) for r in group_rows], dtype=np.float64
        )
        speedups = np.array(
            [float(r["speedup_vs_baseline"]) for r in group_rows], dtype=np.float64
        )
        l2s = np.array(
            [float(r["l2_vs_baseline"]) for r in group_rows], dtype=np.float64
        )
        psnrs = np.array(
            [float(r["psnr_vs_baseline"]) for r in group_rows], dtype=np.float64
        )
        refresh_ratio = [
            float(r["refresh_ratio"])
            for r in group_rows
            if r["refresh_ratio"] is not None
        ]
        delta_scores = [
            float(r["avg_delta_score"])
            for r in group_rows
            if r.get("avg_delta_score") not in (None, "")
        ]
        summary.append(
            {
                "mode": mode,
                "scope": scope,
                "policy_name": policy_name,
                "cache_interval": interval,
                "cache_branch_id": branch,
                "num_samples": len(group_rows),
                "avg_runtime_s": round(float(runtimes.mean()), 6),
                "std_runtime_s": round(float(runtimes.std(ddof=0)), 6),
                "avg_speedup_vs_baseline": round(float(speedups.mean()), 6),
                "std_speedup_vs_baseline": round(float(speedups.std(ddof=0)), 6),
                "avg_l2_vs_baseline": round(float(l2s.mean()), 8),
                "avg_psnr_vs_baseline": round(float(psnrs.mean()), 6),
                "avg_refresh_ratio": round(float(np.mean(refresh_ratio)), 6)
                if refresh_ratio
                else None,
                "avg_delta_score": round(float(np.mean(delta_scores)), 8)
                if delta_scores
                else None,
            }
        )

    summary.sort(
        key=lambda x: (
            x["mode"],
            x["scope"],
            x["policy_name"],
            x["cache_interval"],
            x["cache_branch_id"],
        )
    )

    detail_fields = [
        "mode",
        "scope",
        "policy_name",
        "prompt_index",
        "prompt",
        "seed",
        "repeat",
        "steps",
        "cache_interval",
        "cache_branch_id",
        "runtime_s",
        "speedup_vs_baseline",
        "l2_vs_baseline",
        "psnr_vs_baseline",
        "refresh_count",
        "reuse_count",
        "refresh_ratio",
        "avg_delta_latent",
        "avg_delta_score",
        "avg_step_unet_time_s",
        "threshold_early",
        "threshold_mid",
        "threshold_late",
        "early_ratio",
        "mid_ratio",
        "force_refresh_every",
        "min_refresh_interval",
        "ema_alpha",
        "use_relative_delta",
        "step_log_path",
        "image_path",
    ]
    summary_fields = [
        "mode",
        "scope",
        "policy_name",
        "cache_interval",
        "cache_branch_id",
        "num_samples",
        "avg_runtime_s",
        "std_runtime_s",
        "avg_speedup_vs_baseline",
        "std_speedup_vs_baseline",
        "avg_l2_vs_baseline",
        "avg_psnr_vs_baseline",
        "avg_refresh_ratio",
        "avg_delta_score",
    ]

    write_csv(paths["tables"] / "adaptive_per_prompt_metrics.csv", rows, detail_fields)
    write_csv(paths["tables"] / "adaptive_config_summary.csv", summary, summary_fields)

    raw_payload = {
        "meta": run_meta,
        "per_prompt_metrics": rows,
        "config_summary": summary,
    }
    raw_path = paths["raw"] / "benchmark_adaptive_raw.json"
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    default_adaptive_rows = [
        x for x in summary if x["mode"] == "adaptive" and x["scope"] == "default"
    ]
    best_default_adaptive = max(
        default_adaptive_rows, key=lambda x: x["avg_speedup_vs_baseline"]
    )

    report = {
        "meta": run_meta,
        "highlights": {
            "fixed_default": fixed_compare,
            "best_adaptive_default": best_default_adaptive,
            "layer_policy": adaptive_policies[0]["name"],
        },
        "tables": {
            "detail_csv": (
                paths["tables"] / "adaptive_per_prompt_metrics.csv"
            ).as_posix(),
            "summary_csv": (paths["tables"] / "adaptive_config_summary.csv").as_posix(),
        },
        "raw_json": raw_path.as_posix(),
    }
    report_path = paths["raw"] / "benchmark_adaptive_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Adaptive benchmark complete: %s", report_path.as_posix())


if __name__ == "__main__":
    main()
