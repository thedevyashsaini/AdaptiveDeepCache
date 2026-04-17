import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x)


def plot_fixed_vs_adaptive(summary_rows: list[dict], out_dir: Path):
    rows = [
        r
        for r in summary_rows
        if r["scope"] == "default" and r["mode"] in {"fixed", "adaptive"}
    ]
    labels = [r["policy_name"] for r in rows]
    speed = [f(r["avg_speedup_vs_baseline"]) for r in rows]
    l2 = [f(r["avg_l2_vs_baseline"]) for r in rows]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(x, speed)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_title("Speedup vs Baseline")
    axes[0].set_ylabel("x")

    axes[1].bar(x, l2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_title("L2 vs Baseline")
    axes[1].set_ylabel("L2 (lower better)")

    fig.tight_layout()
    fig.savefig((out_dir / "fixed_vs_adaptive.png").as_posix(), dpi=200)
    plt.close(fig)


def plot_refresh_ratio(summary_rows: list[dict], out_dir: Path):
    rows = [
        r for r in summary_rows if r["scope"] == "default" and r["mode"] == "adaptive"
    ]
    labels = [r["policy_name"] for r in rows]
    refresh_ratio = [f(r["avg_refresh_ratio"]) for r in rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(8.6, 4.8))
    plt.bar(x, refresh_ratio)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Refresh Ratio")
    plt.title("Adaptive Policy Refresh Ratio")
    plt.tight_layout()
    plt.savefig((out_dir / "adaptive_refresh_ratio.png").as_posix(), dpi=200)
    plt.close()


def plot_layer_sensitivity(summary_rows: list[dict], out_dir: Path):
    rows = [
        r
        for r in summary_rows
        if r["scope"] == "layer_sweep" and r["mode"] == "adaptive"
    ]
    rows = sorted(rows, key=lambda x: int(x["cache_branch_id"]))
    branches = [int(r["cache_branch_id"]) for r in rows]
    speed = [f(r["avg_speedup_vs_baseline"]) for r in rows]
    l2 = [f(r["avg_l2_vs_baseline"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.0))
    ax1.plot(branches, speed, marker="o", label="Speedup")
    ax1.set_xlabel("Cache Branch ID")
    ax1.set_ylabel("Speedup (x)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(branches, l2, marker="s", linestyle="--", label="L2")
    ax2.set_ylabel("L2 (lower better)")

    ax1.set_title("Adaptive Layer Sensitivity")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig((out_dir / "adaptive_layer_sensitivity.png").as_posix(), dpi=200)
    plt.close(fig)


def plot_prompt_wise(adaptive_detail_rows: list[dict], out_dir: Path):
    fixed_rows = [
        r
        for r in adaptive_detail_rows
        if r["scope"] == "default" and r["mode"] == "fixed"
    ]
    adaptive_rows = [
        r
        for r in adaptive_detail_rows
        if r["scope"] == "default" and r["mode"] == "adaptive"
    ]
    policy_names = sorted({r["policy_name"] for r in adaptive_rows})
    prompt_ids = sorted({int(r["prompt_index"]) for r in fixed_rows})

    x = np.arange(len(prompt_ids))
    width = 0.8 / (len(policy_names) + 1)

    plt.figure(figsize=(11.0, 5.0))

    fixed_vals = []
    for p in prompt_ids:
        vals = [
            f(r["speedup_vs_baseline"])
            for r in fixed_rows
            if int(r["prompt_index"]) == p
        ]
        fixed_vals.append(np.mean(vals) if vals else 0.0)
    plt.bar(x - 0.4 + width / 2, fixed_vals, width=width, label="fixed_k3")

    for idx, policy in enumerate(policy_names):
        vals = []
        for p in prompt_ids:
            arr = [
                f(r["speedup_vs_baseline"])
                for r in adaptive_rows
                if int(r["prompt_index"]) == p and r["policy_name"] == policy
            ]
            vals.append(np.mean(arr) if arr else 0.0)
        plt.bar(
            x - 0.4 + width * (idx + 1) + width / 2, vals, width=width, label=policy
        )

    plt.xticks(x, [f"P{p}" for p in prompt_ids])
    plt.ylabel("Speedup vs Baseline")
    plt.title("Prompt-wise Speedup: Fixed vs Adaptive")
    plt.legend()
    plt.tight_layout()
    plt.savefig((out_dir / "prompt_wise_speedup_compare.png").as_posix(), dpi=200)
    plt.close()


def plot_merged_current_vs_adaptive(
    current_summary_rows: list[dict], adaptive_summary_rows: list[dict], out_dir: Path
):
    current_fixed = [
        r
        for r in current_summary_rows
        if r["mode"] == "deepcache_fixed"
        and int(r["cache_interval"]) == 3
        and int(r["cache_branch_id"]) == 0
    ]
    adaptive_fixed = [
        r
        for r in adaptive_summary_rows
        if r["mode"] == "fixed" and r["scope"] == "default"
    ]
    adaptive_best = [
        r
        for r in adaptive_summary_rows
        if r["mode"] == "adaptive" and r["scope"] == "default"
    ]

    if not current_fixed or not adaptive_fixed or not adaptive_best:
        return

    best_adaptive = max(adaptive_best, key=lambda x: f(x["avg_speedup_vs_baseline"]))

    labels = [
        "Current Fixed",
        "Adaptive Run Fixed",
        f"Best Adaptive ({best_adaptive['policy_name']})",
    ]
    speed = [
        f(current_fixed[0]["avg_speedup_vs_baseline"]),
        f(adaptive_fixed[0]["avg_speedup_vs_baseline"]),
        f(best_adaptive["avg_speedup_vs_baseline"]),
    ]
    l2 = [
        f(current_fixed[0]["avg_l2_vs_baseline"]),
        f(adaptive_fixed[0]["avg_l2_vs_baseline"]),
        f(best_adaptive["avg_l2_vs_baseline"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].bar(labels, speed)
    axes[0].set_ylabel("Speedup (x)")
    axes[0].set_title("Merged: Speedup")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, l2)
    axes[1].set_ylabel("L2")
    axes[1].set_title("Merged: Quality Proxy")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig((out_dir / "merged_current_vs_adaptive.png").as_posix(), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot adaptive DeepCache comparisons")
    parser.add_argument(
        "--adaptive_tables", type=str, default="results/benchmarks/adaptive/tables"
    )
    parser.add_argument(
        "--current_tables", type=str, default="results/benchmarks/current/tables"
    )
    parser.add_argument(
        "--output_plots", type=str, default="results/benchmarks/adaptive/plots"
    )
    args = parser.parse_args()

    adaptive_tables = Path(args.adaptive_tables)
    current_tables = Path(args.current_tables)
    out_dir = Path(args.output_plots)
    out_dir.mkdir(parents=True, exist_ok=True)

    adaptive_summary = read_csv(adaptive_tables / "adaptive_config_summary.csv")
    adaptive_detail = read_csv(adaptive_tables / "adaptive_per_prompt_metrics.csv")
    current_summary = (
        read_csv(current_tables / "config_summary.csv")
        if current_tables.exists()
        else []
    )

    plot_fixed_vs_adaptive(adaptive_summary, out_dir)
    plot_refresh_ratio(adaptive_summary, out_dir)
    plot_layer_sensitivity(adaptive_summary, out_dir)
    plot_prompt_wise(adaptive_detail, out_dir)
    if current_summary:
        plot_merged_current_vs_adaptive(current_summary, adaptive_summary, out_dir)


if __name__ == "__main__":
    main()
