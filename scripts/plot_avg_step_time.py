#!/usr/bin/env python3
"""Simple bar chart: average warm ML-step time per library configuration.

One horizontal bar per configuration (sub-directory of the results dir that
contains a cmi_phase_summary.csv). The bar value is the Rank_Mean of the
solver_step_ml_steady region, i.e. the average duration of one steady ML step
averaged over all ranks (the same metric the totals in config_comparison are
built from). Bars are sorted fastest -> slowest.

Usage:
    python3 plot_avg_step_time.py <results_dir> [--output png_path]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_REGION = "solver_step_ml_steady"
BAR_COLORS = {
    "smartsim": "#5B8FF9",
    "aix": "#F6903D",
    "phydll": "#61C0BF",
}


def load_step_mean(config_dir: Path):
    summary = config_dir / "cmi_phase_summary.csv"
    if not summary.exists():
        return None
    with summary.open() as stream:
        for row in csv.DictReader(stream):
            if row.get("Region") == METRIC_REGION:
                try:
                    return float(row["Rank_Mean (ms)"])
                except (KeyError, ValueError):
                    return None
    return None


def collect_params(config_dirs) -> dict:
    meta: dict = {}
    for config_dir in config_dirs:
        meta_file = config_dir / "run_metadata.json"
        if not meta_file.exists():
            continue
        try:
            with meta_file.open() as stream:
                data = json.load(stream)
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("model", "resolution", "batch_size", "steady_steps"):
            if data.get(key) is not None and key not in meta:
                meta[key] = data[key]
    return meta


def title_from_meta(meta: dict) -> str:
    parts = []
    if meta.get("model"):
        parts.append(str(meta["model"]))
    if meta.get("resolution"):
        parts.append(f"{meta['resolution']}")
    if meta.get("batch_size"):
        parts.append(f"batch {meta['batch_size']}")
    if meta.get("steady_steps"):
        parts.append(f"{int(meta['steady_steps'])} steady ML steps")
    suffix = f" ({', '.join(parts)})" if parts else ""
    return f"Average warm ML-step time per configuration{suffix}"


def bar_color(label: str) -> str:
    lowered = label.lower()
    if "phydll" in lowered:
        return BAR_COLORS["phydll"]
    if "aix" in lowered:
        return BAR_COLORS["aix"]
    return BAR_COLORS["smartsim"]


def render(configs, output_path: Path, meta: dict):
    configs = sorted(configs, key=lambda c: c["mean"])  # fastest last -> top bar
    labels = [c["label"] for c in configs]
    values = [c["mean"] for c in configs]
    colors = [bar_color(label) for label in labels]

    fig, axis = plt.subplots(figsize=(11, 0.42 * len(configs) + 2.0))
    y = range(len(configs))
    axis.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    axis.set_yticks(list(y))
    axis.set_yticklabels(labels)
    axis.invert_yaxis()  # fastest at the top
    axis.set_xlabel("Average warm ML-step time [ms]")
    axis.set_title(title_from_meta(meta), fontsize=11)
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    axis.set_axisbelow(True)
    xmax = max(values)
    axis.set_xlim(0, xmax * 1.12)
    for yi, value in zip(y, values):
        axis.text(value + xmax * 0.012, yi, f"{value:.1f}",
                  va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing one sub-directory per configuration, "
                             "each with a cmi_phase_summary.csv.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: <results_dir>/avg_warm_step_time.png).")
    args = parser.parse_args()

    configs = []
    for config_dir in sorted(p for p in args.results_dir.iterdir() if p.is_dir()):
        step_mean = load_step_mean(config_dir)
        if step_mean is None:
            print(f"[WARN] no {METRIC_REGION} summary in {config_dir.name}; skipping")
            continue
        print(f"  {config_dir.name}: avg warm step = {step_mean:.1f} ms")
        configs.append({"label": config_dir.name, "mean": step_mean, "dir": config_dir})

    if not configs:
        raise SystemExit("No configuration with a solver_step_ml_steady summary found.")

    output = args.output or (args.results_dir / "avg_warm_step_time.png")
    meta = collect_params([c["dir"] for c in configs])
    render(configs, output, meta)

    table_path = output.with_suffix(".csv")
    with table_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Config", "AvgWarmStep_Mean_Ranks_ms"])
        for config in sorted(configs, key=lambda c: c["mean"]):
            writer.writerow([config["label"], f"{config['mean']:.3f}"])

    print(f"Wrote {output} and {table_path}")


if __name__ == "__main__":
    main()
