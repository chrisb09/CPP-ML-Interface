#!/usr/bin/env python3
"""Cross-configuration comparison of the steady-state ML-step phases.

Reads the per-config cmi_phase_summary.csv files produced by
analyze_cmi_scorep_profiles.py and renders stacked horizontal bars that
decompose one steady ML step (solver_step_ml_steady) into:

  * Pre-processing  - input preparation, gather/put/send, H2D staging
  * Inference       - forward / run_model / device inference
  * Post-processing - D2H, receive/scatter/unpack, output reconstruction
  * Wait / other    - token waits, syncs, unknown remainder

Classification reuses the analyzer's semantic phase scoring, so every
coupling library (SmartSim / AIxelerator / PhyDLL) is mapped with the same
rules. The Inference segment always covers the ENTIRE inference-device
subtree (the same region the icicle graphs collapse into a single
`inferenceDevice` / `smartsim_run_model` node): H2D copy, forward and D2H
copy are all attributed to Inference, never split across Pre/Post. The
segments aggregate per-region *self* times, so they are non-overlapping
and add up to the step total. Only the rank-mean view is rendered.

Usage:
  plot_config_comparison.py results/96c4g [--output results/96c4g/config_comparison.png]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cmi_scorep_profiles import get_semantic_phase_score  # noqa: E402

SCORE_TO_PHASE = {
    20: "pre",   # input preparation
    30: "pre",   # send / put / gather
    40: "pre",   # H2D (only outside the inference-device subtree)
    50: "inference",
    55: "inference",
    60: "post",  # D2H (only outside the inference-device subtree)
    70: "post",  # receive / scatter / unpack
    80: "post",  # output reconstruction
}

# Subtrees that are entirely attributed to the Inference segment (mirrors the
# icicle graphs, where the single-child collapse produces one "inferenceDevice"
# / "smartsim_run_model" node containing H2D + forward + D2H).
INFERENCE_SUBTREE_MARKERS = ("inferencedevice", "smartsim_run_model")

PHASE_ORDER = ["pre", "inference", "post", "other"]
PHASE_LABELS = {
    "pre": "Pre-processing (prepare / gather / put / H2D)",
    "inference": "Inference (entire inferenceDevice: H2D + forward + D2H)",
    "post": "Post-processing (D2H / scatter / unpack)",
    "other": "Wait / sync / other",
}
PHASE_COLORS = {
    "pre": "#4c9ed9",
    "inference": "#8e6bbd",
    "post": "#f4a261",
    "other": "#b8b8b8",
}


def bucket_of(region_name: str) -> str:
    return SCORE_TO_PHASE.get(get_semantic_phase_score(region_name), "other")


def load_config(config_dir: Path):
    """Returns dict with per-phase self-time sums (rank mean, per ML step) or None.

    Only the Rank_Mean self-time decomposition is used: per-region self times
    add up exactly to the step inclusive total, which keeps the stacked bars
    consistent. (The CSV's Controller_Rank0 column is an INCLUSIVE rank-0 time,
    which must not be summed over nested regions.)
    """
    summary = config_dir / "cmi_phase_summary.csv"
    if not summary.exists():
        return None
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    root = next((r for r in rows if r["Region"] == "solver_step_ml_steady"), None)
    if root is None:
        return None
    prefix = root["Callpath"] + " -> "

    rows_map = {r["Callpath"]: r for r in rows if r["Callpath"].startswith(prefix)}
    rows_map[root["Callpath"]] = root
    children = {cp: [] for cp in rows_map}
    for cp in rows_map:
        parent = cp.rsplit(" -> ", 1)[0]
        if parent in children:
            children[parent].append(cp)

    def subtree_self_sum(cp: str) -> float:
        total = float(rows_map[cp]["Self (ms)"])
        return total + sum(subtree_self_sum(ch) for ch in children[cp])

    buckets = {p: 0.0 for p in PHASE_ORDER}

    def walk(cp: str):
        row = rows_map[cp]
        if row["Region"].lower() in INFERENCE_SUBTREE_MARKERS:
            buckets["inference"] += subtree_self_sum(cp)
            return
        buckets[bucket_of(row["Region"])] += float(row["Self (ms)"])
        for ch in children[cp]:
            walk(ch)

    walk(root["Callpath"])

    return {
        "label": config_dir.name,
        "dir": config_dir,
        "mean": buckets,
        "step_mean": float(root["Rank_Mean (ms)"]),
    }


def collect_params(config_dirs) -> dict:
    """Merge model/resolution/batch_size found in the configs' run_metadata.json."""
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


def render(configs, output_path: Path, meta: dict):
    configs = sorted(configs, key=lambda c: c["step_mean"])  # fastest last -> top bar
    labels = [c["label"] for c in configs]
    y = range(len(configs))
    global_max = max(sum(c["mean"].values()) for c in configs)

    fig, axis = plt.subplots(figsize=(11, 0.5 * len(configs) + 2.2))
    left = [0.0] * len(configs)
    for phase in PHASE_ORDER:
        values = [c["mean"][phase] for c in configs]
        axis.barh(y, values, left=left, height=0.62,
                  color=PHASE_COLORS[phase], edgecolor="white", linewidth=0.4,
                  label=PHASE_LABELS[phase])
        for yi, (v, l) in enumerate(zip(values, left)):
            if v > 0.035 * global_max:
                axis.text(l + v / 2.0, yi, f"{v:.0f}", ha="center", va="center",
                          fontsize=6.5, color="black")
        left = [l + v for l, v in zip(left, values)]
    for yi, c in enumerate(configs):
        axis.text(c["step_mean"], yi, f" {c['step_mean']:.0f} ms", va="center",
                  fontsize=7.5, weight="bold")

    axis.set_yticks(list(y))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("ms per steady ML step (rank mean)", fontsize=9)
    axis.grid(axis="x", alpha=0.25, linewidth=0.4)
    axis.tick_params(labelsize=7)

    handles, labels_ = axis.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center",
               ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    params = " ".join(str(v) for v in (
        meta.get("model", "watercnn"),
        meta.get("resolution", ""),
        f"batch {meta['batch_size']}" if meta.get("batch_size") else "",
        "96 ranks / 4 GPUs",
    ) if v)
    fig.suptitle(f"Steady ML-step phase decomposition per coupling-library configuration ({params})",
                 fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved configuration comparison to: {output_path}")


def write_table(configs, output_path: Path):
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["config", "step_total_ms"] + [f"{p}_mean_ms" for p in PHASE_ORDER])
        for c in configs:
            writer.writerow([c["label"], f"{c['step_mean']:.3f}"]
                            + [f"{c['mean'][p]:.3f}" for p in PHASE_ORDER])
    print(f"[+] Saved phase table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing one sub-directory per configuration, "
                             "each with a cmi_phase_summary.csv.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: <results_dir>/config_comparison.png).")
    args = parser.parse_args()

    configs = []
    for config_dir in sorted(p for p in args.results_dir.iterdir() if p.is_dir()):
        loaded = load_config(config_dir)
        if loaded is None:
            print(f"[WARN] no solver_step_ml_steady phase summary in {config_dir.name}; skipping")
            continue
        total = sum(loaded["mean"].values())
        print(f"  {loaded['label']}: step={loaded['step_mean']:.1f} ms, "
              f"bucket sum={total:.1f} ms "
              f"(pre={loaded['mean']['pre']:.1f}, inf={loaded['mean']['inference']:.1f}, "
              f"post={loaded['mean']['post']:.1f}, other={loaded['mean']['other']:.1f})")
        configs.append(loaded)

    if not configs:
        raise SystemExit("No configuration with a solver_step_ml_steady summary found.")

    output = args.output or (args.results_dir / "config_comparison.png")
    meta = collect_params([c["dir"] for c in configs])
    render(configs, output, meta)
    write_table(configs, output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
