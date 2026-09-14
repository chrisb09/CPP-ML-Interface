#!/usr/bin/env python3
"""Per-ML-step makespan analysis: instability across steps and workgroup load imbalance.

Two complementary views that the aggregate (per-config mean) numbers hide:

1. Step makespan series per library-config (feature: instability over time)
   Source: solver log lines `Step <N>, ML, local moved: ..., time: <X> ms`
   (printed by rank 0 once per ML step). One series per config, plotted on a
   log y-axis so fast and slow libraries fit into one figure; the warmup ML
   step is marked. A high step-to-step variance = scheduling/coordination
   instability of that coupling library.

2. Per-workgroup makespan spread for the AIx runs (feature: static load imbalance)
   AIx pins workloads to fixed workgroups (1 GPU controller + workers) with no
   dynamic rebalancing, so the spread between the longest and shortest
   workgroup makespan quantifies structural load imbalance per step.
   Sources:
     - pipelined (p2p) runs: the CSV P2P timelines
       (logs/aix_p2p_timeline_<jobid>/aix_p2p_timeline_rank_<rank>.csv);
       workgroup makespan = max(ml_step_end) - min(ml_step_start) over the
       workgroup's ranks.
     - collective runs: the OTF2 trace (scorep_runs/<suite>_<config>_<jobid>/);
       ENTER/LEAVE of region `solver_step_ml_steady` are extracted with
       otf2-print (single node -> shared clock, so cross-rank differences are
       valid).

Usage:
  analyze_ml_step_makespan.py --output-dir results/96c4g_hires \
      --log-config smartsim_c0=3738506 [name=jobid ...]
      [--timeline aix_p2p=3738549 aix_p2p_fullcredits=3738560]
      [--trace aix_coll=3738533] [--num-workgroups 4]
"""

import argparse
import csv
import os
import re
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML_STEP_RE = re.compile(
    r"^Step (\d+), ML, local moved: .*, time: ([0-9.eE+-]+) ms\s*$")
OTF2_REGION_RE = re.compile(
    r"^(ENTER|LEAVE)\s+(\d+)\s+(\d+)\s+Region: \"solver_step_ml_steady\"")
OTF2_CLOCK_RE = re.compile(r"Ticks per Seconds:\s*([0-9.eE+-]+)")

COLORS = ["#4c9ed9", "#8e6bbd", "#f4a261", "#2a9d8f", "#e76f51",
          "#606c38", "#bc6c25", "#7209b7", "#4361ee", "#999999"]


def parse_solver_log_steps(log_path: Path) -> dict:
    """Returns {step_number: ms} from the solver log (rank-0 view)."""
    steps = {}
    with log_path.open(errors="replace") as stream:
        for line in stream:
            m = ML_STEP_RE.match(line.strip())
            if m:
                steps[int(m.group(1))] = float(m.group(2))
    return steps


def summarize_series(label, steps_dict):
    values = [v for _, v in sorted(steps_dict.items())]
    if not values:
        return None
    med = st.median(values)
    p5 = sorted(values)[int(0.05 * (len(values) - 1))]
    p95 = sorted(values)[int(0.95 * (len(values) - 1))]
    cv = st.stdev(values[1:]) / med if len(values) > 2 else 0.0
    print(f"  {label}: n={len(values)} median={med:.1f} ms  p5={p5:.1f}  p95={p95:.1f}  "
          f"CV(steady)={100*cv:.1f}%  max={max(values):.1f}")
    return {"median": med, "p5": p5, "p95": p95, "cv": cv}


def resolve_num_workgroups(label: str, override):
    """AIx workgroup count for a config label.

    HH2 scenario tags encode the setup: 4-GPU scenarios (s6/s7) use 4
    workgroups, the 1-GPU scenarios (s1..s5) exactly 1. Anything else keeps
    the historical default of 4.
    """
    if override:
        return override
    m = re.search(r"_s(\d+)_", label)
    if m:
        return 4 if int(m.group(1)) >= 6 else 1
    return 4


def has_series_data(steps_dict) -> bool:
    """At least two parseable ML-step entries (a lone warmup row gives no
    median for the steady curve)."""
    return sum(1 for _ in steps_dict.items()) >= 2


def parse_log_configs(pairs, base_dir: Path):
    """{config: {step: ms}} from logs/mini_app_output_<jobid>.txt.

    A value containing "/" is used as an explicit log path (HH2 runs write
    logs/mini_app_output_<tag>.txt).
    """
    out = {}
    for name, job in pairs:
        if "/" in job:
            log_path = base_dir / job
        else:
            log_path = base_dir / "logs" / f"mini_app_output_{job}.txt"
        if not log_path.exists():
            print(f"[WARN] solver log not found for {name} (job {job}): {log_path}")
            continue
        steps = parse_solver_log_steps(log_path)
        if not has_series_data(steps):
            print(f"[WARN] no usable ML-step series in {log_path} for {name}; skipping")
            continue
        out[name] = steps
    return out


def parse_timeline_config(timeline_dir: Path, num_workgroups: int):
    """{ml_step_idx: {wg: makespan_ms}} from the P2P timeline CSVs.

    AIx round-robin workgroup layout: workgroup = world_rank % num_workgroups
    (controllers = the lowest ranks, e.g. 0..3 for 4 GPUs); verified against
    the timeline CSVs' (workgroup_rank, is_controller) fields.
    """
    per_step = {}
    for csv_path in sorted(timeline_dir.glob("aix_p2p_timeline_rank_*.csv")):
        with csv_path.open(newline="") as stream:
            for r in csv.DictReader(stream):
                ev = r["event"]
                if ev not in ("ml_step_start", "ml_step_end"):
                    continue
                s = int(r["step"])
                wg = int(r["world_rank"]) % num_workgroups
                t = float(r["time_s"])
                d = per_step.setdefault(s, {}).setdefault(wg, {})
                if ev == "ml_step_start":
                    d["start"] = min(d.get("start", t), t)
                else:
                    d["end"] = max(d.get("end", t), t)
    out = {}
    for s, wgs in per_step.items():
        out[s] = {wg: (d["end"] - d["start"]) * 1e3 for wg, d in wgs.items()
                  if "start" in d and "end" in d}
    return out


def parse_trace_config(trace_anchor: Path, num_workgroups: int):
    """{ml_step_idx: {wg: makespan_ms}} from the OTF2 trace of a collective run.

    Streams `otf2-print` once and extracts ENTER/LEAVE of the
    `solver_step_ml_steady` region per location. Locations are Score-P MPI
    ranks in world order (single node, shared clock). Workgroups are the AIx
    round-robin layout: wg = rank % num_workgroups (controllers 0..3).
    """
    otf2_print = os.environ.get("OTF2_PRINT", shutil.which("otf2-print") or "otf2-print")
    try:
        proc = subprocess.Popen([otf2_print, trace_anchor],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True)
    except FileNotFoundError:
        sys.exit(f"[ERROR] otf2-print not found ({otf2_print}); set OTF2_PRINT")
    per_loc = {}
    for line in proc.stdout:
        m = OTF2_REGION_RE.match(line)
        if not m:
            continue
        ev, loc, ts = m.group(1), int(m.group(2)), int(m.group(3))
        d = per_loc.setdefault(loc, {})
        if ev == "ENTER":
            d.setdefault("starts", []).append(ts)
        else:
            d.setdefault("ends", []).append(ts)
    proc.stdout.close()
    if proc.wait() != 0:
        print(f"[WARN] otf2-print exited with {proc.returncode} for {trace_anchor}")

    n_steps = min(len(d.get("starts", [])) for d in per_loc.values())
    out = {}
    for s in range(n_steps):
        wgs = {}
        for loc, d in per_loc.items():
            wg = loc % num_workgroups
            start = d["starts"][s]
            end = d["ends"][s] if s < len(d.get("ends", [])) else None
            if end is None:
                continue
            wgs.setdefault(wg, {})[loc] = (start, end)
        out[s] = {}
        for wg, ranks in wgs.items():
            t0 = min(t for t, _ in ranks.values())
            t1 = max(t for _, t in ranks.values())
            out[s][wg] = (t1 - t0) / 1e9 * 1e3  # ticks (~2.1 GHz) -> ms
    return out


def write_workgroup_csv(path: Path, label: str, per_step, warmup_idx: int):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        wgs = sorted({w for m in per_step.values() for w in m})
        writer.writerow(["ml_step_index", "solver_step"] + [f"wg{w}_makespan_ms" for w in wgs]
                        + ["spread_ms", "is_warmup"])
        for s in sorted(per_step):
            spans = per_step[s]
            vals = [spans.get(w, float("nan")) for w in wgs]
            spread = max(vals) - min(vals)
            writer.writerow([s, 2 * (s + 1)] + [f"{v:.3f}" for v in vals]
                            + [f"{spread:.3f}", int(s == warmup_idx)])
    print(f"[+] Saved workgroup makespan table to: {path}")


def render_spread(configs, output_path: Path):
    """Per-step workgroup makespan spread, one line per config."""
    fig, axis = plt.subplots(figsize=(11, 4.2))
    for i, (label, series) in enumerate(configs):
        steps = sorted(s for s, spans in series.items() if s > 0)  # skip warmup
        spreads = [max(series[s].values()) - min(series[s].values()) for s in steps]
        axis.plot([2 * (s + 1) for s in steps], spreads,
                  color=COLORS[i % len(COLORS)], linewidth=1.0, alpha=0.9,
                  label=f"{label} (med {st.median(spreads):.1f} ms)")
        med = st.median(spreads)
        axis.axhline(med, color=COLORS[i % len(COLORS)], linewidth=0.6,
                     linestyle=":", alpha=0.5)
    axis.set_xlabel("solver step number (ML steps are even)", fontsize=9)
    axis.set_ylabel("workgroup makespan spread (ms)", fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.4)
    axis.legend(fontsize=8, frameon=False)
    axis.tick_params(labelsize=7)
    fig.suptitle("AIx static-load imbalance: spread between longest and shortest "
                 "workgroup makespan, per ML step", fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved workgroup makespan spread plot to: {output_path}")


def render_step_series(configs, warmup_step, output_path: Path, meta: dict):
    """Rank-0 ML-step makespan per config across all warm ML steps."""
    fig, axis = plt.subplots(figsize=(11, 4.2))
    for i, (label, steps_dict) in enumerate(configs):
        items = sorted(steps_dict.items())
        xs = [s for s, _ in items]
        ys = [v for _, v in items]
        axis.plot(xs, ys, color=COLORS[i % len(COLORS)], linewidth=1.0,
                  alpha=0.9, marker="o", markersize=2.0,
                  label=f"{label} (med {st.median(ys[1:]):.1f} ms)")
    axis.set_yscale("log")
    axis.axvline(warmup_step, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.text(warmup_step + 1, axis.get_ylim()[0] * 1.3, "warmup ML step",
              fontsize=7, color="grey")
    params = " ".join(str(v) for v in (
        meta.get("model", "watercnn"), meta.get("resolution", ""),
        f"batch {meta['batch_size']}" if meta.get("batch_size") else "") if v)
    axis.set_xlabel("solver step number (ML steps are even)", fontsize=9)
    axis.set_ylabel("ML step makespan, rank 0 (ms, log)", fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.4, which="both")
    axis.legend(fontsize=8, frameon=False, loc="upper right")
    axis.tick_params(labelsize=7)
    fig.suptitle(f"Per-step ML makespan per coupling-library configuration ({params})",
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved step makespan plot to: {output_path}")


def write_step_table(configs, output_path: Path):
    all_steps = sorted({s for _, d in configs for s in d})
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["solver_step"] + [name for name, _ in configs])
        for s in all_steps:
            row = [s] + [f"{d[s]:.3f}" if s in d else "" for _, d in configs]
            writer.writerow(row)
    print(f"[+] Saved step makespan table to: {output_path}")


def parse_pairs(values):
    out = []
    for v in values:
        if "=" not in v:
            continue
        name, job = v.split("=", 1)
        out.append((name, job))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Suite results dir (results/<suite>).")
    parser.add_argument("--log-config", nargs="*", default=[],
                        help="config=jobid pairs; makespan series from solver logs.")
    parser.add_argument("--timeline", nargs="*", default=[],
                        help="config=jobid pairs; workgroup makespan from P2P CSV timelines.")
    parser.add_argument("--trace", nargs="*", default=[],
                        help="config=jobid pairs; workgroup makespan from OTF2 traces.")
    parser.add_argument("--num-workgroups", type=int, default=None,
                        help="Number of AIx workgroups; default auto: 4 for s6/s7 tags, else 1 (4 for untagged legacy configs).")
    args = parser.parse_args()

    suite_dir = args.output_dir
    suite_dir.mkdir(parents=True, exist_ok=True)
    base_dir = suite_dir.resolve().parent.parent  # repo root containing logs/ and scorep_runs/

    log_configs = parse_log_configs(parse_pairs(args.log_config), base_dir)
    if log_configs:
        print("Rank-0 ML-step makespan series:")
        for name, steps_dict in log_configs.items():
            summarize_series(name, steps_dict)
        render_step_series(sorted(log_configs.items()), 2,
                           suite_dir / "ml_step_makespan.png", collect_params(suite_dir))
        write_step_table(sorted(log_configs.items()), suite_dir / "ml_step_makespan.csv")

    wg_configs = []
    for name, job in parse_pairs(args.timeline):
        if "/" in job:
            tdir = base_dir / job
        else:
            tdir = base_dir / "logs" / f"aix_p2p_timeline_{job}"
        if not tdir.is_dir():
            print(f"[WARN] timeline dir missing for {name} (job {job}): {tdir}")
            continue
        wg = resolve_num_workgroups(name, args.num_workgroups)
        per_step = parse_timeline_config(tdir, wg)
        wg_configs.append((name, per_step))
        write_workgroup_csv(suite_dir / name / "workgroup_makespan.csv",
                            name, per_step, warmup_idx=0)
    for name, job in parse_pairs(args.trace):
        if "/" in job:
            anchor = base_dir / job / "traces.otf2"
        else:
            anchor = base_dir / "scorep_runs" / f"{suite_dir.name}_{name}_{job}" / "traces.otf2"
        if not anchor.exists():
            print(f"[WARN] OTF2 trace missing for {name} (job {job}): {anchor}")
            continue
        wg = resolve_num_workgroups(name, args.num_workgroups)
        per_step = parse_trace_config(anchor, wg)
        wg_configs.append((name, per_step))
        write_workgroup_csv(suite_dir / name / "workgroup_makespan.csv",
                            name, per_step, warmup_idx=0)

    if wg_configs:
        print("Workgroup makespan spread (max-min across workgroups):")
        for name, per_step in wg_configs:
            spreads = [max(m.values()) - min(m.values())
                       for s, m in sorted(per_step.items()) if s > 0]
            print(f"  {name}: median={st.median(spreads):.2f} ms  "
                  f"p95={sorted(spreads)[int(0.95*(len(spreads)-1))]:.2f}  "
                  f"max={max(spreads):.2f}")
        render_spread(wg_configs, suite_dir / "workgroup_makespan_spread.png")


def collect_params(suite_dir: Path) -> dict:
    """Merge model/resolution/batch_size from the configs' run_metadata.json."""
    meta = {}
    for config_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        meta_file = config_dir / "run_metadata.json"
        if not meta_file.exists():
            continue
        try:
            import json
            with meta_file.open() as stream:
                data = json.load(stream)
        except (OSError, ValueError):
            continue
        for key in ("model", "resolution", "batch_size"):
            if data.get(key) is not None and key not in meta:
                meta[key] = data[key]
    return meta


if __name__ == "__main__":
    main()
