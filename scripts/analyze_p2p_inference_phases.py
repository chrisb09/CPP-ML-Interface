#!/usr/bin/env python3
"""P2P vs collective AIx phase analysis for one ML step (+ per-step p2p metrics).

Part A - per-controller P2P metrics (ALL steady ML steps, both p2p runs):
  per workgroup (controller): ready-range inference region count and total
  time (sum of range_inference spans, i.e. including the per-chunk H2D/D2H),
  the full ready-range inference window (first start / last end), the input
  window (first worker send -> last input ready), the output window (first
  result send -> last worker output received), and the average time per
  non-controller rank spent sending (input) and receiving (result).
  Written to <config>/p2p_inference_metrics.csv.

Part B - four graphs for one selected ML step (default: the LAST steady one):
  1. the regular P2P timeline figure of that step, overlaid with the
     collective run's controller reference points (gather done / inference
     done, each relative to the collective controller's own step start);
  2. per-controller 3-phase bars (input / inference / output windows drawn at
     their actual start times) for collective vs P2P - collective is
     sequential, P2P overlaps (hence the vertical offset per phase);
  3. one bar per controller subdivided into pre-inference / inference /
     post-inference (collective 4, p2p 4 each, plus one SmartSim c0 bar where
     "inference" = first run_model start -> last run_model end across ranks);
  4. device-busy time per controller: P2P = summed ready-range inference
     spans (incl. H2D/D2H), collective = the inferenceDevice region span;
     P2P bars are annotated with their ready-range count.

Usage:
  analyze_p2p_inference_phases.py --output-dir results/96c4g_hires \
      --timeline aix_p2p=3738549 aix_p2p_fullcredits=3738560 \
      --coll-trace aix_coll=3738533 --smartsim-trace smartsim_c0=3738506 \
      [--step 100] [--num-workgroups 4]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p2p_timeline import (  # noqa: E402
    read_events, get_workgroup_mapping, synchronize_step_clocks, render_step)

OTF2_REGION_RE = re.compile(r"^(ENTER|LEAVE)\s+(\d+)\s+(\d+)\s+Region: \"([^\"]+)\"")
OTF2_CLOCK_RE = re.compile(r"Ticks per Seconds:\s*([0-9.eE+-]+)")

PHASE_COLORS = {"input": "#4c9ed9", "inference": "#8e6bbd", "output": "#f4a261"}
RUN_COLORS = {"coll": "#2a9d8f", "p2p": "#4c9ed9", "fullcredits": "#8e6bbd"}


# ---------------------------------------------------------------------------
# P2P timeline metrics (Part A)
# ---------------------------------------------------------------------------

def p2p_controller_metrics(events, step, wg_map):
    """Per-controller metrics dict for one clock-synchronized ML step."""
    step_events = synchronize_step_clocks(
        [e for e in events if e["step"] == step], wg_map)
    out = {}
    for ctrl, members in wg_map.items():
        ctrl_ev = [e for e in step_events if e["world_rank"] == ctrl]
        worker_ev = [e for e in step_events
                     if e["world_rank"] in members and e["world_rank"] != ctrl]
        if not ctrl_ev:
            continue
        t0 = min(e["time_s"] for e in ctrl_ev if e["event"] == "ml_step_start")
        t_end = max((e["time_s"] for e in ctrl_ev if e["event"] == "ml_step_end"),
                    default=None)
        m = {"wg": ctrl,
             "ctrl_start_ms": 0.0,
             "ctrl_end_ms": (t_end - t0) * 1e3 if t_end else float("nan")}

        # ready-range inference spans (controller GPU-side, incl. per-chunk H2D/D2H)
        ranges = []
        for e in ctrl_ev:
            if e["event"] != "range_inference_start":
                continue
            end = next((c["time_s"] for c in ctrl_ev
                        if c["event"] == "range_inference_end"
                        and c["range_first_rank"] == e["range_first_rank"]
                        and c["range_end_rank"] == e["range_end_rank"]
                        and c["time_s"] >= e["time_s"]), None)
            if end is not None:
                ranges.append((e["time_s"], end))
        fwd = []
        for e in ctrl_ev:
            if e["event"] != "torch_forward_start":
                continue
            end = next((c["time_s"] for c in ctrl_ev
                        if c["event"] == "torch_forward_end"
                        and c["sample_start"] == e["sample_start"]
                        and c["time_s"] >= e["time_s"]), None)
            if end is not None:
                fwd.append((e["time_s"], end))
        m["ready_range_count"] = len(ranges)
        m["ready_range_sum_ms"] = sum((b - a) * 1e3 for a, b in ranges)
        m["torch_forward_sum_ms"] = sum((b - a) * 1e3 for a, b in fwd)
        if ranges:
            m["range_first_start_ms"] = (ranges[0][0] - t0) * 1e3
            m["range_last_end_ms"] = (max(b for _, b in ranges) - t0) * 1e3
        else:
            m["range_first_start_ms"] = m["range_last_end_ms"] = float("nan")

        # input window: first worker send -> last controller-side input ready
        sends = [e["time_s"] for e in worker_ev if e["event"] == "input_send_start"]
        readies = [e["time_s"] for e in ctrl_ev if e["event"] == "input_ready"]
        m["input_first_send_ms"] = (min(sends) - t0) * 1e3 if sends else float("nan")
        m["input_last_ready_ms"] = (max(readies) - t0) * 1e3 if readies else float("nan")

        # output window: first result send -> last worker output received
        rs = [e["time_s"] for e in ctrl_ev if e["event"] == "result_send_start"]
        rr = [e["time_s"] for e in worker_ev if e["event"] == "result_received"]
        m["output_first_send_ms"] = (min(rs) - t0) * 1e3 if rs else float("nan")
        m["output_last_recv_ms"] = (max(rr) - t0) * 1e3 if rr else float("nan")

        # average per non-controller rank: input-send span and result-receive span
        send_spans, recv_spans = [], []
        for w in sorted({e["world_rank"] for e in worker_ev}):
            w_ev = [e for e in worker_ev if e["world_rank"] == w]
            s0 = [e["time_s"] for e in w_ev if e["event"] == "input_send_start"]
            s1 = [e["time_s"] for e in w_ev if e["event"] == "input_send_complete"]
            r0 = [e["time_s"] for e in w_ev if e["event"] == "result_wait_start"]
            r1 = [e["time_s"] for e in w_ev if e["event"] == "result_received"]
            if s0 and s1:
                send_spans.append(s1[0] - s0[0])
            if r0 and r1:
                recv_spans.append(r1[0] - r0[0])
        m["avg_worker_send_ms"] = st.median(send_spans) * 1e3 if send_spans else float("nan")
        m["avg_worker_recv_ms"] = st.median(recv_spans) * 1e3 if recv_spans else float("nan")
        out[ctrl] = m
    return out


def write_p2p_metrics_csv(path, metrics_by_step):
    rows = []
    for step in sorted(metrics_by_step):
        for ctrl in sorted(metrics_by_step[step]):
            rows.append({"step": step, **metrics_by_step[step][ctrl]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Saved P2P controller metrics to: {path}")


# ---------------------------------------------------------------------------
# OTF2 trace extraction (Part B)
# ---------------------------------------------------------------------------

def stream_otf2_regions(anchor, regions):
    """{loc: {region: [(enter, leave), ...]}} via one otf2-print stream."""
    otf2_print = os.environ.get("OTF2_PRINT", shutil.which("otf2-print") or "otf2-print")
    proc = subprocess.Popen([otf2_print, str(anchor)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    enters = {}
    out = {}
    for line in proc.stdout:
        m = OTF2_REGION_RE.match(line)
        if not m:
            continue
        ev, loc, ts, reg = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if reg not in regions:
            continue
        if ev == "ENTER":
            enters.setdefault((loc, reg), []).append(ts)
        else:
            stack = enters.get((loc, reg))
            if stack:
                out.setdefault(loc, {}).setdefault(reg, []).append((stack.pop(0), ts))
    proc.stdout.close()
    if proc.wait() != 0:
        print(f"[WARN] otf2-print exit={proc.returncode} for {anchor}")
    return out


def trace_clock_ms_per_tick(anchor):
    otf2_print = os.environ.get("OTF2_PRINT", shutil.which("otf2-print") or "otf2-print")
    proc = subprocess.run([otf2_print, "-G", str(anchor)], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True)
    m = OTF2_CLOCK_RE.search(proc.stdout)
    ticks_per_s = float(m.group(1)) if m else 2.1e9
    return 1e3 / ticks_per_s


def extract_coll_controller_windows(anchor, num_workgroups, steady_idx):
    """Per controller (ranks 0..k-1): input/inference/output/step windows in ms
    relative to that controller's own step start."""
    regions = {"solver_step_ml_steady", "gatherInputData", "inferenceDevice",
               "scatterOutputData"}
    data = stream_otf2_regions(anchor, regions)
    ms = trace_clock_ms_per_tick(anchor)
    # Controllers are the ranks that actually execute inferenceDevice (AIx).
    # In het jobs the controller is NOT rank 0 (e.g. the lone c23g task), so
    # detect the ranks by their device-region BUSY TIME; worker ranks may enter
    # the region for a tiny moment, hence the >=50%-of-max filter. Ordinals
    # (0..k-1) are used as keys to match the P2P timeline workgroup indices.
    totals = {r: sum((b - a) for a, b in d.get("inferenceDevice", []))
              for r, d in data.items()}
    mx_total = max(totals.values()) if totals else 0.0
    ctrl_ranks = sorted(r for r, v in totals.items()
                        if mx_total > 0 and v >= 0.5 * mx_total)
    if not ctrl_ranks:
        ctrl_ranks = list(range(num_workgroups))
    out = {}
    for ordinal, ctrl in enumerate(ctrl_ranks):
        d = data.get(ctrl, {})
        if len(d.get("solver_step_ml_steady", [])) <= steady_idx:
            continue
        s0, s1 = d["solver_step_ml_steady"][steady_idx]

        def win(reg, idx=None):
            w = d.get(reg, [])
            # The comm/device regions also wrap the warmup ML step (2xx vs 2xx-2
            # occurrences), so they are offset by one against solver_step_ml_steady.
            use_idx = steady_idx if idx is None else idx
            if len(w) <= use_idx:
                return (float("nan"), float("nan"))
            a, b = w[use_idx]
            return ((a - s0) * ms, (b - s0) * ms)

        out[ordinal] = {
            "step_ms": (s1 - s0) * ms,
            "input": win("gatherInputData", steady_idx + 1),
            "inference": win("inferenceDevice", steady_idx + 1),
            "output": win("scatterOutputData", steady_idx + 1),
        }
    if ctrl_ranks:
        print(f"    collective controllers detected at ranks {ctrl_ranks}")
    return out


def extract_smartsim_windows(anchor, steady_idx):
    """Rank-0 step window and the run_model window (first enter across all
    ranks -> last leave), both relative to rank 0's step start."""
    regions = {"solver_step_ml_steady", "smartsim_run_model"}
    data = stream_otf2_regions(anchor, regions)
    ms = trace_clock_ms_per_tick(anchor)
    loc0 = data.get(0, {}).get("solver_step_ml_steady", [])
    if len(loc0) <= steady_idx:
        return None
    s0, s1 = loc0[steady_idx]
    models = [pair for pairs in (d.get("smartsim_run_model", [])
                                 for d in data.values()) for pair in pairs
              if pair[0] < s1 and pair[1] > s0]  # overlap with rank-0's step window
    first_enter = min(p[0] for p in models) if models else s0
    last_leave = max(p[1] for p in models) if models else s1
    return {"step_ms": (s1 - s0) * ms,
            "inference": ((first_enter - s0) * ms, (last_leave - s0) * ms)}


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_phase_bars(coll_windows, p2p_windows, p2p_label, output_path):
    """Per controller: input/inference/output windows at their actual start
    times; collective rows vs P2P rows, offset so overlaps stay visible.
    One y-label per workgroup (wgX); lanes are identified by the legend."""
    fig, axis = plt.subplots(figsize=(11, 7.5))
    controllers = sorted(set(coll_windows) & set(p2p_windows))
    y = 0
    group_centers = []
    for c in controllers:
        group_top = y
        for run, windows, color in (("coll", coll_windows, RUN_COLORS["coll"]),
                                    (p2p_label, p2p_windows, RUN_COLORS.get(
                                        p2p_label, RUN_COLORS["p2p"]))):
            for phase in ("input", "inference", "output"):
                a, b = windows[c][phase]
                axis.broken_barh([(a, b - a)], (y - 0.33, 0.66),
                                 facecolors=PHASE_COLORS[phase],
                                 edgecolors=color, linewidth=1.0)
                y += 1
        group_centers.append((group_top + y - 1) / 2.0)
        y += 1  # gap between workgroups
        axis.axhline(y - 0.5, color="0.85", linewidth=0.6)
    axis.set_yticks(group_centers)
    axis.set_yticklabels([f"wg{c}" for c in controllers], fontsize=9, weight="bold")
    axis.invert_yaxis()
    axis.set_ylabel("AIx workgroup", fontsize=9)
    axis.set_xlabel("ms since that controller's ML-step start", fontsize=9)
    axis.grid(axis="x", alpha=0.25, linewidth=0.4)
    axis.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_COLORS[p]) for p in ("input", "inference", "output")]
    edge_handles = [plt.Line2D([0], [0], color=RUN_COLORS["coll"], linewidth=2),
                    plt.Line2D([0], [0], color=RUN_COLORS.get(p2p_label, RUN_COLORS["p2p"]), linewidth=2)]
    axis.legend(handles + edge_handles,
                ["Input send / gather", "Inference / ready-ranges", "Output send / scatter",
                 "collective", p2p_label],
                fontsize=8, frameon=False, ncol=5, loc="upper center",
                bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"ML-step phase windows per workgroup: collective vs {p2p_label} "
                 f"(step 202, last steady ML step)", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved phase-window comparison to: {output_path}")


def render_phase_stack(rows, output_path):
    """rows: [(label, step_ms, (inf_start_ms, inf_end_ms))] -> stacked bars.

    An inference window may extend past the controller's own step end (e.g.
    SmartSim: the slowest rank's run_model finishes after rank 0 left the ML
    step region); in that case the bar is extended and labeled with the true
    world-wide end.
    """
    fig, axis = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.8))
    max_total = 0.0
    for i, (label, step_ms, (i0, i1)) in enumerate(rows):
        total = max(step_ms, i1 if i1 == i1 else 0.0)
        max_total = max(max_total, total)
        pre = max(i0, 0.0)
        inf = max(i1 - i0, 0.0)
        post = max(total - i1, 0.0)
        left = 0.0
        for width, phase in ((pre, "input"), (inf, "inference"), (post, "output")):
            axis.barh(i, width, left=left, height=0.68,
                      color=PHASE_COLORS[phase], edgecolor="white", linewidth=0.4)
            if width > 0.04 * total:
                axis.text(left + width / 2, i, f"{width:.0f}", ha="center",
                          va="center", fontsize=6.5)
            left += width
        axis.text(total * 1.01, i, f"{total:.0f} ms", va="center", fontsize=7.5,
                  weight="bold")
    axis.set_xlim(0, max_total * 1.12)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("ms per ML step (relative to that controller's step start)", fontsize=9)
    axis.grid(axis="x", alpha=0.25, linewidth=0.4)
    axis.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_COLORS[p])
               for p in ("input", "inference", "output")]
    fig.legend(handles, ["Pre-inference", "Inference / ready-ranges", "Post-inference"],
               fontsize=8, frameon=False, ncol=3, loc="upper right",
               bbox_to_anchor=(0.99, 0.955))
    fig.suptitle("ML-step structure per workgroup: pre / inference / post "
                 "(step 202, last steady ML step)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved phase stack to: {output_path}")


def render_device_busy(rows, output_path):
    """rows: [(label, busy_ms, range_count or None)] horizontal bars."""
    fig, axis = plt.subplots(figsize=(9, 0.38 * len(rows) + 1.6))
    for i, (label, busy_ms, count) in enumerate(rows):
        axis.barh(i, busy_ms, height=0.66, color="#8e6bbd", edgecolor="white", linewidth=0.4)
        txt = f"{busy_ms:.1f} ms" + (f"  ({count} ranges)" if count is not None else "")
        axis.text(busy_ms, i, f" {txt}", va="center", fontsize=7.5)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("device-busy inference time per workgroup (ms), incl. per-chunk H2D/D2H",
                    fontsize=9)
    axis.grid(axis="x", alpha=0.25, linewidth=0.4)
    axis.tick_params(labelsize=7)
    fig.suptitle("Ready-range / device inference time per workgroup "
                 "(step 202, last steady ML step)", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved device-busy chart to: {output_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_pairs(values):
    out = []
    for v in values:
        if "=" in v:
            name, job = v.split("=", 1)
            out.append((name, job))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeline", nargs="*", default=[],
                        help="config=jobid pairs (P2P runs with CSV timelines).")
    parser.add_argument("--coll-trace", nargs="*", default=[],
                        help="config=jobid pair of the collective AIX run (OTF2).")
    parser.add_argument("--smartsim-trace", nargs="*", default=[],
                        help="config=jobid pair of a SmartSim run (OTF2), for graph 3.")
    parser.add_argument("--step", type=int, default=None,
                        help="ML-step index in timeline numbering (0 = warmup). "
                             "Default: the LAST steady step.")
    parser.add_argument("--num-workgroups", type=int, default=None,
                        help="Fallback AIx workgroup count when controllers "
                             "cannot be detected (default: 4).")
    parser.add_argument("--file-prefix", type=str, default="",
                        help="Prefix for the suite-level combined graphs "
                             "(phase_bars_*, ml_step_phase_stack, "
                             "ready_range_inference_time); per-config outputs "
                             "are unaffected.")
    args = parser.parse_args()
    pfx = f"{args.file_prefix}_" if args.file_prefix else ""

    base_dir = args.output_dir.resolve().parent.parent
    suite_dir = args.output_dir
    suite_dir.mkdir(parents=True, exist_ok=True)

    timeline_runs = []  # (name, job, events, wg_map, metrics_by_step)
    for name, job in parse_pairs(args.timeline):
        if "/" in job:
            tdir = base_dir / job
        else:
            tdir = base_dir / "logs" / f"aix_p2p_timeline_{job}"
        if not tdir.is_dir():
            print(f"[WARN] timeline dir missing for {name}: {tdir}")
            continue
        events = read_events(tdir)
        wg_map = get_workgroup_mapping(events)
        max_step = max(e["step"] for e in events)
        step = args.step if args.step is not None else max_step
        metrics = {s: p2p_controller_metrics(events, s, wg_map)
                   for s in range(1, max_step + 1)}
        write_p2p_metrics_csv(suite_dir / name / "p2p_inference_metrics.csv", metrics)
        timeline_runs.append((name, job, events, wg_map, step, metrics))
        print(f"[+] {name}: rendered+metrics for steps 1..{max_step}")

    # collective controller windows from the OTF2 trace
    coll_windows, coll_name = None, None
    for name, job in parse_pairs(args.coll_trace):
        if "/" in job:
            anchor = base_dir / job / "traces.otf2"
        else:
            anchor = base_dir / "scorep_runs" / f"{suite_dir.name}_{name}_{job}" / "traces.otf2"
        if not anchor.exists():
            print(f"[WARN] coll trace missing for {name}: {anchor}")
            continue
        max_steady = args.step - 1 if args.step is not None else None
        if max_steady is None:
            probe = stream_otf2_regions(anchor, {"solver_step_ml_steady"})
            max_steady = min(len(d.get("solver_step_ml_steady", []))
                             for d in probe.values()) - 1
        coll_windows = extract_coll_controller_windows(anchor, args.num_workgroups, max_steady)
        coll_name = name
        print(f"[+] {name}: extracted controller windows for steady step #{max_steady}")

    sel_step = args.step
    if sel_step is None and timeline_runs:
        sel_step = max(run[4] for run in timeline_runs)  # last steady ML step
    steady_idx = max((sel_step or 1) - 1, 0)  # timeline idx -> trace steady idx

    smartsim_windows, smartsim_name = None, None
    for name, job in parse_pairs(args.smartsim_trace):
        if "/" in job:
            anchor = base_dir / job / "traces.otf2"
        else:
            anchor = base_dir / "scorep_runs" / f"{suite_dir.name}_{name}_{job}" / "traces.otf2"
        if not anchor.exists():
            print(f"[WARN] smartsim trace missing for {name}: {anchor}")
            continue
        probe = stream_otf2_regions(anchor, {"solver_step_ml_steady"})
        n = min(len(d.get("solver_step_ml_steady", [])) for d in probe.values())
        idx = min(steady_idx, n - 1)
        smartsim_windows = extract_smartsim_windows(anchor, idx)
        smartsim_name = name

    # Graph 1: overlaid step timeline per p2p run
    if coll_windows and timeline_runs:
        overlays = {}
        for c, w in coll_windows.items():
            ga, ge = w["input"]
            ia, ie = w["inference"]
            ov = []
            if ge == ge:
                ov.append((ge, f"coll wg{c}: inputs done", ":", RUN_COLORS["coll"]))
            if ie == ie:
                ov.append((ie, f"coll wg{c}: inference done", "--", RUN_COLORS["coll"]))
            overlays[c] = ov
        for name, job, events, wg_map, step, _ in timeline_runs:
            out_png = suite_dir / name / f"p2p_timeline_step_{step:03d}_vs_coll.png"
            render_step(events, step, suite_dir / name, "watercnn",
                        output_name=out_png.name, overlays=overlays)
            print(f"[+] Saved overlaid step timeline to: {out_png}")

    # Graph 2: coll vs p2p phase windows (one graph per p2p run)
    if coll_windows:
        for name, job, events, wg_map, step, metrics in timeline_runs:
            p2p_windows = {}
            for c, m in metrics[step].items():
                p2p_windows[c] = {
                    "input": (m["input_first_send_ms"], m["input_last_ready_ms"]),
                    "inference": (m["range_first_start_ms"], m["range_last_end_ms"]),
                    "output": (m["output_first_send_ms"], m["output_last_recv_ms"]),
                    "step_ms": m["ctrl_end_ms"],
                }
            render_phase_bars(coll_windows, p2p_windows, name,
                              suite_dir / f"{pfx}phase_bars_coll_vs_{name}.png")

    # Graph 3: stacked pre/inference/post per controller (+ smartsim c0)
    rows = []
    if coll_windows:
        for c, w in sorted(coll_windows.items()):
            rows.append((f"{coll_name} wg{c}", w["step_ms"], w["inference"]))
    for name, job, events, wg_map, step, metrics in timeline_runs:
        for c, m in sorted(metrics[step].items()):
            rows.append((f"{name} wg{c}", m["ctrl_end_ms"],
                         (m["range_first_start_ms"], m["range_last_end_ms"])))
    if smartsim_windows:
        rows.append((f"{smartsim_name}", smartsim_windows["step_ms"],
                     smartsim_windows["inference"]))
    if rows:
        render_phase_stack(rows, suite_dir / f"{pfx}ml_step_phase_stack.png")

    # Graph 4: device-busy inference time per controller
    busy_rows = []
    if coll_windows:
        for c, w in sorted(coll_windows.items()):
            busy_rows.append((f"{coll_name} wg{c}", w["inference"][1] - w["inference"][0], None))
    for name, job, events, wg_map, step, metrics in timeline_runs:
        for c, m in sorted(metrics[step].items()):
            busy_rows.append((f"{name} wg{c}", m["ready_range_sum_ms"], m["ready_range_count"]))
    if busy_rows:
        render_device_busy(busy_rows, suite_dir / f"{pfx}ready_range_inference_time.png")


if __name__ == "__main__":
    main()
