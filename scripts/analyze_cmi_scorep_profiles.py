#!/usr/bin/env python3
"""
analyze_cmi_scorep_profiles.py
==============================
Generalized CLI analysis and visualization tool for CMI (C++ ML Interface)
Score-P CUBE profiles and AIx P2P timeline CSVs.

Features:
- Extracts hierarchical calltree profiles from .cubex files via `cube_dump`.
- Computes inclusive/exclusive metrics and per-rank statistics (mean, max, sum, controller).
- Generates publication-ready Icicle plots (hierarchical calltree breakdown).
- Generates component/phase breakdown bar charts.
- Parses AIx P2P timeline CSVs to compute pipeline makespans, stage overlap, and Gantt charts.
- Exports structured CSV summaries and Markdown reports.

Usage:
  python3 analyze_cmi_scorep_profiles.py --cubex path/to/profile.cubex [options]
  python3 analyze_cmi_scorep_profiles.py --p2p-timeline-dir path/to/aix_timeline_dir [options]
"""

import os
import sys
import re
import glob
import argparse
import subprocess
from shutil import which as shutil_which
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Plotting style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Canonical phase groupings and color palette
COLOR_MAP = {
    "Preprocess": "#4393C3",        # Light Blue
    "app_prepare_input": "#4393C3",
    "flowex_extract_cubes": "#92C5DE",
    "Provider Step": "#F46D43",     # Coral / Red-Orange
    "app_provider_inference": "#F46D43",
    "smartsim_library_static_step": "#F46D43",
    "aix_library_static_step": "#F46D43",
    "phydll_library_static_step": "#F46D43",
    "smartsim_token_wait": "#FDAE61",
    "smartsim_chunk_plan": "#FEE090",
    "smartsim_put_tensor": "#E08214",
    "smartsim_run_model": "#D73027",
    "smartsim_unpack_tensor": "#8073AC",
    "gatherInputData": "#E08214",
    "aix_collective_gather_mpi": "#E08214",
    "inferenceDevice": "#D73027",
    "aix_controller_inference": "#D73027",
    "h2d_copy": "#FEE090",
    "torchInference::forward": "#A50026",
    "d2h_copy": "#8073AC",
    "scatterOutputData": "#542788",
    "aix_collective_scatter_mpi": "#542788",
    "aix_collective_worker_wait_for_controller": "#B2ABD2",
    "phydll_prepack": "#FEE090",
    "phydll_send": "#E08214",
    "phydll_recv": "#D73027",
    "phydll_unpack": "#8073AC",
    "dl_recv": "#FDAE61",
    "dl_input_unpack": "#FEE090",
    "dl_h2d": "#E08214",
    "dl_torch_forward": "#D73027",
    "dl_d2h": "#8073AC",
    "dl_output_reorder": "#542788",
    "dl_send": "#2D004B",
    "py_recv": "#FDAE61",
    "py_input_unpack": "#FEE090",
    "py_h2d": "#E08214",
    "py_torch_forward": "#D73027",
    "py_d2h": "#8073AC",
    "py_output_reorder": "#542788",
    "py_send": "#2D004B",
    "Postprocess": "#74ADD1",       # Muted Blue
    "app_finalize_output": "#74ADD1",
    "flowex_reconstruct_output": "#ABD9E9",
    "Other": "#D9D9D9",
}

def get_color(name: str) -> str:
    clean = name.removeprefix("user:")
    for key, col in COLOR_MAP.items():
        if key in clean or clean in key:
            return col
    return "#CCCCCC"


# =============================================================================
# CUBE4 Profile Parsing via cube_dump
# =============================================================================

def find_cube_dump() -> str:
    # Check PATH first
    cmd = shutil_which("cube_dump")
    if cmd:
        return cmd
    
    # Check well-known local paths in this repo / environment
    known_paths = [
        "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump",
        "/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump",
    ]
    for p in known_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return "cube_dump"

def check_cube_dump_available() -> bool:
    cmd = find_cube_dump()
    try:
        subprocess.run([cmd, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def parse_cubex_metric(cubex_path: Path, metric: str = "time") -> Dict[str, Dict[str, Any]]:
    """
    Parses region values from a CUBE4 profile via `cube_dump -m <metric> -s human <cubex>`.
    Returns dict mapping region_name -> { 'per_rank': [...], 'mean': ..., 'max': ..., 'sum': ... }
    """
    if not cubex_path.exists():
        raise FileNotFoundError(f"CUBE profile not found: {cubex_path}")
    
    dump_bin = find_cube_dump()
    cmd = [dump_bin, "-m", metric, "-s", "human", str(cubex_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[WARN] cube_dump failed on {cubex_path}: {e.stderr}\n")
        return {}
    except FileNotFoundError:
        sys.stderr.write("[ERROR] `cube_dump` command not found in PATH. Ensure Score-P/CubeLib modules are loaded.\n")
        return {}

    regions: Dict[str, Dict[str, Any]] = {}
    for line in res.stdout.splitlines():
        m = re.match(r'^\s*([a-zA-Z0-9_:<>*&-]+)\s*\(id=\d+\)\s+(.*)$', line)
        if m:
            reg = m.group(1).strip()
            raw_vals = m.group(2).split()
            try:
                vals = [float(v) for v in raw_vals]
            except ValueError:
                continue
            if vals:
                arr = np.array(vals)
                regions[reg] = {
                    "per_rank": vals,
                    "mean": float(np.mean(arr)),
                    "max": float(np.max(arr)),
                    "min": float(np.min(arr)),
                    "sum": float(np.sum(arr)),
                    "num_ranks": len(vals)
                }
    return regions

def parse_multiple_cubex_metrics(cubex_paths: List[Path], metric: str = "time") -> Dict[str, Dict[str, Any]]:
    """
    Parses and aggregates multiple per-rank .cubex profiles into a unified multi-rank dataset.
    """
    all_regions: Dict[str, List[float]] = {}
    for p in cubex_paths:
        r_data = parse_cubex_metric(p, metric)
        for reg, d in r_data.items():
            if reg not in all_regions:
                all_regions[reg] = []
            all_regions[reg].extend(d.get("per_rank", []))

    combined: Dict[str, Dict[str, Any]] = {}
    for reg, vals in all_regions.items():
        if vals:
            arr = np.array(vals)
            combined[reg] = {
                "per_rank": vals,
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "sum": float(np.sum(arr)),
                "num_ranks": len(vals)
            }
    return combined

def parse_cubex_calltree(cubex_path: Path) -> List[Dict[str, Any]]:
    """
    Parses call tree structure from `cube_dump -c <cubex>`.
    """
    if not cubex_path.exists():
        return []
    
    cmd = ["cube_dump", "-c", str(cubex_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return []

    calltree: List[Dict[str, Any]] = []
    # Lines look like:
    # 0 root
    #   1 subregion
    #     2 child
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            cnode_id = parts[0]
            region_name = parts[1].strip()
            depth = indent // 2
            calltree.append({
                "cnode_id": cnode_id,
                "region": region_name,
                "depth": depth
            })
    return calltree


# =============================================================================
# Structured Profile Model & Hierarchy
# =============================================================================

class CouplingHierarchyNode:
    def __init__(self, name: str, value: float, children: Optional[List['CouplingHierarchyNode']] = None, label: Optional[str] = None):
        self.name = name
        self.value = value
        self.children = children or []
        self.label = label or name

def build_normalized_hierarchy(region_data: Dict[str, Dict[str, Any]], rank_agg: str = "mean") -> CouplingHierarchyNode:
    """
    Constructs a normalized 3-phase coupling hierarchy:
      Total Coupling
        -> Preprocess (app_prepare_input)
        -> Provider Step (app_provider_inference / *library_static_step)
             -> sub-operations (put, wait, run, unpack, gather, forward, etc.)
        -> Postprocess (app_finalize_output)
    """
    def val(reg_name: str) -> float:
        if reg_name in region_data:
            return region_data[reg_name].get(rank_agg, 0.0)
        user_reg = "user:" + reg_name
        if user_reg in region_data:
            return region_data[user_reg].get(rank_agg, 0.0)
        # Search partial matches
        for k, d in region_data.items():
            k_clean = k.removeprefix("user:")
            if reg_name == k_clean or k_clean.endswith("::" + reg_name):
                return d.get(rank_agg, 0.0)
        return 0.0

    prep_val = val("app_prepare_input") or val("flowex_extract_cubes") or val("solver_ml_prepare_input")
    prep_node = CouplingHierarchyNode("Preprocess", prep_val, label=f"Preprocess\n({prep_val*1000:.2f} ms)" if prep_val > 0 else "Preprocess")

    post_val = val("app_finalize_output") or val("flowex_reconstruct_output") or val("solver_ml_output_copy") or val("solver_ml_apply_output")
    post_node = CouplingHierarchyNode("Postprocess", post_val, label=f"Postprocess\n({post_val*1000:.2f} ms)" if post_val > 0 else "Postprocess")

    # Detect Provider Sub-operations
    provider_children: List[CouplingHierarchyNode] = []
    
    # 1. SmartSim
    smartsim_step = val("smartsim_library_static_step") or val("app_provider_inference")
    if "smartsim_put_tensor" in region_data or "smartsim_run_model" in region_data:
        wait_val = val("smartsim_token_wait")
        if wait_val > 0:
            provider_children.append(CouplingHierarchyNode("smartsim_token_wait", wait_val, label=f"Token Wait\n({wait_val*1000:.2f} ms)"))
        
        plan_val = val("smartsim_chunk_plan")
        if plan_val > 0:
            provider_children.append(CouplingHierarchyNode("smartsim_chunk_plan", plan_val, label=f"Chunk Plan\n({plan_val*1000:.2f} ms)"))
            
        put_val = val("smartsim_put_tensor")
        if put_val > 0:
            provider_children.append(CouplingHierarchyNode("smartsim_put_tensor", put_val, label=f"Put Tensor\n({put_val*1000:.2f} ms)"))
            
        run_val = val("smartsim_run_model")
        if run_val > 0:
            provider_children.append(CouplingHierarchyNode("smartsim_run_model", run_val, label=f"Run Model\n({run_val*1000:.2f} ms)"))
            
        unpack_val = val("smartsim_unpack_tensor")
        if unpack_val > 0:
            provider_children.append(CouplingHierarchyNode("smartsim_unpack_tensor", unpack_val, label=f"Unpack Tensor\n({unpack_val*1000:.2f} ms)"))

    # 2. AIx Collective
    elif "gatherInputData" in region_data or "aix_collective_gather_mpi" in region_data or "aix_controller_inference" in region_data:
        gather_val = val("aix_collective_gather_mpi") or val("gatherInputData")
        if gather_val > 0:
            provider_children.append(CouplingHierarchyNode("gatherInputData", gather_val, label=f"MPI Gather\n({gather_val*1000:.2f} ms)"))
            
        # Inference internals
        infer_children: List[CouplingHierarchyNode] = []
        h2d_val = val("h2d_copy")
        if h2d_val > 0:
            infer_children.append(CouplingHierarchyNode("h2d_copy", h2d_val, label=f"H2D\n({h2d_val*1000:.2f} ms)"))
        fwd_val = val("torchInference::forward")
        if fwd_val > 0:
            infer_children.append(CouplingHierarchyNode("torchInference::forward", fwd_val, label=f"Forward\n({fwd_val*1000:.2f} ms)"))
        d2h_val = val("d2h_copy")
        if d2h_val > 0:
            infer_children.append(CouplingHierarchyNode("d2h_copy", d2h_val, label=f"D2H\n({d2h_val*1000:.2f} ms)"))
            
        inf_val = val("aix_controller_inference") or val("inferenceDevice") or val("torchInference::inference")
        provider_children.append(CouplingHierarchyNode("inferenceDevice", inf_val if inf_val > 0 else (h2d_val + fwd_val + d2h_val), infer_children, label=f"Device Inf\n({inf_val*1000:.2f} ms)" if inf_val > 0 else "Device Inf"))

        scatter_val = val("aix_collective_scatter_mpi") or val("scatterOutputData")
        if scatter_val > 0:
            provider_children.append(CouplingHierarchyNode("scatterOutputData", scatter_val, label=f"MPI Scatter\n({scatter_val*1000:.2f} ms)"))

    # 3. PhyDLL Solver
    elif "phydll_send" in region_data or "phydll_recv" in region_data:
        prepack_val = val("phydll_prepack")
        if prepack_val > 0:
            provider_children.append(CouplingHierarchyNode("phydll_prepack", prepack_val, label=f"Prepack\n({prepack_val*1000:.2f} ms)"))
        send_val = val("phydll_send")
        if send_val > 0:
            provider_children.append(CouplingHierarchyNode("phydll_send", send_val, label=f"PhyDLL Send\n({send_val*1000:.2f} ms)"))
        recv_val = val("phydll_recv")
        if recv_val > 0:
            provider_children.append(CouplingHierarchyNode("phydll_recv", recv_val, label=f"PhyDLL Recv\n({recv_val*1000:.2f} ms)"))
        unp_val = val("phydll_unpack")
        if unp_val > 0:
            provider_children.append(CouplingHierarchyNode("phydll_unpack", unp_val, label=f"Unpack\n({unp_val*1000:.2f} ms)"))

    # Fallback provider value
    step_val = val("app_provider_inference") or val("smartsim_library_static_step") or val("aix_library_static_step") or val("phydll_library_static_step") or val("solver_ml_provider_call")
    if step_val == 0.0 and provider_children:
        step_val = sum(c.value for c in provider_children)
        
    prov_node = CouplingHierarchyNode("Provider Step", step_val, provider_children, label=f"Provider Step\n({step_val*1000:.2f} ms)" if step_val > 0 else "Provider Step")

    total_val = prep_val + step_val + post_val
    root = CouplingHierarchyNode("Total Coupling Step", total_val, [prep_node, prov_node, post_node], label=f"Total Coupling Step\n({total_val*1000:.2f} ms)")
    return root


# =============================================================================
# Icicle Plot Generation
# =============================================================================

def plot_icicle(root: CouplingHierarchyNode, output_path: Path, title: str = "CMI Coupling Phase Breakdown"):
    """
    Renders a multi-level Icicle plot representing hierarchical time decomposition.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate levels and layouts
    # We assign y-coordinates based on depth (Root=0, Phase=1, Subphase=2, Sub-subphase=3)
    max_depth = 3
    row_height = 0.8
    y_gap = 0.2

    def render_node(node: CouplingHierarchyNode, x_start: float, width: float, depth: int):
        if width <= 0 or depth > max_depth:
            return
        
        y_pos = (max_depth - depth) * (row_height + y_gap)
        color = get_color(node.name)
        
        rect = patches.FancyBboxPatch(
            (x_start, y_pos), width, row_height,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=1.2, edgecolor="#333333", facecolor=color, alpha=0.92
        )
        ax.add_patch(rect)

        # Label if wide enough
        if width > 0.05 * root.value and node.value > 0:
            pct = (node.value / root.value) * 100.0 if root.value > 0 else 0.0
            lbl = f"{node.name}\n{node.value*1000:.2f} ms\n({pct:.1f}%)"
            fontsize = 9 if width > 0.15 * root.value else 7.5
            ax.text(
                x_start + width / 2.0, y_pos + row_height / 2.0,
                lbl, ha="center", va="center", fontsize=fontsize, weight="bold", color="#111111"
            )
        elif width > 0.02 * root.value and node.value > 0:
            ax.text(
                x_start + width / 2.0, y_pos + row_height / 2.0,
                f"{node.value*1000:.1f}ms", ha="center", va="center", fontsize=7, color="#222222"
            )

        # Render children
        if node.children:
            child_sum = sum(c.value for c in node.children)
            curr_x = x_start
            for child in node.children:
                # Scale child relative to node width or total
                child_w = (child.value / max(node.value, child_sum, 1e-9)) * width if node.value > 0 else 0.0
                render_node(child, curr_x, child_w, depth + 1)
                curr_x += child_w

    render_node(root, 0.0, root.value if root.value > 0 else 1.0, 0)

    ax.set_xlim(-0.02 * root.value, 1.02 * root.value if root.value > 0 else 1.0)
    ax.set_ylim(-0.2, (max_depth + 1) * (row_height + y_gap))
    ax.set_xlabel(f"Aggregated Duration ({'seconds' if root.value > 1.0 else 's'})", weight="bold")
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, weight="bold", pad=15)
    
    # Clean grid/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Icicle plot to: {output_path}")


# =============================================================================
# Stage Breakdown Bar Plot
# =============================================================================

def plot_stage_breakdown(region_data: Dict[str, Dict[str, Any]], output_path: Path, title: str = "CMI Phase Breakdown"):
    """
    Renders horizontal stacked and grouped bar charts of detected CMI phases.
    """
    phases = []
    times = []
    colors = []
    
    # Priority ordered list of interesting regions
    candidates = [
        "app_prepare_input",
        "smartsim_token_wait",
        "smartsim_put_tensor",
        "smartsim_run_model",
        "smartsim_unpack_tensor",
        "gatherInputData",
        "aix_collective_gather_mpi",
        "h2d_copy",
        "torchInference::forward",
        "d2h_copy",
        "scatterOutputData",
        "aix_collective_scatter_mpi",
        "phydll_prepack",
        "phydll_send",
        "phydll_recv",
        "phydll_unpack",
        "dl_recv",
        "dl_input_unpack",
        "dl_h2d",
        "dl_torch_forward",
        "dl_d2h",
        "dl_output_reorder",
        "dl_send",
        "py_recv",
        "py_input_unpack",
        "py_h2d",
        "py_torch_forward",
        "py_d2h",
        "py_output_reorder",
        "py_send",
        "app_finalize_output",
    ]
    
    for c in candidates:
        val = 0.0
        if c in region_data:
            val = region_data[c].get("mean", 0.0) * 1000.0
        elif "user:" + c in region_data:
            val = region_data["user:" + c].get("mean", 0.0) * 1000.0
        if val > 0.001:
            phases.append(c)
            times.append(val)
            colors.append(get_color(c))

    if not phases:
        # Fallback to any region
        for k, v in region_data.items():
            if k.startswith(("app_", "smartsim_", "aix_", "phydll_", "dl_", "py_")):
                val = v.get("mean", 0.0) * 1000.0
                if val > 0.001:
                    phases.append(k)
                    times.append(val)
                    colors.append(get_color(k))

    if not phases:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(phases) * 0.45)))
    y_pos = np.arange(len(phases))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="#333333", height=0.65, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases, fontsize=9, weight="bold")
    ax.invert_yaxis()  # Top-down
    ax.set_xlabel("Mean Duration (ms)", weight="bold")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)

    # Value annotations
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + (max(times) * 0.015),
            bar.get_y() + bar.get_height() / 2.0,
            f"{t:.2f} ms",
            va="center", ha="left", fontsize=8.5, weight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved breakdown bar plot to: {output_path}")


# =============================================================================
# AIx P2P Pipelined Timeline & Gantt Analysis
# =============================================================================

def parse_aix_p2p_timeline(timeline_dir: Path) -> pd.DataFrame:
    """
    Loads all `aix_p2p_timeline_rank_*.csv` files and aligns events.
    """
    csv_files = sorted(timeline_dir.glob("aix_p2p_timeline_rank_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            sys.stderr.write(f"[WARN] Error reading {f}: {e}\n")

    if not dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["time_s"] = pd.to_numeric(full_df["time_s"], errors="coerce")
    full_df = full_df.sort_values(by=["step", "time_s"]).reset_index(drop=True)
    return full_df

def plot_aix_pipeline_gantt(df: pd.DataFrame, output_path: Path, target_step: Optional[int] = None):
    """
    Generates a Gantt/timeline chart of pipelined P2P operations across ranks for a given step.
    """
    if df.empty:
        return

    steps = df["step"].unique()
    if target_step is None or target_step not in steps:
        target_step = steps[-1]  # Latest / warm step

    step_df = df[df["step"] == target_step].copy()
    if step_df.empty:
        return

    # Normalize time to start of step
    t_min = step_df["time_s"].min()
    step_df["rel_time_ms"] = (step_df["time_s"] - t_min) * 1000.0

    fig, ax = plt.subplots(figsize=(13, 7))

    ranks = sorted(step_df["world_rank"].unique())
    y_map = {r: i for i, r in enumerate(ranks)}

    # Track intervals per rank
    for rank in ranks:
        r_df = step_df[step_df["world_rank"] == rank].sort_values(by="rel_time_ms")
        y = y_map[rank]

        # Draw worker sends/receives
        sends = r_df[r_df["event"] == "input_send_start"]
        s_ends = r_df[r_df["event"] == "input_send_complete"]
        for _, s in sends.iterrows():
            # Find matching complete
            end_match = s_ends[s_ends["rel_time_ms"] >= s["rel_time_ms"]]
            t_end = end_match.iloc[0]["rel_time_ms"] if not end_match.empty else s["rel_time_ms"] + 1.0
            ax.barh(y, t_end - s["rel_time_ms"], left=s["rel_time_ms"], height=0.5, color="#E08214", edgecolor="black", alpha=0.85, label="P2P Send" if rank == ranks[0] else "")

        # Controller range inferences
        inf_starts = r_df[r_df["event"] == "range_inference_start"]
        inf_ends = r_df[r_df["event"] == "range_inference_end"]
        for _, inf in inf_starts.iterrows():
            end_match = inf_ends[inf_ends["rel_time_ms"] >= inf["rel_time_ms"]]
            t_end = end_match.iloc[0]["rel_time_ms"] if not end_match.empty else inf["rel_time_ms"] + 2.0
            rng = f"[{int(inf.get('range_first_rank', 0))}..{int(inf.get('range_end_rank', 0))})]"
            ax.barh(y, t_end - inf["rel_time_ms"], left=inf["rel_time_ms"], height=0.5, color="#D73027", edgecolor="black", alpha=0.9, label="Range Inference" if rank == ranks[0] else "")
            ax.text((inf["rel_time_ms"] + t_end) / 2.0, y, rng, ha="center", va="center", fontsize=7, color="white", weight="bold")

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([f"Rank {r}" + (" (Ctrl)" if step_df[step_df['world_rank']==r]['is_controller'].any() else "") for r in ranks], weight="bold")
    ax.set_xlabel("Relative Time (ms)", weight="bold")
    ax.set_title(f"AIx Pipelined P2P Overlap Timeline (Step {target_step})", fontsize=13, weight="bold", pad=15)
    
    # Legend deduplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved AIx Pipeline Gantt chart to: {output_path}")


# =============================================================================
# Markdown Summary & CSV Export
# =============================================================================

def export_summary(region_data: Dict[str, Dict[str, Any]], out_dir: Path, title: str = "CMI Profile Summary"):
    """
    Exports summary tables to CSV and Markdown.
    """
    rows = []
    for reg, d in sorted(region_data.items()):
        rows.append({
            "Region": reg,
            "Mean (ms)": d.get("mean", 0.0) * 1000.0,
            "Max (ms)": d.get("max", 0.0) * 1000.0,
            "Min (ms)": d.get("min", 0.0) * 1000.0,
            "Sum (ms)": d.get("sum", 0.0) * 1000.0,
            "Ranks": d.get("num_ranks", 0),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "cmi_phase_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved phase summary CSV to: {csv_path}")

    md_path = out_dir / "cmi_profile_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| Region | Mean (ms) | Max (ms) | Min (ms) | Sum (ms) | Ranks |\n")
        f.write("|:---|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            f.write(f"| `{r['Region']}` | {r['Mean (ms)']:.3f} | {r['Max (ms)']:.3f} | {r['Min (ms)']:.3f} | {r['Sum (ms)']:.3f} | {int(r['Ranks'])} |\n")
    print(f"[+] Saved summary Markdown to: {md_path}")


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CMI Score-P CUBE and AIx Timeline Analysis Tool"
    )
    parser.add_argument("--cubex", type=Path, default=None, help="Path to profile.cubex or directory containing it")
    parser.add_argument("--dl-cubex", type=Path, default=None, help="Optional DL-side profile.cubex for MPMD PhyDLL")
    parser.add_argument("--p2p-timeline-dir", type=Path, default=None, help="Path to directory containing aix_p2p_timeline_rank_*.csv")
    parser.add_argument("--metric", type=str, default="time", help="Metric to extract from CUBE (default: time)")
    parser.add_argument("--rank-agg", type=str, choices=["mean", "max", "min", "sum"], default="mean", help="Rank aggregation method (default: mean)")
    parser.add_argument("--output-dir", type=Path, default=Path("./cmi_profile_analysis"), help="Output directory for plots and reports")
    parser.add_argument("--title", type=str, default=None, help="Plot title override")
    parser.add_argument("--no-plots", action="store_true", help="Skip rendering plots and only generate CSV/MD reports")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process CUBE Profile
    cubex_files: List[Path] = []
    if args.cubex:
        p_str = str(args.cubex)
        if "*" in p_str or "?" in p_str:
            matched = [Path(p) for p in glob.glob(p_str)]
            for m in matched:
                if m.is_dir() and (m / "profile.cubex").exists():
                    cubex_files.append(m / "profile.cubex")
                elif m.is_file() and m.suffix == ".cubex":
                    cubex_files.append(m)
        elif args.cubex.is_dir():
            # Check if directory has rank subdirectories
            rank_dirs = sorted(args.cubex.glob("*_rank_*"))
            if rank_dirs:
                for rd in rank_dirs:
                    if (rd / "profile.cubex").exists():
                        cubex_files.append(rd / "profile.cubex")
            elif (args.cubex / "profile.cubex").exists():
                cubex_files.append(args.cubex / "profile.cubex")
        elif args.cubex.is_file():
            cubex_files.append(args.cubex)

    if cubex_files:
        print(f"[*] Analyzing {len(cubex_files)} CUBE profile(s): {[str(f) for f in cubex_files]}")
        if len(cubex_files) == 1:
            region_data = parse_cubex_metric(cubex_files[0], args.metric)
        else:
            region_data = parse_multiple_cubex_metrics(cubex_files, args.metric)

        if region_data:
            title = args.title or f"CMI Coupling Profile ({cubex_files[0].parent.parent.name if len(cubex_files)>1 else cubex_files[0].parent.name})"
            export_summary(region_data, args.output_dir, title=title)
            
            if not args.no_plots:
                hierarchy = build_normalized_hierarchy(region_data, rank_agg=args.rank_agg)
                plot_icicle(hierarchy, args.output_dir / "cmi_icicle_plot.png", title=f"{title} - Hierarchical Calltree")
                plot_stage_breakdown(region_data, args.output_dir / "cmi_stage_breakdown.png", title=f"{title} - Phase Durations")

    # 2. Process AIx P2P Timeline CSVs
    if args.p2p_timeline_dir and args.p2p_timeline_dir.exists():
        print(f"[*] Analyzing AIx P2P Timeline in: {args.p2p_timeline_dir}")
        p2p_df = parse_aix_p2p_timeline(args.p2p_timeline_dir)
        if not p2p_df.empty and not args.no_plots:
            plot_aix_pipeline_gantt(p2p_df, args.output_dir / "aix_pipeline_gantt.png")

    print("[✓] Analysis complete.")

if __name__ == "__main__":
    main()
