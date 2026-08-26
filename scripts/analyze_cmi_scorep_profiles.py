#!/usr/bin/env python3
"""
analyze_cmi_scorep_profiles.py
==============================
Generalized CLI analysis and visualization tool for CMI (C++ ML Interface)
Score-P CUBE profiles and AIx P2P timeline CSVs.

Features:
- Extracts hierarchical calltree profiles from .cubex files via `cube_dump`.
- Computes true call-path inclusive, exclusive, and per-step normalized metrics.
- Aggregates metrics across MPI ranks (mean, min, max, std, sum).
- Renders true hierarchical Icicle plots (where children fit within parent width).
- Renders phase breakdown bar charts (steady step decomposition).
- Generates AIx P2P timeline Gantt charts and makespan/overlap analyses.
- Exports comprehensive Markdown summaries and CSV reports.

Usage:
  python3 analyze_cmi_scorep_profiles.py --cubex "scorep_runs/*_rank_*" --output-dir ./analysis
  python3 analyze_cmi_scorep_profiles.py --cubex path/to/profile.cubex --dl-cubex path/to/dl/profile.cubex
  python3 analyze_cmi_scorep_profiles.py --p2p-timeline-dir path/to/timeline_dir
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

# Canonical color palette for coupling operations
COLOR_MAP = {
    # Application level
    "solver_ml_provider_call": "#2B5C8F",
    "app_prepare_input": "#4393C3",
    "flowex_extract_cubes": "#92C5DE",
    "app_provider_inference": "#D6604D",
    "app_finalize_output": "#74ADD1",
    "flowex_reconstruct_output": "#ABD9E9",
    
    # SmartSim
    "smartsim_library_static_step": "#F46D43",
    "smartsim_token_wait": "#FDAE61",
    "smartsim_chunk_plan": "#FEE090",
    "smartsim_put_tensor": "#E08214",
    "smartsim_run_model": "#D73027",
    "smartsim_unpack_tensor": "#8073AC",
    
    # AIx Collective
    "aix_library_static_step": "#F46D43",
    "aix_provider_setup": "#FEE090",
    "aix_inference": "#D73027",
    "aix_collective_setup": "#FEE090",
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
    "aix_collective_post_scatter": "#D9D9D9",
    
    # AIx Pipelined
    "p2p_pipelined_exchange": "#E08214",
    "aix_controller_pipelined_inference": "#D73027",
    
    # PhyDLL Solver Side
    "phydll_library_static_step": "#F46D43",
    "phydll_prepack": "#FEE090",
    "phydll_send": "#E08214",
    "phydll_recv": "#D73027",
    "phydll_unpack": "#8073AC",
    
    # PhyDLL DL Side (C++ & Python)
    "dl_recv": "#FDAE61",
    "py_recv": "#FDAE61",
    "dl_frame_copy": "#FEE090",
    "dl_input_allocate": "#E0E0E0",
    "dl_input_unpack": "#E08214",
    "py_input_unpack": "#E08214",
    "dl_h2d": "#FEE090",
    "py_h2d": "#FEE090",
    "dl_torch_forward": "#D73027",
    "py_torch_forward": "#D73027",
    "dl_d2h": "#8073AC",
    "py_d2h": "#8073AC",
    "dl_output_allocate": "#E0E0E0",
    "dl_output_reorder": "#542788",
    "py_output_reorder": "#542788",
    "dl_send": "#2D004B",
    "dl_send_output": "#2D004B",
    "py_send": "#2D004B",
    "py_inference": "#D73027",
    
    # Residual
    "Self / Overhead": "#BDBDBD",
}

def get_color(name: str) -> str:
    clean = name.removeprefix("user:")
    for key, col in COLOR_MAP.items():
        if key == clean or clean.endswith("::" + key) or key.endswith("::" + clean):
            return col
    for key, col in COLOR_MAP.items():
        if key in clean or clean in key:
            return col
    return "#CCCCCC"


# =============================================================================
# CUBE4 Profile Call Tree Parsing via cube_dump
# =============================================================================

def find_cube_dump() -> str:
    cmd = shutil_which("cube_dump")
    if cmd:
        return cmd
    known_paths = [
        "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump",
        "/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump",
    ]
    for p in known_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return "cube_dump"

class CallTreeNode:
    def __init__(self, cid: int, name: str, parent_id: Optional[int] = None):
        self.id = cid
        self.name = name.removeprefix("user:")
        self.raw_name = name
        self.parent_id = parent_id
        self.children_ids: List[int] = []
        
        # Raw metrics for this callpath
        self.incl_time: float = 0.0
        self.excl_time: float = 0.0
        self.visits: int = 0
        
        # Per-step normalized values
        self.step_incl_time: float = 0.0
        self.step_excl_time: float = 0.0
        self.step_self_time: float = 0.0

def parse_single_cubex_tree(cubex_path: Path) -> Dict[int, CallTreeNode]:
    """
    Parses full call-tree structure and extracts inclusive/exclusive time and visits.
    """
    if not cubex_path.exists():
        raise FileNotFoundError(f"CUBE profile not found: {cubex_path}")
    
    dump_bin = find_cube_dump()
    
    # 1. Calltree structure
    try:
        tree_out = subprocess.check_output([dump_bin, "-w", "calltree", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
        incl_out = subprocess.check_output([dump_bin, "-m", "time", "-z", "incl", "-s", "human", "-c", "all", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
        excl_out = subprocess.check_output([dump_bin, "-m", "time", "-z", "excl", "-s", "human", "-c", "all", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
        vis_out = subprocess.check_output([dump_bin, "-m", "visits", "-z", "excl", "-s", "human", "-c", "all", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to run cube_dump on {cubex_path}: {e}\n")
        return {}

    def parse_data_map(raw: str) -> Dict[int, float]:
        res: Dict[int, float] = {}
        for line in raw.splitlines():
            m = re.match(r"^\s*([a-zA-Z0-9_:<>*&-]+)\(id=(\d+)\)\s+([\d\.eE+-]+)", line)
            if m:
                cid = int(m.group(2))
                val = float(m.group(3))
                res[cid] = val
        return res

    incl_map = parse_data_map(incl_out)
    excl_map = parse_data_map(excl_out)
    vis_map = parse_data_map(vis_out)

    nodes: Dict[int, CallTreeNode] = {}
    parent_stack: List[Tuple[int, int]] = [] # (depth, cid)

    calltree_section = False
    for line in tree_out.splitlines():
        if "--- CALL TREE ---" in line:
            calltree_section = True
            continue
        if not calltree_section or not line.strip():
            continue
        
        m = re.match(r"^(\s*\|*-*\s*)*([a-zA-Z0-9_:<>*&-]+)\s*\[\s*\(\s*id=(\d+)", line)
        if m:
            reg_name = m.group(2)
            cid = int(m.group(3))
            prefix = line[:line.find(reg_name)]
            depth = len(prefix.replace("-", " ").replace("|", " "))
            
            while parent_stack and parent_stack[-1][0] >= depth:
                parent_stack.pop()
            parent_id = parent_stack[-1][1] if parent_stack else None
            parent_stack.append((depth, cid))
            
            node = CallTreeNode(cid, reg_name, parent_id)
            node.incl_time = incl_map.get(cid, 0.0)
            node.excl_time = excl_map.get(cid, 0.0)
            node.visits = int(vis_map.get(cid, 0))
            
            nodes[cid] = node
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].children_ids.append(cid)

    # Determine normalization base (visits of steady ML step)
    norm_visits = 1
    for n in nodes.values():
        if n.name in ("solver_step_ml_steady", "solver_ml_provider_call") and n.visits > 0:
            norm_visits = n.visits
            break

    # Compute per-step metrics
    for n in nodes.values():
        n.step_incl_time = n.incl_time / norm_visits
        n.step_excl_time = n.excl_time / norm_visits
        
        # Self-time is exclusive time
        child_incl_sum = sum(nodes[c].step_incl_time for c in n.children_ids if c in nodes)
        n.step_self_time = max(0.0, n.step_incl_time - child_incl_sum)

    return nodes


# =============================================================================
# Multi-Rank Profile Aggregator
# =============================================================================

class AggregatedCallTree:
    def __init__(self):
        # Callpath key: tuple of region names from root to node
        self.paths: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def add_tree(self, nodes: Dict[int, CallTreeNode]):
        if not nodes:
            return

        def get_path(cid: int) -> Tuple[str, ...]:
            p = []
            curr: Optional[int] = cid
            while curr is not None and curr in nodes:
                p.append(nodes[curr].name)
                curr = nodes[curr].parent_id
            return tuple(reversed(p))

        for cid, n in nodes.items():
            path = get_path(cid)
            if path not in self.paths:
                self.paths[path] = {
                    "name": n.name,
                    "parent_path": path[:-1] if len(path) > 1 else None,
                    "children_paths": set(),
                    "step_incl_list": [],
                    "step_excl_list": [],
                    "step_self_list": [],
                    "visits_list": []
                }
            self.paths[path]["step_incl_list"].append(n.step_incl_time)
            self.paths[path]["step_excl_list"].append(n.step_excl_time)
            self.paths[path]["step_self_list"].append(n.step_self_time)
            self.paths[path]["visits_list"].append(n.visits)

            if len(path) > 1:
                parent_path = path[:-1]
                if parent_path in self.paths:
                    self.paths[parent_path]["children_paths"].add(path)

    def compute_summary(self, rank_agg: str = "mean") -> Dict[Tuple[str, ...], Dict[str, Any]]:
        summary: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for path, d in self.paths.items():
            incls = np.array(d["step_incl_list"])
            excls = np.array(d["step_excl_list"])
            selfs = np.array(d["step_self_list"])
            vis = np.array(d["visits_list"])

            def agg_val(arr: np.ndarray) -> float:
                if len(arr) == 0: return 0.0
                if rank_agg == "mean": return float(np.mean(arr))
                elif rank_agg == "max": return float(np.max(arr))
                elif rank_agg == "min": return float(np.min(arr))
                elif rank_agg == "sum": return float(np.sum(arr))
                return float(np.mean(arr))

            summary[path] = {
                "name": d["name"],
                "parent_path": d["parent_path"],
                "children_paths": sorted(list(d["children_paths"])),
                "incl_mean": float(np.mean(incls)),
                "incl_max": float(np.max(incls)),
                "incl_min": float(np.min(incls)),
                "incl_std": float(np.std(incls)) if len(incls)>1 else 0.0,
                "incl_agg": agg_val(incls),
                "excl_mean": float(np.mean(excls)),
                "excl_agg": agg_val(excls),
                "self_mean": float(np.mean(selfs)),
                "self_agg": agg_val(selfs),
                "visits_mean": float(np.mean(vis)),
                "num_ranks": len(incls)
            }
        return summary


# =============================================================================
# True Icicle Plot Renderer
# =============================================================================

def render_icicle_plot(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]],
                       root_path: Tuple[str, ...],
                       output_path: Path,
                       title: str = "CMI Coupling Phase Breakdown"):
    """
    Renders a true Icicle plot where every child fits strictly within its parent rectangle.
    """
    if root_path not in tree_summary:
        # Search for closest match (e.g. solver_ml_provider_call or app_provider_inference or root)
        candidates = [p for p in tree_summary if p[-1] in ("solver_ml_provider_call", "app_provider_inference", "phydll_dl_client", "py_inference")]
        if candidates:
            root_path = candidates[0]
        else:
            root_path = list(tree_summary.keys())[0]

    root_info = tree_summary[root_path]
    root_val = root_info["incl_agg"]
    if root_val <= 0:
        root_val = 1e-6

    fig, ax = plt.subplots(figsize=(12, 6))

    row_height = 0.85
    y_gap = 0.15
    max_depth = 4

    # Recursive render
    def draw_node(path: Tuple[str, ...], x_start: float, width: float, depth: int):
        if width <= 0 or depth > max_depth or path not in tree_summary:
            return

        node_info = tree_summary[path]
        name = node_info["name"]
        y_pos = (max_depth - depth) * (row_height + y_gap)
        color = get_color(name)

        # Draw rectangle
        rect = patches.Rectangle(
            (x_start, y_pos), width, row_height,
            linewidth=1.0, edgecolor="#222222", facecolor=color, alpha=0.92
        )
        ax.add_patch(rect)

        # Label
        val_ms = node_info["incl_agg"] * 1000.0
        pct = (node_info["incl_agg"] / root_val) * 100.0
        
        clean_name = name.replace("smartsim_", "").replace("phydll_", "").replace("aix_", "").replace("solver_", "").replace("app_", "").replace("torchInference::", "")
        if width > 0.12 * root_val:
            lbl = f"{clean_name}\n{val_ms:.2f} ms ({pct:.1f}%)"
            ax.text(x_start + width / 2.0, y_pos + row_height / 2.0, lbl,
                    ha="center", va="center", fontsize=8.5, weight="bold", color="#111111")
        elif width > 0.04 * root_val:
            ax.text(x_start + width / 2.0, y_pos + row_height / 2.0, f"{clean_name}\n{val_ms:.1f}ms",
                    ha="center", va="center", fontsize=7.5, color="#111111")

        # Children
        children = node_info["children_paths"]
        if children and depth < max_depth:
            child_sum = sum(tree_summary[c]["incl_agg"] for c in children if c in tree_summary)
            curr_x = x_start
            for c in children:
                c_val = tree_summary[c]["incl_agg"]
                # Scale proportionally if child_sum > width
                c_width = (c_val / max(width, child_sum, 1e-9)) * width
                draw_node(c, curr_x, c_width, depth + 1)
                curr_x += c_width

            # If parent has notable self-time / residual
            residual = max(0.0, width - child_sum)
            if residual > 0.02 * root_val:
                res_y = (max_depth - (depth + 1)) * (row_height + y_gap)
                rect_res = patches.Rectangle(
                    (curr_x, res_y), residual, row_height,
                    linewidth=1.0, edgecolor="#555555", facecolor=COLOR_MAP["Self / Overhead"], alpha=0.7, linestyle=":"
                )
                ax.add_patch(rect_res)
                if residual > 0.06 * root_val:
                    ax.text(curr_x + residual / 2.0, res_y + row_height / 2.0,
                            f"Overhead\n{residual*1000:.2f} ms", ha="center", va="center", fontsize=7.5, color="#333333")

    draw_node(root_path, 0.0, root_val, 0)

    ax.set_xlim(-0.01 * root_val, 1.01 * root_val)
    ax.set_ylim(-0.2, (max_depth + 1) * (row_height + y_gap))
    ax.set_xlabel("Aggregated Duration (seconds per steady step)", weight="bold")
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, weight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved true Icicle plot to: {output_path}")


# =============================================================================
# Breakdown Bar Plot Renderer
# =============================================================================

def render_breakdown_bars(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]],
                          output_path: Path,
                          title: str = "CMI Phase Breakdown"):
    """
    Renders horizontal bar charts of key leaf and major phase durations per step.
    """
    phases = []
    times = []
    colors = []

    priority = [
        "app_prepare_input",
        "smartsim_token_wait",
        "smartsim_chunk_plan",
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
        "py_recv",
        "dl_input_unpack",
        "py_input_unpack",
        "dl_h2d",
        "py_h2d",
        "dl_torch_forward",
        "py_torch_forward",
        "dl_d2h",
        "py_d2h",
        "dl_output_reorder",
        "py_output_reorder",
        "dl_send",
        "py_send",
        "app_finalize_output"
    ]

    for p_name in priority:
        for path, info in tree_summary.items():
            if info["name"] == p_name:
                val = info["incl_agg"] * 1000.0
                if val > 0.001 and p_name not in phases:
                    phases.append(p_name)
                    times.append(val)
                    colors.append(get_color(p_name))

    if not phases:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(phases) * 0.45)))
    y_pos = np.arange(len(phases))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="#222222", height=0.65, alpha=0.9)
    ax.set_yticks(y_pos)
    clean_labels = [p.replace("smartsim_", "").replace("phydll_", "").replace("aix_", "").replace("app_", "").replace("torchInference::", "") for p in phases]
    ax.set_yticklabels(clean_labels, fontsize=9, weight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Duration per Steady Step (ms)", weight="bold")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)

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
    if df.empty:
        return

    steps = df["step"].unique()
    if target_step is None or target_step not in steps:
        target_step = steps[-1]

    step_df = df[df["step"] == target_step].copy()
    if step_df.empty:
        return

    t_min = step_df["time_s"].min()
    step_df["rel_time_ms"] = (step_df["time_s"] - t_min) * 1000.0

    fig, ax = plt.subplots(figsize=(13, 7))

    ranks = sorted(step_df["world_rank"].unique())
    y_map = {r: i for i, r in enumerate(ranks)}

    for rank in ranks:
        r_df = step_df[step_df["world_rank"] == rank].sort_values(by="rel_time_ms")
        y = y_map[rank]

        # Worker sends
        sends = r_df[r_df["event"] == "input_send_start"]
        s_ends = r_df[r_df["event"] == "input_send_complete"]
        for _, s in sends.iterrows():
            end_match = s_ends[s_ends["rel_time_ms"] >= s["rel_time_ms"]]
            t_end = end_match.iloc[0]["rel_time_ms"] if not end_match.empty else s["rel_time_ms"] + 0.5
            ax.barh(y, t_end - s["rel_time_ms"], left=s["rel_time_ms"], height=0.5, color="#E08214", edgecolor="black", alpha=0.85, label="P2P Send" if rank == ranks[0] else "")

        # Controller range inferences
        inf_starts = r_df[r_df["event"] == "range_inference_start"]
        inf_ends = r_df[r_df["event"] == "range_inference_end"]
        for _, inf in inf_starts.iterrows():
            end_match = inf_ends[inf_ends["rel_time_ms"] >= inf["rel_time_ms"]]
            t_end = end_match.iloc[0]["rel_time_ms"] if not end_match.empty else inf["rel_time_ms"] + 1.0
            rng = f"[{int(inf.get('range_first_rank', 0))}..{int(inf.get('range_end_rank', 0))})]"
            ax.barh(y, t_end - inf["rel_time_ms"], left=inf["rel_time_ms"], height=0.5, color="#D73027", edgecolor="black", alpha=0.9, label="Range Inference" if rank == ranks[0] else "")
            ax.text((inf["rel_time_ms"] + t_end) / 2.0, y, rng, ha="center", va="center", fontsize=7, color="white", weight="bold")

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([f"Rank {r}" + (" (Ctrl)" if step_df[step_df['world_rank']==r]['is_controller'].any() else "") for r in ranks], weight="bold")
    ax.set_xlabel("Relative Time (ms)", weight="bold")
    ax.set_title(f"AIx Pipelined P2P Overlap Timeline (Step {target_step})", fontsize=12, weight="bold", pad=12)

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

def export_summary_tables(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]], out_dir: Path, title: str = "CMI Profile Summary"):
    rows = []
    for path, d in sorted(tree_summary.items(), key=lambda x: x[1]["incl_mean"], reverse=True):
        rows.append({
            "Callpath": " -> ".join(path),
            "Region": d["name"],
            "Mean (ms)": d["incl_mean"] * 1000.0,
            "Max (ms)": d["incl_max"] * 1000.0,
            "Min (ms)": d["incl_min"] * 1000.0,
            "Std (ms)": d["incl_std"] * 1000.0,
            "Self (ms)": d["self_mean"] * 1000.0,
            "Visits/Step": d["visits_mean"],
            "Ranks": d["num_ranks"]
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "cmi_phase_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved phase summary CSV to: {csv_path}")

    md_path = out_dir / "cmi_profile_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| Region | Per-Step Mean (ms) | Max (ms) | Min (ms) | Self (ms) | Visits/Step | Ranks |\n")
        f.write("|:---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            f.write(f"| `{r['Region']}` | {r['Mean (ms)']:.3f} | {r['Max (ms)']:.3f} | {r['Min (ms)']:.3f} | {r['Self (ms)']:.3f} | {r['Visits/Step']:.1f} | {int(r['Ranks'])} |\n")
    print(f"[+] Saved summary Markdown to: {md_path}")


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CMI Score-P CUBE and AIx Timeline Analysis Tool")
    parser.add_argument("--cubex", type=str, default=None, help="Path or glob pattern for profile.cubex")
    parser.add_argument("--dl-cubex", type=str, default=None, help="Optional DL-side profile.cubex for MPMD PhyDLL")
    parser.add_argument("--p2p-timeline-dir", type=Path, default=None, help="Path to directory containing aix_p2p_timeline_rank_*.csv")
    parser.add_argument("--rank-agg", type=str, choices=["mean", "max", "min", "sum"], default="mean", help="Rank aggregation method (default: mean)")
    parser.add_argument("--output-dir", type=Path, default=Path("./cmi_profile_analysis"), help="Output directory for plots and reports")
    parser.add_argument("--title", type=str, default=None, help="Plot title override")
    parser.add_argument("--no-plots", action="store_true", help="Skip rendering plots and only generate CSV/MD reports")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Solver CUBE Profiles
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
        elif Path(p_str).is_dir():
            rank_dirs = sorted(Path(p_str).glob("*_rank_*"))
            if rank_dirs:
                for rd in rank_dirs:
                    if (rd / "profile.cubex").exists():
                        cubex_files.append(rd / "profile.cubex")
            elif (Path(p_str) / "profile.cubex").exists():
                cubex_files.append(Path(p_str) / "profile.cubex")
        elif Path(p_str).is_file():
            cubex_files.append(Path(p_str))

    if cubex_files:
        print(f"[*] Analyzing {len(cubex_files)} solver CUBE profile(s)...")
        agg = AggregatedCallTree()
        for f in cubex_files:
            nodes = parse_single_cubex_tree(f)
            agg.add_tree(nodes)

        tree_summary = agg.compute_summary(rank_agg=args.rank_agg)
        if tree_summary:
            title = args.title or f"CMI Coupling Profile ({cubex_files[0].parent.parent.name if len(cubex_files)>1 else cubex_files[0].parent.name})"
            export_summary_tables(tree_summary, args.output_dir, title=title)

            if not args.no_plots:
                # Find steady ML root
                steady_candidates = [p for p in tree_summary if p[-1] in ("solver_ml_provider_call", "app_provider_inference")]
                root_path = steady_candidates[0] if steady_candidates else list(tree_summary.keys())[0]
                render_icicle_plot(tree_summary, root_path, args.output_dir / "cmi_icicle_plot.png", title=f"{title} - Hierarchical Breakdown")
                render_breakdown_bars(tree_summary, args.output_dir / "cmi_stage_breakdown.png", title=f"{title} - Phase Durations")

    # 2. Process DL-side CUBE Profile if supplied
    dl_files: List[Path] = []
    if args.dl_cubex:
        p_str = str(args.dl_cubex)
        if "*" in p_str:
            matched = [Path(p) for p in glob.glob(p_str)]
            for m in matched:
                if m.is_dir() and (m / "profile.cubex").exists(): dl_files.append(m / "profile.cubex")
                elif m.is_file(): dl_files.append(m)
        elif Path(p_str).is_dir() and (Path(p_str) / "profile.cubex").exists():
            dl_files.append(Path(p_str) / "profile.cubex")
        elif Path(p_str).is_file():
            dl_files.append(Path(p_str))

    if dl_files:
        print(f"[*] Analyzing DL-side CUBE profile: {dl_files[0]}")
        dl_agg = AggregatedCallTree()
        for f in dl_files:
            dl_nodes = parse_single_cubex_tree(f)
            dl_agg.add_tree(dl_nodes)
        dl_summary = dl_agg.compute_summary(rank_agg=args.rank_agg)
        if dl_summary and not args.no_plots:
            dl_roots = [p for p in dl_summary if p[-1] in ("phydll_dl_client", "py_inference", "user:py_inference")]
            dl_root = dl_roots[0] if dl_roots else list(dl_summary.keys())[0]
            render_icicle_plot(dl_summary, dl_root, args.output_dir / "phydll_dl_icicle_plot.png", title="PhyDLL DL-Side Breakdown")
            render_breakdown_bars(dl_summary, args.output_dir / "phydll_dl_stage_breakdown.png", title="PhyDLL DL-Side Phase Durations")

    # 3. Process AIx P2P Timeline CSVs
    if args.p2p_timeline_dir and args.p2p_timeline_dir.exists():
        print(f"[*] Analyzing AIx P2P Timeline in: {args.p2p_timeline_dir}")
        p2p_df = parse_aix_p2p_timeline(args.p2p_timeline_dir)
        if not p2p_df.empty and not args.no_plots:
            plot_aix_pipeline_gantt(p2p_df, args.output_dir / "aix_pipeline_gantt.png")

    print("[✓] Analysis complete.")

if __name__ == "__main__":
    main()
