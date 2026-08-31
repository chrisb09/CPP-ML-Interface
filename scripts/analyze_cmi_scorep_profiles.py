#!/usr/bin/env python3
"""
analyze_cmi_scorep_profiles.py
==============================
Generalized CLI analysis and visualization tool for CMI (C++ ML Interface)
Score-P CUBE profiles and AIx P2P timeline CSVs.

Features:
- Extracts hierarchical calltree profiles from .cubex files via `cube_dump`.
- Computes exact call-path inclusive, exclusive, and per-step normalized metrics.
- Separates one-off lifecycle phases (setup, warmup, teardown) from steady-state ML steps.
- Supports communicator-wide rank averaging (consistent tree algebra) and critical-rank metrics.
- Renders true hierarchical Icicle plots (where children strictly fit within parent width).
- Renders non-overlapping leaf decomposition breakdown bar charts.
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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    "phydll_dl_client": "#2B5C8F",
    "dl_recv": "#FDAE61",
    "py_recv": "#FDAE61",
    "dl_frame_copy": "#FEE090",
    "dl_input_allocate": "#E0E0E0",
    "dl_input_unpack": "#E08214",
    "py_input_unpack": "#E08214",
    "dl_h2d": "#FEE090",
    "py_h2d": "#FEE090",
    "dl_torch_forward": "#A50026",
    "py_torch_forward": "#A50026",
    "dl_d2h": "#8073AC",
    "py_d2h": "#8073AC",
    "dl_output_allocate": "#E0E0E0",
    "dl_output_reorder": "#542788",
    "py_output_reorder": "#542788",
    "dl_send": "#2D004B",
    "dl_send_output": "#2D004B",
    "py_send": "#2D004B",
    "py_inference": "#D73027",
    
    # Residual / Overhead
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
        
        # Raw metrics for this callpath (in seconds and counts)
        self.incl_time: float = 0.0
        self.excl_time: float = 0.0
        self.visits: int = 0
        
        # Normalized per-step metrics
        self.step_incl_time: float = 0.0
        self.step_excl_time: float = 0.0
        self.step_self_time: float = 0.0
        self.is_lifecycle: bool = False

def parse_cubex_file(cubex_path: Path, explicit_steady_steps: Optional[int] = None) -> List[Dict[int, CallTreeNode]]:
    """
    Parses full call-tree structure and extracts inclusive/exclusive time and visits.
    Properly handles both single-process and multi-process (merged MPI) CUBE archives
    using cube_dump -s csv2.
    Returns a list of node-trees, one for each rank/thread found.
    """
    if not cubex_path.exists():
        raise FileNotFoundError(f"CUBE profile not found: {cubex_path}")
    
    dump_bin = find_cube_dump()
    
    try:
        tree_out = subprocess.check_output([dump_bin, "-w", "calltree", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
        incl_out = subprocess.check_output([dump_bin, "-m", "time", "-z", "incl", "-s", "csv2", "-c", "all", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
        excl_out = subprocess.check_output([dump_bin, "-m", "time", "-z", "excl", "-s", "csv2", "-c", "all", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
        vis_out = subprocess.check_output([dump_bin, "-m", "visits", "-z", "excl", "-s", "csv2", "-c", "all", str(cubex_path)], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to run cube_dump on {cubex_path}: {e}\n")
        return []

    def parse_csv2_metric(raw: str) -> Dict[int, Dict[int, float]]:
        # Returns {thread_id/rank_id: {cnode_id: value}}
        m: Dict[int, Dict[int, float]] = defaultdict(dict)
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("Cnode ID") or line.startswith("#"):
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    cid = int(parts[0].strip())
                    tid = int(parts[1].strip())
                    val = float(parts[2].strip())
                    m[tid][cid] = val
                except (ValueError, IndexError):
                    pass
        return m

    incl_map = parse_csv2_metric(incl_out)
    excl_map = parse_csv2_metric(excl_out)
    vis_map = parse_csv2_metric(vis_out)

    # Determine threads/ranks present
    thread_ids = sorted(incl_map.keys()) if incl_map else [0]

    # Parse call tree skeleton once
    skeleton: List[Tuple[int, str, Optional[int]]] = [] # (cid, clean_name, parent_id)
    parent_stack: List[Tuple[int, int]] = [] # (depth, cid)
    calltree_section = False
    for line in tree_out.splitlines():
        if "--- CALL TREE ---" in line:
            calltree_section = True
            continue
        if not calltree_section or not line.strip():
            continue
        
        idx = line.find('[ ( id=')
        if idx != -1:
            prefix_and_name = line[:idx].rstrip()
            rest = line[idx + len('[ ( id='):]
            try:
                cid = int(rest.split(',')[0].split(')')[0].strip())
            except (ValueError, IndexError):
                continue
            
            clean_name = prefix_and_name.lstrip(' |-\t')
            depth = len(prefix_and_name) - len(clean_name)
            
            while parent_stack and parent_stack[-1][0] >= depth:
                parent_stack.pop()
            parent_id = parent_stack[-1][1] if parent_stack else None
            parent_stack.append((depth, cid))
            skeleton.append((cid, clean_name, parent_id))

    rank_trees: List[Dict[int, CallTreeNode]] = []

    for tid in thread_ids:
        nodes: Dict[int, CallTreeNode] = {}
        t_incl = incl_map.get(tid, {})
        t_excl = excl_map.get(tid, {})
        t_vis = vis_map.get(tid, {})

        for cid, clean_name, parent_id in skeleton:
            node = CallTreeNode(cid, clean_name, parent_id)
            node.incl_time = t_incl.get(cid, 0.0)
            node.excl_time = t_excl.get(cid, 0.0)
            node.visits = int(t_vis.get(cid, 0))
            nodes[cid] = node
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].children_ids.append(cid)

        # Determine normalization base (visits of steady ML step)
        steady_visits = explicit_steady_steps or 0
        if steady_visits <= 0:
            for n in nodes.values():
                if n.name in ("solver_step_ml_steady", "solver_ml_provider_call") and n.visits > 0:
                    steady_visits = n.visits
                    break
        if steady_visits <= 0:
            # DL-side client profile detection
            for n in nodes.values():
                if n.name in ("dl_torch_forward", "py_torch_forward", "py_inference", "dl_recv", "py_recv") and n.visits > 0:
                    steady_visits = n.visits
                    break
        if steady_visits <= 0:
            steady_visits = 1

        def get_ancestor_names(cid: int) -> Set[str]:
            names = set()
            curr: Optional[int] = cid
            while curr is not None and curr in nodes:
                names.add(nodes[curr].name)
                curr = nodes[curr].parent_id
            return names

        # Compute per-step metrics
        for n in nodes.values():
            ancestors = get_ancestor_names(n.id)
            
            # Lifecycle classification
            if "solver_setup" in ancestors or "solver_teardown" in ancestors or "solver_step_ml_warmup" in ancestors or n.name in ("solver_setup", "solver_teardown", "solver_step_ml_warmup", "terrain_solver", "solver_main_loop"):
                n.is_lifecycle = True
                n.step_incl_time = n.incl_time
                n.step_excl_time = n.excl_time
                n.step_self_time = n.excl_time
            else:
                n.is_lifecycle = False
                n.step_incl_time = n.incl_time / steady_visits
                n.step_excl_time = n.excl_time / steady_visits
                n.step_self_time = n.excl_time / steady_visits

        rank_trees.append(nodes)

    return rank_trees


# =============================================================================
# Multi-Rank Profile Aggregator
# =============================================================================

class AggregatedCallTree:
    def __init__(self):
        # Callpath key: tuple of region names from root to node
        self.paths: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self.total_ranks: int = 0
        self.rank_trees: List[Dict[int, CallTreeNode]] = []

    def add_tree(self, nodes: Dict[int, CallTreeNode]):
        if not nodes:
            return
        self.total_ranks += 1
        self.rank_trees.append(nodes)

        def get_path(cid: int) -> Tuple[str, ...]:
            p = []
            curr: Optional[int] = cid
            while curr is not None and curr in nodes:
                p.append(nodes[curr].name)
                curr = nodes[curr].parent_id
            return tuple(reversed(p))

        rank_idx = self.total_ranks - 1
        for cid, n in nodes.items():
            path = get_path(cid)
            if path not in self.paths:
                self.paths[path] = {
                    "name": n.name,
                    "parent_path": path[:-1] if len(path) > 1 else None,
                    "children_paths": set(),
                    "is_lifecycle": n.is_lifecycle,
                    "per_rank_step_incl": {},
                    "per_rank_step_excl": {},
                    "per_rank_step_self": {},
                    "per_rank_visits": {},
                }
            self.paths[path]["per_rank_step_incl"][rank_idx] = n.step_incl_time
            self.paths[path]["per_rank_step_excl"][rank_idx] = n.step_excl_time
            self.paths[path]["per_rank_step_self"][rank_idx] = n.step_self_time
            self.paths[path]["per_rank_visits"][rank_idx] = n.visits

            if len(path) > 1:
                parent_path = path[:-1]
                if parent_path in self.paths:
                    self.paths[parent_path]["children_paths"].add(path)

    def compute_summary(self, rank_agg: str = "mean") -> Dict[Tuple[str, ...], Dict[str, Any]]:
        summary: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        num_ranks = max(1, self.total_ranks)

        for path, d in self.paths.items():
            # Pad missing ranks with 0.0 for consistent tree algebra across MPI communicator
            incls_all = [d["per_rank_step_incl"].get(r, 0.0) for r in range(num_ranks)]
            excls_all = [d["per_rank_step_excl"].get(r, 0.0) for r in range(num_ranks)]
            selfs_all = [d["per_rank_step_self"].get(r, 0.0) for r in range(num_ranks)]
            vis_all = [d["per_rank_visits"].get(r, 0) for r in range(num_ranks)]

            active_incls = [v for v in d["per_rank_step_incl"].values() if v > 1e-9]
            active_ranks = len(active_incls)

            rank0_incl = d["per_rank_step_incl"].get(0, 0.0)
            rank0_self = d["per_rank_step_self"].get(0, 0.0)

            # Communicator-wide mean (sum across ranks / total_ranks)
            comm_mean_incl = float(np.mean(incls_all))
            comm_mean_excl = float(np.mean(excls_all))
            comm_mean_self = float(np.mean(selfs_all))

            # Active-only mean
            active_mean_incl = float(np.mean(active_incls)) if active_incls else 0.0

            max_incl = float(np.max(incls_all))
            min_incl = float(np.min(incls_all)) if active_ranks == num_ranks else 0.0
            std_incl = float(np.std(incls_all)) if num_ranks > 1 else 0.0

            summary[path] = {
                "name": d["name"],
                "parent_path": d["parent_path"],
                "children_paths": sorted(list(d["children_paths"])),
                "is_lifecycle": d["is_lifecycle"],
                "incl_mean": comm_mean_incl,
                "incl_active_mean": active_mean_incl,
                "incl_max": max_incl,
                "incl_min": min_incl,
                "incl_std": std_incl,
                "incl_rank0": rank0_incl,
                "self_mean": comm_mean_self,
                "self_rank0": rank0_self,
                "excl_mean": comm_mean_excl,
                "visits_mean": float(np.mean(vis_all)),
                "active_ranks": active_ranks,
                "total_ranks": num_ranks,
            }
        return summary


# =============================================================================
# True Icicle Plot Renderer
# =============================================================================

def render_icicle_plot(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]],
                       root_path: Tuple[str, ...],
                       output_path: Path,
                       title: str = "CMI Coupling Phase Breakdown",
                       use_rank0: bool = False):
    """
    Renders a true Icicle plot where every child fits strictly within its parent rectangle.
    """
    if root_path not in tree_summary:
        candidates = [p for p in tree_summary if p[-1] in ("solver_ml_provider_call", "app_provider_inference", "phydll_dl_client", "py_inference")]
        if candidates:
            root_path = candidates[0]
        else:
            root_path = list(tree_summary.keys())[0]

    val_key = "incl_rank0" if use_rank0 else "incl_mean"
    self_key = "self_rank0" if use_rank0 else "self_mean"

    root_info = tree_summary[root_path]
    root_val = root_info[val_key]
    if root_val <= 0:
        root_val = 1e-6

    fig, ax = plt.subplots(figsize=(13, 6.5))

    row_height = 0.85
    y_gap = 0.18
    max_depth = 5

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
            linewidth=1.0, edgecolor="#111111", facecolor=color, alpha=0.92
        )
        ax.add_patch(rect)

        val_ms = node_info[val_key] * 1000.0
        pct = (node_info[val_key] / root_val) * 100.0
        
        clean_name = name.replace("smartsim_", "").replace("phydll_", "").replace("aix_", "").replace("solver_", "").replace("app_", "").replace("torchInference::", "")
        if width > 0.12 * root_val:
            lbl = f"{clean_name}\n{val_ms:.2f} ms ({pct:.1f}%)"
            ax.text(x_start + width / 2.0, y_pos + row_height / 2.0, lbl,
                    ha="center", va="center", fontsize=8.5, weight="bold", color="#111111")
        elif width > 0.04 * root_val:
            ax.text(x_start + width / 2.0, y_pos + row_height / 2.0, f"{clean_name}\n{val_ms:.1f}ms",
                    ha="center", va="center", fontsize=7.5, color="#111111")

        children = node_info["children_paths"]
        if children and depth < max_depth:
            child_sum = sum(tree_summary[c][val_key] for c in children if c in tree_summary)
            curr_x = x_start
            for c in children:
                c_val = tree_summary[c][val_key]
                if c_val <= 0:
                    continue
                c_width = (c_val / max(width, child_sum, 1e-9)) * width
                draw_node(c, curr_x, c_width, depth + 1)
                curr_x += c_width

            residual = max(0.0, width - child_sum)
            if residual > 0.015 * root_val:
                res_y = (max_depth - (depth + 1)) * (row_height + y_gap)
                rect_res = patches.Rectangle(
                    (curr_x, res_y), residual, row_height,
                    linewidth=1.0, edgecolor="#555555", facecolor=COLOR_MAP["Self / Overhead"], alpha=0.75, linestyle=":"
                )
                ax.add_patch(rect_res)
                if residual > 0.05 * root_val:
                    ax.text(curr_x + residual / 2.0, res_y + row_height / 2.0,
                            f"Overhead\n{residual*1000:.2f} ms", ha="center", va="center", fontsize=7.5, color="#333333")

    draw_node(root_path, 0.0, root_val, 0)

    ax.set_xlim(-0.01 * root_val, 1.01 * root_val)
    ax.set_ylim(-0.2, (max_depth + 1) * (row_height + y_gap))
    ax.set_xlabel("Duration (seconds per steady ML step)", weight="bold")
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
# Exact Leaf Decomposition Bar Chart
# =============================================================================

def collect_leaf_decomposition(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]],
                                root_path: Tuple[str, ...],
                                val_key: str = "incl_mean") -> List[Tuple[str, float]]:
    """
    Collects a strictly non-overlapping partition of leaf stages under root_path.
    """
    leaves: List[Tuple[str, float]] = []

    def traverse(path: Tuple[str, ...]):
        if path not in tree_summary:
            return
        node = tree_summary[path]
        node_val = node[val_key]
        if node_val <= 0:
            return

        children = [c for c in node["children_paths"] if c in tree_summary and tree_summary[c][val_key] > 0]
        if not children:
            leaves.append((node["name"], node_val * 1000.0))
        else:
            child_sum = sum(tree_summary[c][val_key] for c in children)
            for c in children:
                traverse(c)
            residual = max(0.0, node_val - child_sum)
            if residual * 1000.0 > 0.005:
                leaves.append((f"{node['name']} [Overhead]", residual * 1000.0))

    traverse(root_path)
    return leaves

def render_breakdown_bars(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]],
                          root_path: Tuple[str, ...],
                          output_path: Path,
                          title: str = "CMI Steady Step Leaf Breakdown",
                          use_rank0: bool = False):
    """
    Renders non-overlapping leaf phase breakdown bar charts.
    """
    val_key = "incl_rank0" if use_rank0 else "incl_mean"
    raw_leaves = collect_leaf_decomposition(tree_summary, root_path, val_key=val_key)
    if not raw_leaves:
        return

    # Filter out tiny leaves and collapse minor entries
    significant_leaves = [l for l in raw_leaves if l[1] >= 0.005]
    other_sum = sum(l[1] for l in raw_leaves if l[1] < 0.005)

    if len(significant_leaves) > 20:
        top_leaves = sorted(significant_leaves, key=lambda x: x[1], reverse=True)[:20]
        top_set = set(id(l) for l in top_leaves)
        other_sum += sum(l[1] for l in significant_leaves if id(l) not in top_set)
        leaves = top_leaves
    else:
        leaves = sorted(significant_leaves, key=lambda x: x[1], reverse=True)

    if other_sum > 0.005:
        leaves.append(("Other / Minor", other_sum))

    names = [x[0] for x in leaves]
    times = [x[1] for x in leaves]
    colors = [get_color(n.split()[0]) for n in names]

    fig, ax = plt.subplots(figsize=(11, max(3.5, len(leaves) * 0.45)))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="#222222", height=0.65, alpha=0.92)
    ax.set_yticks(y_pos)
    clean_labels = [p.replace("smartsim_", "").replace("phydll_", "").replace("aix_", "").replace("app_", "").replace("torchInference::", "") for p in names]
    ax.set_yticklabels(clean_labels, fontsize=9.5, weight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Duration per Steady ML Step (ms)", weight="bold")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)

    total_time = sum(times)
    for bar, t in zip(bars, times):
        pct = (t / total_time) * 100.0 if total_time > 0 else 0.0
        ax.text(
            bar.get_width() + (max(times) * 0.015),
            bar.get_y() + bar.get_height() / 2.0,
            f"{t:.3f} ms ({pct:.1f}%)",
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

def is_interesting_region(name: str) -> bool:
    bad_prefixes = ("std::", "__gnu_cxx::", "void", "decltype", "unsigned", "int&", "bool", "char", "unsigned long", "_GLOBAL__", "__static_")
    if any(name.startswith(p) for p in bad_prefixes):
        return False
    if "allocator" in name or "_M_realloc" in name or "_M_construct" in name or "deallocate" in name:
        return False
    return True

def export_summary_tables(tree_summary: Dict[Tuple[str, ...], Dict[str, Any]], out_dir: Path, title: str = "CMI Profile Summary"):
    rows = []
    for path, d in sorted(tree_summary.items(), key=lambda x: x[1]["incl_mean"], reverse=True):
        rows.append({
            "Callpath": " -> ".join(path),
            "Region": d["name"],
            "Is_Lifecycle": d["is_lifecycle"],
            "Rank_Mean (ms)": d["incl_mean"] * 1000.0,
            "Controller_Rank0 (ms)": d["incl_rank0"] * 1000.0,
            "Max_Rank (ms)": d["incl_max"] * 1000.0,
            "Min_Rank (ms)": d["incl_min"] * 1000.0,
            "Std_Rank (ms)": d["incl_std"] * 1000.0,
            "Self (ms)": d["self_mean"] * 1000.0,
            "Visits/Step": d["visits_mean"],
            "Active_Ranks": d["active_ranks"],
            "Total_Ranks": d["total_ranks"]
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "cmi_phase_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved phase summary CSV to: {csv_path}")

    steady_df = df[~df["Is_Lifecycle"]].copy()
    lifecycle_df = df[df["Is_Lifecycle"]].copy()

    md_path = out_dir / "cmi_profile_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("## Steady-State Coupling Metrics (Normalized Per ML Step)\n\n")
        f.write("| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Step | Ranks |\n")
        f.write("|:---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in steady_df.iterrows():
            if is_interesting_region(r["Region"]):
                f.write(f"| `{r['Region']}` | {r['Rank_Mean (ms)']:.3f} | {r['Controller_Rank0 (ms)']:.3f} | {r['Max_Rank (ms)']:.3f} | {r['Self (ms)']:.3f} | {r['Visits/Step']:.1f} | {int(r['Active_Ranks'])}/{int(r['Total_Ranks'])} |\n")
        
        f.write("\n## Lifecycle Phase Totals (One-Off Cumulative)\n\n")
        f.write("| Phase | Duration (ms) | Ranks |\n")
        f.write("|:---|---:|---:|\n")
        for _, r in lifecycle_df.iterrows():
            if r["Region"] in ("terrain_solver", "solver_setup", "solver_step_ml_warmup", "solver_teardown"):
                f.write(f"| `{r['Region']}` | {r['Rank_Mean (ms)']:.3f} | {int(r['Active_Ranks'])}/{int(r['Total_Ranks'])} |\n")

    print(f"[+] Saved summary Markdown to: {md_path}")


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CMI Score-P CUBE and AIx Timeline Analysis Tool")
    parser.add_argument("--cubex", type=str, default=None, help="Path or glob pattern for profile.cubex")
    parser.add_argument("--dl-cubex", type=str, default=None, help="Optional DL-side profile.cubex for MPMD PhyDLL")
    parser.add_argument("--p2p-timeline-dir", type=Path, default=None, help="Path to directory containing aix_p2p_timeline_rank_*.csv")
    parser.add_argument("--steady-steps", type=int, default=None, help="Explicit number of steady ML steps for normalization")
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
        if any(c in p_str for c in "*?[]"):
            matched = [Path(p) for p in sorted(glob.glob(p_str))]
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
            trees = parse_cubex_file(f, explicit_steady_steps=args.steady_steps)
            for nodes in trees:
                agg.add_tree(nodes)

        tree_summary = agg.compute_summary(rank_agg=args.rank_agg)
        if tree_summary:
            title = args.title or f"CMI Coupling Profile ({cubex_files[0].parent.parent.name if len(cubex_files)>1 else cubex_files[0].parent.name})"
            export_summary_tables(tree_summary, args.output_dir, title=title)

            if not args.no_plots:
                steady_candidates = [p for p in tree_summary if p[-1] in ("solver_ml_provider_call", "app_provider_inference") and not tree_summary[p]["is_lifecycle"]]
                root_path = steady_candidates[0] if steady_candidates else list(tree_summary.keys())[0]
                
                has_controller_diff = any(tree_summary[p]["active_ranks"] < tree_summary[p]["total_ranks"] for p in tree_summary)
                if has_controller_diff:
                    # Render Controller/Rank 0 perspective (where forward execution happens)
                    render_icicle_plot(tree_summary, root_path, args.output_dir / "cmi_icicle_plot_controller_rank0.png", title=f"{title} - Controller/Rank 0 Breakdown", use_rank0=True)
                    render_breakdown_bars(tree_summary, root_path, args.output_dir / "cmi_stage_breakdown_controller_rank0.png", title=f"{title} - Controller/Rank 0 Leaf Breakdown", use_rank0=True)
                
                # Render Communicator Mean
                render_icicle_plot(tree_summary, root_path, args.output_dir / "cmi_icicle_plot.png", title=f"{title} - Hierarchical Breakdown (Comm Mean)", use_rank0=False)
                render_breakdown_bars(tree_summary, root_path, args.output_dir / "cmi_stage_breakdown.png", title=f"{title} - Leaf Breakdown (Comm Mean)", use_rank0=False)

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
            dl_trees = parse_cubex_file(f, explicit_steady_steps=args.steady_steps)
            for dl_nodes in dl_trees:
                dl_agg.add_tree(dl_nodes)
        dl_summary = dl_agg.compute_summary(rank_agg=args.rank_agg)
        if dl_summary and not args.no_plots:
            dl_roots = [p for p in dl_summary if p[-1] in ("phydll_dl_client", "py_inference", "user:py_inference")]
            dl_root = dl_roots[0] if dl_roots else list(dl_summary.keys())[0]
            render_icicle_plot(dl_summary, dl_root, args.output_dir / "phydll_dl_icicle_plot.png", title="PhyDLL DL-Side Breakdown")
            render_breakdown_bars(dl_summary, dl_root, args.output_dir / "phydll_dl_stage_breakdown.png", title="PhyDLL DL-Side Leaf Breakdown")

    # 3. Process AIx P2P Timeline CSVs
    if args.p2p_timeline_dir and args.p2p_timeline_dir.exists():
        print(f"[*] Analyzing AIx P2P Timeline in: {args.p2p_timeline_dir}")
        p2p_df = parse_aix_p2p_timeline(args.p2p_timeline_dir)
        if not p2p_df.empty and not args.no_plots:
            plot_aix_pipeline_gantt(p2p_df, args.output_dir / "aix_pipeline_gantt.png")

    print("[✓] Analysis complete.")

if __name__ == "__main__":
    main()
