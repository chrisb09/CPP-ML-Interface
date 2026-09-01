#!/usr/bin/env python3
"""
compare_cmi_frameworks.py
=========================
Comprehensive comparative analysis and visualization tool for CMI frameworks:
- SmartSim (c=0, c=1, c=2, c=3)
- AIxelerator (Collective, Pipelined)
- PhyDLL (C++, Python)

Generates standardized stacked bar charts, deep ML transaction breakdowns,
critical-path imbalance analyses, and executive dashboard figures.

Usage:
  python3 compare_cmi_frameworks.py --results-dirs results_gpu_smartsim_c0 results_gpu_smartsim_c1 ... --output-dir ./results_gpu_comparison
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Global typography and theme
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Canonical color scheme
COLORS = {
    # High-level phases
    "PDE Compute": "#7FBC41",        # Green
    "Input Prep": "#4393C3",         # Blue
    "ML Transaction": "#D6604D",     # Coral / Red
    "Output Finalize": "#74ADD1",    # Light Blue
    "Residual / Other": "#CCCCCC",   # Grey
    
    # Detailed transaction stages
    "Input Transport (Put/Gather/Send)": "#E08214",     # Orange
    "Token / Idle / Wait": "#FDAE61",                   # Yellow-Orange
    "GPU Compute (Forward)": "#A50026",                 # Crimson
    "H2D / D2H Copy": "#FEE090",                        # Pale Yellow
    "Output Transport (Unpack/Scatter/Recv)": "#542788",# Purple
    "Post-Scatter / Cleanup": "#B2ABD2",                # Light Purple
}


def parse_summary_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV at: {csv_path}")
    return pd.read_csv(csv_path)


def extract_framework_metrics(df: pd.DataFrame, framework_name: str) -> Dict[str, Any]:
    """
    Extracts standardized high-level and detailed phase durations (ms) from cmi_phase_summary.csv.
    """
    def get_val(region: str, col: str = "Rank_Mean (ms)", fallback: float = 0.0) -> float:
        # Prefer steady-state (non-lifecycle) rows to avoid warmup contamination
        matches = df[(df["Region"] == region) & (~df["Is_Lifecycle"])]
        if not matches.empty:
            return float(matches[col].iloc[0])
        matches_all = df[df["Region"] == region]
        if not matches_all.empty:
            return float(matches_all[col].iloc[0])
        return fallback

    steady_step_mean = get_val("solver_step_ml_steady", "Rank_Mean (ms)")
    steady_step_r0 = get_val("solver_step_ml_steady", "Controller_Rank0 (ms)")
    steady_step_max = get_val("solver_step_ml_steady", "Max_Rank (ms)")

    compute_mean = get_val("solver_step_compute", "Rank_Mean (ms)")
    compute_r0 = get_val("solver_step_compute", "Controller_Rank0 (ms)")

    prep_mean = get_val("app_prepare_input", "Rank_Mean (ms)")
    prep_r0 = get_val("app_prepare_input", "Controller_Rank0 (ms)")

    fin_mean = get_val("app_finalize_output", "Rank_Mean (ms)")
    fin_r0 = get_val("app_finalize_output", "Controller_Rank0 (ms)")

    provider_inf_mean = get_val("app_provider_inference", "Rank_Mean (ms)")
    provider_inf_r0 = get_val("app_provider_inference", "Controller_Rank0 (ms)")
    provider_inf_max = get_val("app_provider_inference", "Max_Rank (ms)")

    # Detailed Sub-stages (Framework Specific Extraction)
    # 1. SmartSim
    put_tensor_mean = get_val("smartsim_put_tensor", "Rank_Mean (ms)")
    token_wait_mean = get_val("smartsim_token_wait", "Rank_Mean (ms)")
    run_model_mean = get_val("smartsim_run_model", "Rank_Mean (ms)")
    unpack_tensor_mean = get_val("smartsim_unpack_tensor", "Rank_Mean (ms)")

    put_tensor_r0 = get_val("smartsim_put_tensor", "Controller_Rank0 (ms)")
    token_wait_r0 = get_val("smartsim_token_wait", "Controller_Rank0 (ms)")
    run_model_r0 = get_val("smartsim_run_model", "Controller_Rank0 (ms)")
    unpack_tensor_r0 = get_val("smartsim_unpack_tensor", "Controller_Rank0 (ms)")

    # 2. AIx Collective
    gather_mean = get_val("gatherInputData", "Rank_Mean (ms)", fallback=get_val("MPI_Gatherv", "Rank_Mean (ms)"))
    gather_r0 = get_val("gatherInputData", "Controller_Rank0 (ms)", fallback=get_val("MPI_Gatherv", "Controller_Rank0 (ms)"))

    h2d_mean = get_val("h2d_copy", "Rank_Mean (ms)")
    h2d_r0 = get_val("h2d_copy", "Controller_Rank0 (ms)")

    forward_mean = get_val("torchInference::forward", "Rank_Mean (ms)")
    forward_r0 = get_val("torchInference::forward", "Controller_Rank0 (ms)")

    d2h_mean = get_val("d2h_copy", "Rank_Mean (ms)")
    d2h_r0 = get_val("d2h_copy", "Controller_Rank0 (ms)")

    scatter_mean = get_val("scatterOutputData", "Rank_Mean (ms)", fallback=get_val("MPI_Scatterv", "Rank_Mean (ms)"))
    scatter_r0 = get_val("scatterOutputData", "Controller_Rank0 (ms)", fallback=get_val("MPI_Scatterv", "Controller_Rank0 (ms)"))

    post_scatter_mean = get_val("aix_collective_post_scatter", "Rank_Mean (ms)")
    post_scatter_r0 = get_val("aix_collective_post_scatter", "Controller_Rank0 (ms)")

    # 3. AIx Pipelined
    p2p_test_mean = get_val("MPI_Testsome", "Rank_Mean (ms)", fallback=get_val("p2p_pipelined_exchange", "Rank_Mean (ms)"))
    p2p_test_r0 = get_val("MPI_Testsome", "Controller_Rank0 (ms)", fallback=get_val("p2p_pipelined_exchange", "Controller_Rank0 (ms)"))

    # 4. PhyDLL
    phydll_prepack_mean = get_val("phydll_prepack", "Rank_Mean (ms)")
    phydll_prepack_r0 = get_val("phydll_prepack", "Controller_Rank0 (ms)")
    phydll_send_mean = get_val("phydll_send", "Rank_Mean (ms)")
    phydll_send_r0 = get_val("phydll_send", "Controller_Rank0 (ms)")
    phydll_recv_mean = get_val("phydll_recv", "Rank_Mean (ms)")
    phydll_recv_r0 = get_val("phydll_recv", "Controller_Rank0 (ms)")
    phydll_unpack_mean = get_val("phydll_unpack", "Rank_Mean (ms)")
    phydll_unpack_r0 = get_val("phydll_unpack", "Controller_Rank0 (ms)")

    # Consolidate standard transaction breakdown
    # Mode mapping
    tx_transport_in_mean = 0.0
    tx_wait_mean = 0.0
    tx_gpu_compute_mean = 0.0
    tx_gpu_copy_mean = 0.0
    tx_transport_out_mean = 0.0
    tx_post_mean = 0.0

    tx_transport_in_r0 = 0.0
    tx_wait_r0 = 0.0
    tx_gpu_compute_r0 = 0.0
    tx_gpu_copy_r0 = 0.0
    tx_transport_out_r0 = 0.0
    tx_post_r0 = 0.0

    if "SmartSim" in framework_name:
        tx_transport_in_mean = put_tensor_mean
        tx_wait_mean = token_wait_mean
        tx_gpu_compute_mean = run_model_mean
        tx_transport_out_mean = unpack_tensor_mean

        tx_transport_in_r0 = put_tensor_r0
        tx_wait_r0 = token_wait_r0
        tx_gpu_compute_r0 = run_model_r0
        tx_transport_out_r0 = unpack_tensor_r0

    elif "AIx Collective" in framework_name:
        tx_transport_in_mean = gather_mean
        tx_gpu_copy_mean = h2d_mean + d2h_mean
        tx_gpu_compute_mean = forward_mean
        tx_transport_out_mean = scatter_mean
        tx_post_mean = post_scatter_mean

        tx_transport_in_r0 = gather_r0
        tx_gpu_copy_r0 = h2d_r0 + d2h_r0
        tx_gpu_compute_r0 = forward_r0
        tx_transport_out_r0 = scatter_r0
        tx_post_r0 = post_scatter_r0

    elif "AIx Pipelined" in framework_name:
        tx_transport_in_mean = p2p_test_mean
        tx_gpu_compute_mean = max(0.0, provider_inf_mean - p2p_test_mean)

        tx_transport_in_r0 = p2p_test_r0
        tx_gpu_compute_r0 = max(0.0, provider_inf_r0 - p2p_test_r0)

    elif "PhyDLL" in framework_name:
        # Prepack is added into input prep
        prep_mean += phydll_prepack_mean
        prep_r0 += phydll_prepack_r0
        fin_mean += phydll_unpack_mean
        fin_r0 += phydll_unpack_r0

        tx_transport_in_mean = phydll_send_mean
        tx_transport_out_mean = phydll_recv_mean
        tx_transport_in_r0 = phydll_send_r0
        tx_transport_out_r0 = phydll_recv_r0

    return {
        "framework": framework_name,
        "step_mean": steady_step_mean,
        "step_r0": steady_step_r0,
        "step_max": steady_step_max,
        
        "compute_mean": compute_mean,
        "compute_r0": compute_r0,
        
        "prep_mean": prep_mean,
        "prep_r0": prep_r0,
        
        "inf_mean": provider_inf_mean,
        "inf_r0": provider_inf_r0,
        "inf_max": provider_inf_max,
        
        "fin_mean": fin_mean,
        "fin_r0": fin_r0,
        
        # Detailed transaction (Mean)
        "tx_transport_in_mean": tx_transport_in_mean,
        "tx_wait_mean": tx_wait_mean,
        "tx_gpu_compute_mean": tx_gpu_compute_mean,
        "tx_gpu_copy_mean": tx_gpu_copy_mean,
        "tx_transport_out_mean": tx_transport_out_mean,
        "tx_post_mean": tx_post_mean,
        
        # Detailed transaction (Rank 0)
        "tx_transport_in_r0": tx_transport_in_r0,
        "tx_wait_r0": tx_wait_r0,
        "tx_gpu_compute_r0": tx_gpu_compute_r0,
        "tx_gpu_copy_r0": tx_gpu_copy_r0,
        "tx_transport_out_r0": tx_transport_out_r0,
        "tx_post_r0": tx_post_r0,
    }


# =============================================================================
# Plot 1: Standardized 4-Phase Stacked Bar Chart
# =============================================================================

def plot_warm_step_stacked_bars(data: List[Dict[str, Any]], output_path: Path):
    """
    Renders a 2-panel comparative stacked bar chart:
    - Left Panel: Communicator Mean (MPI All-Rank Average)
    - Right Panel: Controller / Rank 0 (Critical Path)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    
    frameworks = [d["framework"] for d in data]
    y_pos = np.arange(len(frameworks))
    
    # Baseline for speedup calculation (SmartSim c=0)
    base_mean = next((d["step_mean"] for d in data if "c=0" in d["framework"]), data[0]["step_mean"])
    base_r0 = next((d["step_r0"] for d in data if "c=0" in d["framework"]), data[0]["step_r0"])
    
    def render_panel(ax, metric_key_suffix: str, base_val: float, panel_title: str):
        prep = np.array([d[f"prep_{metric_key_suffix}"] for d in data])
        inf = np.array([d[f"inf_{metric_key_suffix}"] for d in data])
        fin = np.array([d[f"fin_{metric_key_suffix}"] for d in data])
        comp = np.array([d[f"compute_{metric_key_suffix}"] for d in data])
        totals = np.array([d[f"step_{metric_key_suffix}"] for d in data])
        
        # Stack horizontal bars
        p1 = ax.barh(y_pos, prep, color=COLORS["Input Prep"], edgecolor="#222222", height=0.62, label="Input Preparation")
        p2 = ax.barh(y_pos, inf, left=prep, color=COLORS["ML Transaction"], edgecolor="#222222", height=0.62, label="ML Transaction (Round-Trip)")
        p3 = ax.barh(y_pos, fin, left=prep+inf, color=COLORS["Output Finalize"], edgecolor="#222222", height=0.62, label="Output Finalization")
        p4 = ax.barh(y_pos, comp, left=prep+inf+fin, color=COLORS["PDE Compute"], edgecolor="#222222", height=0.62, label="PDE Solver Compute")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(frameworks, weight="bold", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Duration per Steady ML Step (ms)", weight="bold")
        ax.set_title(panel_title, weight="bold", fontsize=12, pad=12)
        
        max_total = max(totals)
        for i, (tot, p_inf) in enumerate(zip(totals, inf)):
            speedup = base_val / tot if tot > 0 else 1.0
            sp_str = f" ({speedup:.2f}x)" if "c=0" not in frameworks[i] else " (1.00x)"
            ax.text(
                tot + (max_total * 0.015),
                i,
                f"{tot:.2f} ms{sp_str}",
                va="center", ha="left", fontsize=9, weight="bold", color="#111111"
            )
        ax.set_xlim(0, max_total * 1.25)

    render_panel(ax1, "mean", base_mean, "A. Communicator Mean (All Ranks)")
    render_panel(ax2, "r0", base_r0, "B. Controller / Rank 0 (Critical Path)")
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved warm step comparative stacked bar chart to: {output_path}")


# =============================================================================
# Plot 2: Detailed ML Transaction Breakdown Stacked Bar Chart
# =============================================================================

def plot_ml_transaction_breakdown_bars(data: List[Dict[str, Any]], output_path: Path):
    """
    Renders detailed breakdown of provider_inference across frameworks:
    - Input Transport (Put / Gather / Send)
    - Token / Barrier / Idle Wait
    - GPU Kernel (Forward)
    - Host-to-Device / Device-to-Host Copies
    - Output Transport (Unpack / Scatter / Recv)
    - Post-Scatter / Cleanup
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    
    frameworks = [d["framework"] for d in data]
    y_pos = np.arange(len(frameworks))
    
    def render_tx_panel(ax, suffix: str, panel_title: str):
        t_in = np.array([d[f"tx_transport_in_{suffix}"] for d in data])
        wait = np.array([d[f"tx_wait_{suffix}"] for d in data])
        h2d_d2h = np.array([d[f"tx_gpu_copy_{suffix}"] for d in data])
        fwd = np.array([d[f"tx_gpu_compute_{suffix}"] for d in data])
        t_out = np.array([d[f"tx_transport_out_{suffix}"] for d in data])
        post = np.array([d[f"tx_post_{suffix}"] for d in data])
        totals = np.array([d[f"inf_{suffix}"] for d in data])
        
        l1 = t_in
        l2 = l1 + wait
        l3 = l2 + h2d_d2h
        l4 = l3 + fwd
        l5 = l4 + t_out
        
        b1 = ax.barh(y_pos, t_in, color=COLORS["Input Transport (Put/Gather/Send)"], edgecolor="#222222", height=0.62, label="Input Transport (Put/Gather/Send)")
        b2 = ax.barh(y_pos, wait, left=l1, color=COLORS["Token / Idle / Wait"], edgecolor="#222222", height=0.62, label="Token / Idle / Barrier Wait")
        b3 = ax.barh(y_pos, h2d_d2h, left=l2, color=COLORS["H2D / D2H Copy"], edgecolor="#222222", height=0.62, label="H2D / D2H Memory Copies")
        b4 = ax.barh(y_pos, fwd, left=l3, color=COLORS["GPU Compute (Forward)"], edgecolor="#222222", height=0.62, label="GPU Compute (Forward / Model Execution)")
        b5 = ax.barh(y_pos, t_out, left=l4, color=COLORS["Output Transport (Unpack/Scatter/Recv)"], edgecolor="#222222", height=0.62, label="Output Transport (Unpack/Scatter/Recv)")
        b6 = ax.barh(y_pos, post, left=l5, color=COLORS["Post-Scatter / Cleanup"], edgecolor="#222222", height=0.62, label="Post-Scatter / Cleanup")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(frameworks, weight="bold", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Duration per Steady ML Transaction (ms)", weight="bold")
        ax.set_title(panel_title, weight="bold", fontsize=12, pad=12)
        
        max_tot = max(totals)
        for i, tot in enumerate(totals):
            ax.text(
                tot + (max_tot * 0.015),
                i,
                f"{tot:.2f} ms",
                va="center", ha="left", fontsize=9, weight="bold", color="#111111"
            )
        ax.set_xlim(0, max_tot * 1.22)

    render_tx_panel(ax1, "mean", "A. ML Transaction Breakdown (Communicator Mean)")
    render_tx_panel(ax2, "r0", "B. ML Transaction Breakdown (Controller / Rank 0)")
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved ML transaction comparative breakdown chart to: {output_path}")


# =============================================================================
# Plot 3: Load Imbalance Comparison (Rank 0 vs. Mean vs. Max Rank)
# =============================================================================

def plot_imbalance_comparison_bars(data: List[Dict[str, Any]], output_path: Path):
    """
    Renders grouped bar chart showing rank disparity (Controller Rank 0 vs. Comm Mean vs. Max Rank).
    """
    fig, ax = plt.subplots(figsize=(13, 6.5))
    
    frameworks = [d["framework"] for d in data]
    y_pos = np.arange(len(frameworks))
    bar_height = 0.26
    
    r0_vals = [d["step_r0"] for d in data]
    mean_vals = [d["step_mean"] for d in data]
    max_vals = [d["step_max"] for d in data]
    
    b1 = ax.barh(y_pos - bar_height, r0_vals, height=bar_height, color="#2B5C8F", edgecolor="#111111", label="Controller / Rank 0")
    b2 = ax.barh(y_pos, mean_vals, height=bar_height, color="#4393C3", edgecolor="#111111", label="Communicator Mean")
    b3 = ax.barh(y_pos + bar_height, max_vals, height=bar_height, color="#D6604D", edgecolor="#111111", label="Max Bottleneck Rank")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frameworks, weight="bold", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Duration per Steady ML Step (ms)", weight="bold")
    ax.set_title("MPI Rank Load Disparity: Controller (Rank 0) vs. Mean vs. Bottleneck Rank", weight="bold", fontsize=12, pad=14)
    
    max_val = max(max_vals)
    for i in range(len(frameworks)):
        # Annotate Rank 0
        ax.text(r0_vals[i] + max_val * 0.01, i - bar_height, f"{r0_vals[i]:.2f}ms", va="center", ha="left", fontsize=8, color="#2B5C8F", weight="bold")
        # Annotate Mean
        ax.text(mean_vals[i] + max_val * 0.01, i, f"{mean_vals[i]:.2f}ms", va="center", ha="left", fontsize=8, color="#4393C3", weight="bold")
        # Annotate Max
        imbalance_pct = ((max_vals[i] - r0_vals[i]) / r0_vals[i] * 100.0) if r0_vals[i] > 0 else 0.0
        imb_str = f" (+{imbalance_pct:.0f}%)" if abs(imbalance_pct) > 2.0 else ""
        ax.text(max_vals[i] + max_val * 0.01, i + bar_height, f"{max_vals[i]:.2f}ms{imb_str}", va="center", ha="left", fontsize=8, color="#D6604D", weight="bold")
        
    ax.set_xlim(0, max_val * 1.25)
    ax.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved load imbalance comparative chart to: {output_path}")


# =============================================================================
# Plot 4: Executive Dashboard (4-Panel Multi-Metric Comparison)
# =============================================================================

def plot_executive_dashboard(data: List[Dict[str, Any]], output_path: Path):
    """
    Renders publication-ready 4-panel executive dashboard:
    - Panel A: Speedup & Relative Throughput vs. SmartSim c=0
    - Panel B: High-Level Phase Breakdown (Prepare, Inference, Finalize, Compute)
    - Panel C: Deep ML Transaction Breakdown
    - Panel D: Performance Matrix & Key Insights Table
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    
    frameworks = [d["framework"] for d in data]
    y_pos = np.arange(len(frameworks))
    
    base_mean = next((d["step_mean"] for d in data if "c=0" in d["framework"]), data[0]["step_mean"])
    
    # -------------------------------------------------------------------------
    # Panel A: Speedup vs Baseline
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    speedups = [base_mean / d["step_mean"] for d in data]
    colors_speedup = ["#D73027" if s < 1.0 else "#2B5C8F" if s == 1.0 else "#1B7837" for s in speedups]
    bars_sp = ax1.barh(y_pos, speedups, color=colors_speedup, edgecolor="#111111", height=0.6)
    ax1.axvline(1.0, color="#666666", linestyle="--", linewidth=1.2, label="SmartSim c=0 Baseline (1.0x)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(frameworks, weight="bold", fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Speedup Factor Relative to SmartSim c=0 (Higher is Better)", weight="bold")
    ax1.set_title("A. Framework Speedup & Efficiency", weight="bold", fontsize=12, pad=10)
    for i, s in enumerate(speedups):
        ax1.text(s + 0.05, i, f"{s:.2f}x ({data[i]['step_mean']:.2f} ms)", va="center", ha="left", fontsize=9, weight="bold")
    ax1.set_xlim(0, max(speedups) * 1.25)
    ax1.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel B: High-Level Phase Breakdown
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    prep = np.array([d["prep_mean"] for d in data])
    inf = np.array([d["inf_mean"] for d in data])
    fin = np.array([d["fin_mean"] for d in data])
    comp = np.array([d["compute_mean"] for d in data])
    totals = np.array([d["step_mean"] for d in data])
    
    ax2.barh(y_pos, prep, color=COLORS["Input Prep"], edgecolor="#222222", height=0.6, label="Input Preparation")
    ax2.barh(y_pos, inf, left=prep, color=COLORS["ML Transaction"], edgecolor="#222222", height=0.6, label="ML Transaction")
    ax2.barh(y_pos, fin, left=prep+inf, color=COLORS["Output Finalize"], edgecolor="#222222", height=0.6, label="Output Finalization")
    ax2.barh(y_pos, comp, left=prep+inf+fin, color=COLORS["PDE Compute"], edgecolor="#222222", height=0.6, label="PDE Compute")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(frameworks, weight="bold", fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Duration per Steady Step (ms)", weight="bold")
    ax2.set_title("B. High-Level Step Breakdown (Comm Mean)", weight="bold", fontsize=12, pad=10)
    for i, tot in enumerate(totals):
        ax2.text(tot + max(totals)*0.015, i, f"{tot:.2f} ms", va="center", ha="left", fontsize=9, weight="bold")
    ax2.set_xlim(0, max(totals) * 1.25)
    ax2.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel C: ML Transaction Breakdown
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    t_in = np.array([d["tx_transport_in_r0"] for d in data])
    wait = np.array([d["tx_wait_r0"] for d in data])
    h2d_d2h = np.array([d["tx_gpu_copy_r0"] for d in data])
    fwd = np.array([d["tx_gpu_compute_r0"] for d in data])
    t_out = np.array([d["tx_transport_out_r0"] for d in data])
    post = np.array([d["tx_post_r0"] for d in data])
    inf_r0 = np.array([d["inf_r0"] for d in data])
    
    l1 = t_in
    l2 = l1 + wait
    l3 = l2 + h2d_d2h
    l4 = l3 + fwd
    l5 = l4 + t_out
    
    ax3.barh(y_pos, t_in, color=COLORS["Input Transport (Put/Gather/Send)"], edgecolor="#222222", height=0.6, label="Input Transport")
    ax3.barh(y_pos, wait, left=l1, color=COLORS["Token / Idle / Wait"], edgecolor="#222222", height=0.6, label="Token / Idle Wait")
    ax3.barh(y_pos, h2d_d2h, left=l2, color=COLORS["H2D / D2H Copy"], edgecolor="#222222", height=0.6, label="H2D/D2H Copy")
    ax3.barh(y_pos, fwd, left=l3, color=COLORS["GPU Compute (Forward)"], edgecolor="#222222", height=0.6, label="GPU Forward")
    ax3.barh(y_pos, t_out, left=l4, color=COLORS["Output Transport (Unpack/Scatter/Recv)"], edgecolor="#222222", height=0.6, label="Output Transport")
    ax3.barh(y_pos, post, left=l5, color=COLORS["Post-Scatter / Cleanup"], edgecolor="#222222", height=0.6, label="Post-Scatter")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(frameworks, weight="bold", fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel("Duration per ML Transaction (ms)", weight="bold")
    ax3.set_title("C. ML Transaction Breakdown (Controller / Rank 0)", weight="bold", fontsize=12, pad=10)
    for i, tot in enumerate(inf_r0):
        ax3.text(tot + max(inf_r0)*0.015, i, f"{tot:.2f} ms", va="center", ha="left", fontsize=9, weight="bold")
    ax3.set_xlim(0, max(inf_r0) * 1.25)
    ax3.legend(loc="lower right")

    # -------------------------------------------------------------------------
    # Panel D: Summary Performance Table
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_title("D. Coupling Performance Matrix (480x288, 22 Steps)", weight="bold", fontsize=12, pad=10)
    
    table_data = []
    headers = ["Framework", "Rank 0 (ms)", "Mean (ms)", "Max (ms)", "Speedup", "Throughput"]
    for d in data:
        sp = base_mean / d["step_mean"]
        thp = 1000.0 / d["step_mean"] if d["step_mean"] > 0 else 0.0
        table_data.append([
            d["framework"],
            f"{d['step_r0']:.2f}",
            f"{d['step_mean']:.2f}",
            f"{d['step_max']:.2f}",
            f"{sp:.2f}x",
            f"{thp:.1f} step/s"
        ])
        
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.88]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.auto_set_column_width(col=list(range(len(headers))))
    
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2B5C8F")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.12)
        else:
            cell.set_facecolor("#F8F9FA" if r % 2 == 0 else "#FFFFFF")
            cell.set_height(0.10)
            if c == 0:
                cell.set_text_props(weight="bold", ha="left")
            elif c == 4:
                sp_val = float(table_data[r-1][4].replace("x", ""))
                if sp_val > 1.05:
                    cell.set_text_props(color="#1B7837", weight="bold")
                elif sp_val < 0.95:
                    cell.set_text_props(color="#D73027", weight="bold")

    plt.suptitle("Multi-Framework GPU ML Coupling Performance Benchmark", fontsize=15, weight="bold", y=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved executive dashboard figure to: {output_path}")


# =============================================================================
# Summary Markdown and CSV Export
# =============================================================================

def export_framework_comparison_reports(data: List[Dict[str, Any]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV Export
    df = pd.DataFrame(data)
    csv_path = output_dir / "framework_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved comparison CSV summary to: {csv_path}")
    
    # Markdown Export
    base_mean = next((d["step_mean"] for d in data if "c=0" in d["framework"]), data[0]["step_mean"])
    md_path = output_dir / "framework_comparison_summary.md"
    
    with open(md_path, "w") as f:
        f.write("# ML Coupling Framework Comparison (480x288, 22 Steps, 1 GPU, 4 Solver Ranks)\n\n")
        f.write("## 1. High-Level Step Durations & Speedup\n\n")
        f.write("| Framework | Rank 0 (ms) | Comm Mean (ms) | Max Rank (ms) | Speedup vs c=0 | Throughput (steps/s) |\n")
        f.write("|:---|---:|---:|---:|---:|---:|\n")
        for d in data:
            sp = base_mean / d["step_mean"]
            thp = 1000.0 / d["step_mean"] if d["step_mean"] > 0 else 0.0
            f.write(f"| **{d['framework']}** | {d['step_r0']:.3f} | {d['step_mean']:.3f} | {d['step_max']:.3f} | **{sp:.2f}x** | {thp:.1f} |\n")
            
        f.write("\n## 2. Standardized 4-Phase Step Decomposition (Communicator Mean)\n\n")
        f.write("| Framework | Prepare Input (ms) | ML Transaction (ms) | Finalize Output (ms) | PDE Compute (ms) | Total Step (ms) |\n")
        f.write("|:---|---:|---:|---:|---:|---:|\n")
        for d in data:
            f.write(f"| `{d['framework']}` | {d['prep_mean']:.3f} ({d['prep_mean']/d['step_mean']*100:.1f}%) | {d['inf_mean']:.3f} ({d['inf_mean']/d['step_mean']*100:.1f}%) | {d['fin_mean']:.3f} ({d['fin_mean']/d['step_mean']*100:.1f}%) | {d['compute_mean']:.3f} ({d['compute_mean']/d['step_mean']*100:.1f}%) | {d['step_mean']:.3f} |\n")
            
        f.write("\n## 3. Controller / Rank 0 ML Transaction Deep Dive\n\n")
        f.write("| Framework | Input Transport (ms) | Wait / Idle (ms) | GPU Forward (ms) | H2D/D2H Copy (ms) | Output Transport (ms) | Transaction Total (ms) |\n")
        f.write("|:---|---:|---:|---:|---:|---:|---:|\n")
        for d in data:
            f.write(f"| `{d['framework']}` | {d['tx_transport_in_r0']:.3f} | {d['tx_wait_r0']:.3f} | {d['tx_gpu_compute_r0']:.3f} | {d['tx_gpu_copy_r0']:.3f} | {d['tx_transport_out_r0']:.3f} | **{d['inf_r0']:.3f}** |\n")
            
    print(f"[+] Saved comparison Markdown summary to: {md_path}")


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Framework CMI Comparative Analysis & Visualization Tool")
    parser.add_argument("--results-dirs", nargs="+", type=Path, required=True, help="List of results directories (e.g. results_gpu_smartsim_c0 ...)")
    parser.add_argument("--output-dir", type=Path, default=Path("results_gpu_comparison"), help="Output directory for comparison plots & tables")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    framework_data = []
    
    name_map = {
        "results_gpu_smartsim_c0": "SmartSim c=0",
        "results_gpu_smartsim_c1": "SmartSim c=1",
        "results_gpu_smartsim_c2": "SmartSim c=2",
        "results_gpu_smartsim_c3": "SmartSim c=3",
        "results_gpu_aix_collective": "AIx Collective",
        "results_gpu_aix_pipelined": "AIx Pipelined",
        "results_gpu_phydll_cpp": "PhyDLL C++",
    }
    
    for r_dir in args.results_dirs:
        csv_file = r_dir / "cmi_phase_summary.csv"
        if not csv_file.exists():
            print(f"[WARN] Skipping {r_dir}: missing cmi_phase_summary.csv")
            continue
        
        df = parse_summary_csv(csv_file)
        fname = name_map.get(r_dir.name, r_dir.name.replace("results_gpu_", "").replace("_", " ").title())
        metrics = extract_framework_metrics(df, fname)
        framework_data.append(metrics)
        
    if not framework_data:
        print("[ERROR] No valid framework data found.")
        sys.exit(1)
        
    # Generate all comparative figures
    plot_warm_step_stacked_bars(framework_data, args.output_dir / "fig_comparison_warm_step_stacked.png")
    plot_ml_transaction_breakdown_bars(framework_data, args.output_dir / "fig_comparison_ml_transaction_stacked.png")
    plot_imbalance_comparison_bars(framework_data, args.output_dir / "fig_comparison_imbalance_mean_vs_rank0.png")
    plot_executive_dashboard(framework_data, args.output_dir / "fig_comparison_executive_dashboard.png")
    
    # Export reports
    export_framework_comparison_reports(framework_data, args.output_dir)
    print(f"\n[✓] Cross-framework comparison complete. Artifacts saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
