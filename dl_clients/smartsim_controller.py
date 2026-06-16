#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
import socket
from pathlib import Path

from smartsim.experiment import Experiment

def find_free_port(start_port=6780, max_attempts=100):
    """Finds a free port starting from the given port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port+max_attempts}")

def detect_available_cores():
    """Detects available CPU cores, respecting Slurm allocations if present."""
    # Try standard Slurm allocation
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            # Handle cases where Slurm reports ranges like "16(x2),32"
            return int(slurm_cpus.split('(')[0].split(',')[0])
        except ValueError:
            pass
    # Fallback to raw machine cores
    return os.cpu_count() or 1

def main() -> int:
    parser = argparse.ArgumentParser(description="Start SmartSim DB Orchestrator.")
    
    # Network & Topology
    parser.add_argument("--port", type=int, default=6780, help="Base database port")
    parser.add_argument("--auto-port", action="store_true", help="Automatically find a free port")
    parser.add_argument("--interface", default="lo", help="Network interface (e.g., lo, ib0)")
    parser.add_argument("--db-nodes", type=int, default=1, help="Number of database nodes/shards")
    
    # Threading & Performance
    parser.add_argument("--inter-op-threads", type=int, default=1, help="Parallelism between independent graph ops")
    parser.add_argument("--intra-op-threads", type=int, default=0, help="Parallelism within a single op. 0 = auto-detect cores")
    parser.add_argument("--threads-per-queue", type=int, default=1, help="Threads per queue (usually 1 for GPU, >1 for CPU)")
    parser.add_argument("--use-default-cpu-settings", action="store_true", help="Use SmartSim's default CPU binding settings (does not set --cpu-cores-per-node, --intra-op-threads, --inter-op-threads and --threads-per-queue at all, allowing SmartSim to auto-configure based on the environment)")
    parser.add_argument("--use-gpu", action="store_true", help="Set SmartSim backend to expect GPU usage")
    
    # Synchronization
    parser.add_argument("--endpoint-file", default=".ssdb_endpoint", help="File to write host:port")
    parser.add_argument("--done-file", default=".solver_done", help="File that signals solver completion")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Readiness timeout for the DB")
    
    # Slurm/Env Overrides
    parser.add_argument("--launcher", default=None, choices=["local", "slurm"], help="Launcher to use. Auto-detected if omitted.")
    parser.add_argument("--cpu-cores-per-node", type=int, default=0, help="Explicit CPU cores to bind DB to (overrides auto-detect for db.set_cpus)")
    parser.add_argument("--no-cpu-bind", action="store_true", help="Skip db.set_cpus() completely (do not explicitly bind DB step)")
    parser.add_argument("--het-group", default=None, type=str, help="Heterogeneous group to run the database on (Slurm only)")
    parser.add_argument("--exp-dir", default=None, help="Base experiment directory")
    parser.add_argument("--silent", action="store_true", help="Suppress output")
    
    args = parser.parse_args()

    # Determine launcher
    launcher = args.launcher
    if not launcher:
        launcher = "slurm" if "SLURM_JOB_ID" in os.environ else "local"

    # Determine port
    port = args.port
    if args.auto_port:
        port = find_free_port(start_port=args.port)
        if not args.silent:
            print(f"Auto-selected free port: {port}")

    # Determine intra_op_threads
    if args.intra_op_threads <= 0:
        intra_threads = detect_available_cores()
        if not args.silent:
            print(f"Auto-detected {intra_threads} cores for intra-op-threads.")
    else:
        intra_threads = args.intra_op_threads
        if not args.silent:
            print(f"Using manual override: {intra_threads} for intra-op-threads.")

    # Paths
    endpoint_file = Path(args.endpoint_file)
    done_file = Path(args.done_file)

    if endpoint_file.exists():
        endpoint_file.unlink()
    if done_file.exists():
        done_file.unlink()

    exp_name = f"smartsim_orchestrator_{int(time.time())}"
    if args.exp_dir:
        exp_path = Path(args.exp_dir) / exp_name
    else:
        exp_path = Path(os.getcwd()) / "smartsim_experiments" / exp_name
    
    if exp_path.exists():
        shutil.rmtree(exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)

    if not args.silent:
        print("--- Orchestrator Configuration ---")
        print(f"Launcher: {launcher}")
        print(f"Port: {port}")
        print(f"Interface: {args.interface}")
        print(f"DB Nodes: {args.db_nodes}")
        if args.use_default_cpu_settings:
            print("Using SmartSim's default CPU settings (no overrides).")
        else:
            print(f"Intra-op Threads: {intra_threads}")
            print(f"Inter-op Threads: {args.inter_op_threads}")
            print(f"Threads per Queue: {args.threads_per_queue}")
        print(f"Exp Path: {exp_path}")
        print("----------------------------------", flush=True)

    exp = Experiment(name=exp_name, launcher=launcher, exp_path=str(exp_path))
    
    if args.use_default_cpu_settings:
        if not args.silent:
            print("Using SmartSim's default CPU settings. No overrides will be applied for --cpu-cores-per-node, --intra-op-threads, --inter-op-threads, or --threads-per-queue.")
        db = exp.create_database(
            port=port, 
            interface=args.interface, 
            db_nodes=args.db_nodes, 
            single_cmd=False, 
            batch=False,
            # Do not set intra_op_threads, inter_op_threads, or threads_per_queue to allow SmartSim to auto-configure
        )
    else:
        db = exp.create_database(
            port=port, 
            interface=args.interface, 
            db_nodes=args.db_nodes, 
            single_cmd=False, 
            batch=False,
            intra_op_threads=intra_threads,
            inter_op_threads=args.inter_op_threads,
            threads_per_queue=args.threads_per_queue
        )

    if launcher == "slurm":
        db.set_run_arg("export", "ALL")
        db.set_run_arg("mem", "0")
    
    if launcher == "slurm" and args.het_group and ("SLURM_HET_SIZE" in os.environ or "SLURM_JOB_NUM_NODES_HET_GROUP_0" in os.environ):
        db.set_run_arg("het-group", args.het_group)
        
    if not args.use_gpu and not args.use_default_cpu_settings and not args.no_cpu_bind:
        bind_cpus = args.cpu_cores_per_node if args.cpu_cores_per_node > 0 else intra_threads
        db.set_cpus(bind_cpus)
        if not args.silent:
            print(f"Binding database to {bind_cpus} CPU cores per node.")

    exp.start(db, block=False, summary=not args.silent)

    # Apply clustered Redis stability config if needed
    if args.db_nodes > 1:
        timeout_ms = os.environ.get("SMARTSIM_CLUSTER_NODE_TIMEOUT_MS", "120000")
        require_full_coverage = os.environ.get("SMARTSIM_CLUSTER_REQUIRE_FULL_COVERAGE", "no")
        conf_retries = int(os.environ.get("SMARTSIM_DB_CONF_RETRIES", "30"))
        
        config_applied = False
        for attempt in range(1, conf_retries + 1):
            try:
                db.set_db_conf("cluster-node-timeout", timeout_ms)
                db.set_db_conf("cluster-require-full-coverage", require_full_coverage)
                if not args.silent:
                    print(f"Applied clustered Redis stability config (attempt {attempt}).")
                config_applied = True
                break
            except Exception as exc:
                time.sleep(1)

        if not config_applied:
            if not args.silent:
                print("Warning: Failed to apply clustered Redis stability config.")

    # Wait for readiness
    start = time.time()
    addresses = None
    while time.time() - start < args.timeout_s:
        try:
            addresses = db.get_address()
            if addresses:
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not addresses:
        exp.stop(db)
        raise RuntimeError("SmartSim database did not become ready in time.")

    endpoint = ",".join(addresses)
    endpoint_file.write_text(endpoint + "\n", encoding="utf-8")
    if not args.silent:
        print(f"Database ready. SSDB={endpoint}", flush=True)

    # Monitor done file
    if not args.silent:
        print(f"Waiting for solver completion (monitoring {done_file})...", flush=True)
    while not done_file.exists():
        time.sleep(0.5)

    if not args.silent:
        print("Solver done signaled. Shutting down database.", flush=True)
    exp.stop(db)
    
    try:
        done_file.unlink()
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
