#!/usr/bin/env python3
import os
import sys
import time
print("[DL] Starting Python DL Client...", flush=True)
import struct
import numpy as np
PY_SCOREP_WRAPPER = os.environ.get("PHYDLL_PY_SCOREP_WRAPPER", "0") == "1"

print("[DL] Importing mpi4py...", flush=True)
import mpi4py
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
print("[DL] mpi4py imported successfully. MPI world size =", MPI.COMM_WORLD.Get_size(), "rank =", MPI.COMM_WORLD.Get_rank(), flush=True)
print("[DL] Importing torch...", flush=True)
import torch
print("[DL] torch imported successfully.", flush=True)
import contextlib
ENABLE_SCOREP_USER = os.environ.get("ENABLE_SCOREP_USER", "0") == "1"
HAS_SCOREP = False
try:
    if ENABLE_SCOREP_USER:
        print("[DL] ENABLE_SCOREP_USER is set. Importing scorep.user...", flush=True)
        import scorep.user
        import scorep.instrumenter
        scorep.instrumenter.enable()
        HAS_SCOREP = True
        print("[DL] scorep.user and instrumenter enabled successfully. HAS_SCOREP=True", flush=True)
        inst = scorep.instrumenter.get_instrumenter()
        print(f"[DL] Instrumenter class: {inst.__class__.__name__}", flush=True)
        print(f"[DL] Instrumenter registered status: {inst.get_registered()}", flush=True)
    else:
        print("[DL] ENABLE_SCOREP_USER not set to 1.", flush=True)
except Exception as e:
    print(f"[DL] Error importing/enabling scorep: {e}", flush=True)

@contextlib.contextmanager
def scorep_region(name):
    if HAS_SCOREP:
        with scorep.user.region(name):
            yield
    else:
        yield


# Add phydll to path if it's not installed
script_dir = os.path.dirname(os.path.abspath(__file__))
# pyphydll is in ../extern/phydll/src/python/pyphydll
# We want to add the parent of the pyphydll package to sys.path
phydll_py_parent = os.path.abspath(os.path.join(script_dir, "../extern/phydll/src/python"))
if phydll_py_parent not in sys.path:
    sys.path.insert(0, phydll_py_parent)

try:
    from pyphydll.pyphydll import PhyDLL
except ImportError as e:
    print(f"Error: Could not import pyphydll: {e}", file=sys.stderr)
    print(f"PYTHONPATH={sys.path}", file=sys.stderr)
    sys.exit(1)

def receive_p2p_metadata(comm, source_rank):
    """
    Replicates receive_p2p_metadata from dl_client.cpp
    """
    from mpi4py import MPI

    header_size = 88
    header_buf = bytearray(header_size)
    status = MPI.Status()
    # Tag is source_rank to match C++ provider
    comm.Recv([header_buf, MPI.BYTE], source=source_rank, tag=source_rank, status=status)

    header = decode_metadata_header(header_buf)
    if not header['valid']:
        return header

    m_len, b_len, d_len, n_in, n_out, n_in_dims, n_out_dims = (
        header['model_len'], header['backend_len'], header['device_len'],
        header['num_inputs'], header['num_outputs'],
        header['num_input_dims'], header['num_output_dims'])

    payload_size = m_len + b_len + d_len + (n_in_dims + n_out_dims) * 8
    payload_buf = bytearray(payload_size)
    if payload_size > 0:
        comm.Recv([payload_buf, MPI.BYTE], source=source_rank, tag=source_rank)
        
    offset = 0
    model_path = payload_buf[offset:offset+m_len].decode('utf-8') if m_len > 0 else ""
    offset += m_len
    backend = payload_buf[offset:offset+b_len].decode('utf-8') if b_len > 0 else ""
    offset += b_len
    device = payload_buf[offset:offset+d_len].decode('utf-8') if d_len > 0 else ""
    offset += d_len
    
    in_shapes = []
    if n_in_dims > 0:
        flat_in_dims = list(struct.unpack(f"={n_in_dims}q", payload_buf[offset:offset+n_in_dims*8]))
        offset += n_in_dims*8
        
        d_idx = 0
        for _ in range(n_in):
            if d_idx >= len(flat_in_dims): break
            ndim = flat_in_dims[d_idx]
            d_idx += 1
            shape = []
            for _ in range(ndim):
                if d_idx >= len(flat_in_dims): break
                shape.append(flat_in_dims[d_idx])
                d_idx += 1
            in_shapes.append(shape)
            
    out_shapes = []
    if n_out_dims > 0:
        flat_out_dims = list(struct.unpack(f"={n_out_dims}q", payload_buf[offset:offset+n_out_dims*8]))
        
        d_idx = 0
        for _ in range(n_out):
            if d_idx >= len(flat_out_dims): break
            ndim = flat_out_dims[d_idx]
            d_idx += 1
            shape = []
            for _ in range(ndim):
                if d_idx >= len(flat_out_dims): break
                shape.append(flat_out_dims[d_idx])
                d_idx += 1
            out_shapes.append(shape)
        
    return {
        'valid': True,
        'model_path': model_path,
        'backend': backend,
        'device': device,
        'batch_size': header['batch_size'],
        'total_input': header['total_input'],
        'total_output': header['total_output'],
        'in_shapes': in_shapes,
        'out_shapes': out_shapes,
        'field_size': header['field_size'],
        'layout_kind': header['layout_kind'],
        'phy_count': header['phy_count'],
        'dl_count': header['dl_count']
    }


def decode_metadata_header(header_buf):
    """
    Decode the 88-byte BcastMetaHeader sent by the C++ provider
    (ml_coupling_provider_phydll.hpp BcastMetaHeader), matching dl_client.cpp.

    Layout (natural C++ alignment):
        int32_t magic;           // 0
        int32_t version;         // 4
        int32_t model_len;       // 8
        int32_t backend_len;     // 12
        int32_t device_len;      // 16
        int32_t batch_size;      // 20
        int32_t num_inputs;      // 24
        int32_t num_outputs;     // 28
        int64_t total_input;     // 32
        int64_t total_output;    // 40
        int32_t dtype;           // 48
        int32_t layout;          // 52
        int32_t num_input_dims;  // 56
        int32_t num_output_dims; // 60
        int32_t layout_kind;     // 64 (0 = packed, 1 = uniform_chunks)
        int32_t phy_count;       // 68
        int32_t dl_count;        // 72
        [4-byte alignment pad]   // 76
        int64_t field_size;      // 80
    } (Total: 88 bytes)

    "=8i 2q 7i 4x q": 8 int32, 2 int64, 7 int32, 4 pad bytes, 1 int64 == 88 bytes.
    """
    (magic, version, model_len, backend_len, device_len, batch_size,
     num_inputs, num_outputs, total_input, total_output, dtype, layout,
     num_input_dims, num_output_dims, layout_kind, phy_count, dl_count,
     field_size) = struct.unpack("=8i 2q 7i 4x q", header_buf)

    if magic != 0x4D4C434D or version != 3:
        return {'valid': False}
    if layout_kind not in (0, 1):
        print(f"[DL] ERROR: unsupported transport layout {layout_kind} "
              "(valid: 0 = packed, 1 = uniform_chunks).", flush=True)
        raise RuntimeError(f"Unknown PhyDLL transport layout {layout_kind}.")

    return {
        'valid': True,
        'magic': magic,
        'version': version,
        'model_len': model_len,
        'backend_len': backend_len,
        'device_len': device_len,
        'batch_size': batch_size,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'total_input': total_input,
        'total_output': total_output,
        'dtype': dtype,
        'layout': layout,
        'num_input_dims': num_input_dims,
        'num_output_dims': num_output_dims,
        'layout_kind': layout_kind,
        'phy_count': phy_count,
        'dl_count': dl_count,
        'field_size': field_size,
    }

def main():
    dll = None
    world_comm = MPI.COMM_WORLD
    try:
        # Configure Torch threading
        intra_threads = int(os.environ.get("MLCOUPLING_INTRA_OP_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "-1")))
        inter_threads = int(os.environ.get("MLCOUPLING_INTER_OP_THREADS", "-1"))

        if intra_threads > 0:
            torch.set_num_threads(intra_threads)
        if inter_threads > 0:
            torch.set_num_interop_threads(inter_threads)

        dl_count = int(os.environ.get("PHYDLL_DL_FIELD_COUNT", os.environ.get("PHYDLL_DL_COUNT", "1")))
        
        # Match the original Python DL startup: participate in the MPMD split
        # before entering PhyDLL's own internal MPI split.
        color = MPI.UNDEFINED
        print("[DL] Entering world_comm.Split(color=MPI.UNDEFINED)", flush=True)
        local_comm = world_comm.Split(color, world_comm.Get_rank())
        print("[DL] Returned from world_comm.Split", flush=True)
        if local_comm != MPI.COMM_NULL:
            local_comm.Free()
        
        print("[DL] constructing PhyDLL...", flush=True)
        dll = PhyDLL()
        print("[DL] calling dll.init(...)", flush=True)
        dll.init("dl")
        print("[DL] Calling dll.define_dl...", flush=True)
        if intra_threads > 0 or inter_threads > 0:
            print(f"[DL] torch threads: intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()}", flush=True)
        
        dll.define_dl(count=dl_count)
        print("[DL] Returned from dll.define_dl.", flush=True)

        print("[DL] calling dll.get_local_mpi_comm()...", flush=True)
        local_comm = dll.get_local_mpi_comm()
        local_comm.Get_rank()
        local_comm.Get_size()
        print("[DL] Returned from dll.get_local_mpi_comm().", flush=True)
        
        dist_info = dll.get_distribution_info()
        ndest = dist_info["ndest"]
        dests = dist_info["dest"]
        field_size = dll.get_field_size()
        
        meta_initialized = False
        model_loaded = False
        model_path = ""
        device_name = ""
        total_input_size = 0
        total_output_size = 0
        final_meta = None
        rank_batch_sizes = {}
        rank_field_sizes = {}
        rank_total_input = {}
        rank_total_output = {}
        rank_layout_kind = {}
        rank_phy_count = {}
        rank_dl_count = {}
        
        # Receive metadata from each connected physical rank
        for source_rank in dests:
            p2p_meta = receive_p2p_metadata(world_comm, source_rank)
            if p2p_meta['valid']:
                if not meta_initialized:
                    model_path = p2p_meta.get('model_path', '')
                    device_name = p2p_meta.get('device', '')
                    final_meta = p2p_meta
                    meta_initialized = True
                total_input_size += p2p_meta.get('total_input', 0)
                total_output_size += p2p_meta.get('total_output', 0)
                if p2p_meta.get('in_shapes') and len(p2p_meta['in_shapes']) > 0:
                    rank_batch_sizes[source_rank] = p2p_meta['in_shapes'][0][0]
                rank_field_sizes[source_rank] = p2p_meta.get('field_size', 0)
                rank_total_input[source_rank] = p2p_meta.get('total_input', 0)
                rank_total_output[source_rank] = p2p_meta.get('total_output', 0)
                rank_layout_kind[source_rank] = p2p_meta.get('layout_kind', 0)
                rank_phy_count[source_rank] = p2p_meta.get('phy_count', 0)
                rank_dl_count[source_rank] = p2p_meta.get('dl_count', 0)

        # Resolve and validate the transport layout across all coupled ranks.
        first_rank = dests[0]
        uniform_chunks = rank_layout_kind.get(first_rank, 0) == 1
        for source_rank in dests:
            if rank_layout_kind.get(source_rank, 0) != rank_layout_kind.get(first_rank, 0):
                print(f"[DL] ERROR: mixed transport layouts across coupled ranks "
                      f"(rank {source_rank}: {rank_layout_kind.get(source_rank, 0)}, "
                      f"rank {first_rank}: {rank_layout_kind.get(first_rank, 0)}).", file=sys.stderr, flush=True)
                world_comm.Abort(1)
            if uniform_chunks and rank_phy_count.get(source_rank, 0) != rank_phy_count.get(first_rank, 0):
                print(f"[DL] ERROR: uniform_chunks requires identical PHY field counts across ranks "
                      f"(rank {source_rank}: {rank_phy_count.get(source_rank, 0)}, "
                      f"rank {first_rank}: {rank_phy_count.get(first_rank, 0)}).", file=sys.stderr, flush=True)
                world_comm.Abort(1)

        if uniform_chunks:
            if dll.dl_count != final_meta.get('dl_count', 0):
                print(f"[DL] ERROR: uniform_chunks provider expects dl_count={final_meta.get('dl_count', 0)} "
                      f"but the client was launched with PHYDLL_DL_FIELD_COUNT={dll.dl_count}.", file=sys.stderr, flush=True)
                world_comm.Abort(1)
            if dll.phy_count != final_meta.get('phy_count', 0):
                print(f"[DL] ERROR: uniform_chunks PHY field count mismatch: PhyDLL reports {dll.phy_count}, "
                      f"provider sent {final_meta.get('phy_count', 0)}.", file=sys.stderr, flush=True)
                world_comm.Abort(1)
            # dll.size must equal the sum of per-rank per-field sizes.
            expected_agg = sum(rank_field_sizes.get(r, 0) for r in dests)
            if dll.size != expected_agg:
                print(f"[DL] ERROR: uniform_chunks aggregated field size mismatch: PhyDLL reports {dll.size}, "
                      f"expected sum of per-rank sizes = {expected_agg}.", file=sys.stderr, flush=True)
                world_comm.Abort(1)
            for source_rank in dests:
                g_r = rank_field_sizes.get(source_rank, 0)
                t_in = rank_total_input.get(source_rank, 0)
                t_out = rank_total_output.get(source_rank, 0)
                phy_c = rank_phy_count.get(source_rank, 0)
                dl_c = rank_dl_count.get(source_rank, 0)
                if t_in != phy_c * g_r or t_out != dl_c * g_r:
                    print(f"[DL] ERROR: uniform_chunks invariant violated for rank {source_rank}: "
                          f"total_input={t_in} != phy_count*field_size={phy_c}*{g_r}, "
                          f"total_output={t_out} != dl_count*field_size={dl_c}*{g_r}.",
                          file=sys.stderr, flush=True)
                    world_comm.Abort(1)

        torch_device = torch.device('cpu')
        model = None
        
        # Main loop
        frame_id = 0
        while dll.is_phy_signal():
            # Receive fields from PhyDLL
            if uniform_chunks:
                # Bypass the dict: retrieve each field in core order via get_field().
                # Labels are distinct (PHY-IN-###) so they cannot be stored in a dict
                # without losing order; we use them for validation only.
                print(f"[DL {world_comm.rank}] Calling dll.recv(only=True) (uniform_chunks)...", flush=True)
                with scorep_region("py_recv"):
                    dll.recv(only=True)
                received = []
                for i in range(dll.phy_count):
                    field, label = dll.get_field()
                    expected_label = f"PHY-IN-{i:03d}"
                    if label != expected_label:
                        print(f"[DL] ERROR: unexpected field label '{label}' at index {i} "
                              f"(expected '{expected_label}').", file=sys.stderr, flush=True)
                        world_comm.Abort(1)
                    if field.shape[0] != dll.size:
                        print(f"[DL] ERROR: field {i} has length {field.shape[0]}, "
                              f"expected aggregated size {dll.size}.", file=sys.stderr, flush=True)
                        world_comm.Abort(1)
                    received.append(field)
                combined_data = None
            else:
                # With dl_count=1, pyphydll.recv() returns a dict with one entry
                print(f"[DL {world_comm.rank}] Calling dll.recv()...", flush=True)
                with scorep_region("py_recv"):
                    fields = dll.recv()
                print(f"[DL {world_comm.rank}] Returned from dll.recv().", flush=True)
                combined_data = fields.get("PHY-DATA", None)
                if combined_data is None:
                    # Fallback
                    if fields:
                        combined_data = next(iter(fields.values()))
                    else:
                        combined_data = np.zeros(field_size)

            # Load model if metadata was received (deferred load like C++ client)
            if meta_initialized and not model_loaded:
                wants_gpu = bool(device_name) and device_name.upper() != "CPU"
                if wants_gpu:
                    # Query the communicator containing only the DL ranks
                    dl_comm = dll.get_local_mpi_comm()
                    try:
                        local_dl_comm = None
                        local_dl_comm = dl_comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
                        try:
                            local_dl_rank = local_dl_comm.rank
                        finally:
                            if local_dl_comm is not None:
                                local_dl_comm.Free()
                    finally:
                        dl_comm.Free()

                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        local_gpu_count = torch.cuda.device_count()
                        gpu_id = local_dl_rank % local_gpu_count
                        gpu_id = int(os.environ.get("PHYDLL_DL_GPU_ID", gpu_id))
                        print(f"[PHYDLL:DL:PY] Using local DL rank {local_dl_rank} mapped to GPU device index: {gpu_id}", file=sys.stderr)
                        torch_device = torch.device('cuda', gpu_id)
                    else:
                        print("[PHYDLL:DL:PY] requested GPU but no CUDA device available; using CPU", file=sys.stderr)
                        torch_device = torch.device('cpu')
                
                if model_path:
                    try:
                        model = torch.jit.load(model_path)
                        model.eval()
                        model.to(torch_device)
                        print("[PHYDLL:DL:PY] model loaded", file=sys.stderr)
                    except Exception as e:
                        print(f"[PHYDLL:DL:PY] Failed to load TorchScript model: {e}", file=sys.stderr)
                        world_comm.Abort(1)
                
                print(f"[PHYDLL:DL:PY] meta init model_path='{model_path}' total_input={total_input_size} total_output={total_output_size}", file=sys.stderr)
                model_loaded = True

            out_capacity = dll.size * dll.dl_count if uniform_chunks else field_size
            output = np.zeros(out_capacity, dtype=np.float64)
            used_model = False
            
            if model_loaded and model is not None:
                with scorep_region("py_inference"):
                    # Replicate the C++ client's robust dynamic shape and batch extraction logic
                    ndest = len(dests)
                    batch_size = 0
                    for i in range(ndest):
                        source_rank = dests[i]
                        batch_size += rank_batch_sizes.get(source_rank, 1)

                    input_per_rank_used = total_input_size // batch_size
                    
                    # Check for dynamic shapes
                    actual_shape = [batch_size]
                    if final_meta and final_meta.get('in_shapes') and len(final_meta['in_shapes']) > 0:
                        shape = final_meta['in_shapes'][0]
                        if len(shape) > 1:
                            actual_shape.extend(shape[1:])
                        else:
                            actual_shape.append(input_per_rank_used)
                    else:
                        actual_shape.append(input_per_rank_used)
                    
                    # Extract input features per sample (accounting for any rank padding)
                    with scorep_region("py_input_unpack"):
                        input_flat = np.zeros(total_input_size, dtype=np.float32)
                        if uniform_chunks:
                            # received[f] holds all rank segments for field f, in rank order.
                            # Reconstruct the rank-major flattened input directly:
                            #   input_flat[in_off[r] + f*g_r + b] = received[f][agg_off[r] + b]
                            agg_offsets = [0] * (ndest + 1)
                            in_offsets = [0] * (ndest + 1)
                            for i in range(ndest):
                                agg_offsets[i + 1] = agg_offsets[i] + rank_field_sizes.get(dests[i], 0)
                                in_offsets[i + 1] = in_offsets[i] + rank_total_input.get(dests[i], 0)
                            for i in range(ndest):
                                g_r = rank_field_sizes.get(dests[i], 0)
                                for f in range(dll.phy_count):
                                    src = agg_offsets[i]
                                    dst = in_offsets[i] + f * g_r
                                    input_flat[dst:dst + g_r] = received[f][src:src + g_r]
                        else:
                            offset_so_far = 0
                            src_rank_start = 0
                            for i in range(ndest):
                                source_rank = dests[i]
                                rank_batch = rank_batch_sizes.get(source_rank, 1)
                                for s in range(rank_batch):
                                    src_start = src_rank_start + s * input_per_rank_used
                                    dest_start = (offset_so_far + s) * input_per_rank_used
                                    input_flat[dest_start : dest_start + input_per_rank_used] = \
                                        combined_data[src_start : src_start + input_per_rank_used]
                                offset_so_far += rank_batch
                                src_rank_start += rank_field_sizes.get(source_rank, 0)
                        
                        input_tensor = torch.from_numpy(input_flat).reshape(batch_size, input_per_rank_used)
                        input_tensor = input_tensor.view(*actual_shape)

                    with scorep_region("py_h2d"):
                        input_tensor = input_tensor.to(torch_device)
                    
                    try:
                        with torch.no_grad():
                            max_chunk_size = final_meta.get('batch_size', 0)
                            if max_chunk_size <= 0:
                                max_chunk_size = batch_size
                            outputs = []
                            with scorep_region("py_torch_forward"):
                                for chunk_idx in range(0, batch_size, max_chunk_size):
                                    end_idx = min(chunk_idx + max_chunk_size, batch_size)
                                    chunk_tensor = input_tensor[chunk_idx:end_idx]
                                    outputs.append(model(chunk_tensor))
                                output_tensor = torch.cat(outputs, dim=0)
                            
                            with scorep_region("py_d2h"):
                                output_np = output_tensor.cpu().contiguous().numpy().flatten()
                            
                        # Scatter back to output buffer dynamically (matching C++ logic)
                        if output_np.shape[0] != total_output_size:
                            print(f"[PHYDLL:DL:PY] ERROR: model produced {output_np.shape[0]} elements "
                                  f"but metadata declared {total_output_size}. Refusing to send.", file=sys.stderr, flush=True)
                            world_comm.Abort(1)
                        
                        with scorep_region("py_output_reorder"):
                            if uniform_chunks:
                                # Build field-major wire layout:
                                #   wire[f][agg_off[r] + b] = out_np[out_off[r] + f*g_r + b]
                                agg_offsets = [0] * (ndest + 1)
                                out_offsets = [0] * (ndest + 1)
                                for i in range(ndest):
                                    agg_offsets[i + 1] = agg_offsets[i] + rank_field_sizes.get(dests[i], 0)
                                    out_offsets[i + 1] = out_offsets[i] + rank_total_output.get(dests[i], 0)
                                wire = np.zeros((dll.dl_count, dll.size), dtype=np.float64)
                                for i in range(ndest):
                                    g_r = rank_field_sizes.get(dests[i], 0)
                                    for f in range(dll.dl_count):
                                        src = out_offsets[i] + f * g_r
                                        dst = agg_offsets[i]
                                        wire[f, dst:dst + g_r] = output_np[src:src + g_r]
                                output = wire
                            else:
                                outputs_per_rank_used = total_output_size // batch_size
                                offset_so_far = 0
                                dest_rank_start = 0
                                for i in range(ndest):
                                    source_rank = dests[i]
                                    rank_batch = rank_batch_sizes.get(source_rank, 1)
                                    for s in range(rank_batch):
                                        dest_start = dest_rank_start + s * outputs_per_rank_used
                                        src_start = (offset_so_far + s) * outputs_per_rank_used
                                        output[dest_start : dest_start + outputs_per_rank_used] = \
                                            output_np[src_start : src_start + outputs_per_rank_used]
                                    offset_so_far += rank_batch
                                    dest_rank_start += rank_field_sizes.get(source_rank, 0)
                        used_model = True
                    except Exception as e:
                        print(f"[PHYDLL:DL:PY] forward failed: {e}", file=sys.stderr)
                        world_comm.Abort(1)
            
            if not used_model:
                if uniform_chunks:
                    # No model: send zeros in the field-major layout.
                    output = np.zeros((dll.dl_count, dll.size), dtype=np.float64)
                else:
                    # Fallback: negate inputs
                    size = min(len(combined_data), len(output))
                    output[:size] = -combined_data[:size]
                
            # Send results back
            with scorep_region("py_send"):
                if uniform_chunks:
                    # Register every DL output field (labels repeat; a dict would
                    # overwrite them, so use set_field directly).
                    for f in range(dll.dl_count):
                        dll.set_field(output[f], "DL-OUT")
                    dll.send()
                else:
                    dll.send({"DL-OUT": output})
            frame_id += 1

    finally:
        if dll is not None:
            try:
                prefix = f"[DL {world_comm.rank}]" if world_comm is not None else "[DL]"
                print(f"{prefix} Entering dll.finalize()", flush=True)
                dll.finalize()
                print(f"{prefix} Exited dll.finalize()", flush=True)
            except Exception as e:
                print(f"[PHYDLL:DL:PY] dll.finalize() failed: {e}", file=sys.stderr)
        # Keep this MPMD DL rank alive until every solver rank has destroyed its
        # PhyDLL provider and reached teardown, mirroring the C++ DL client and
        # the solver-side barrier in terrain_solver.cpp. Without this, srun aborts
        # the whole hetjob step once the DL task exits ahead of the solver tasks.
        if os.environ.get("PHYDLL_MPMD_SHUTDOWN_BARRIER", "0") == "1" and not MPI.Is_finalized():
            prefix = f"[DL {world_comm.rank}]" if world_comm is not None else "[DL]"
            print(f"{prefix} waiting for solver teardown (PHYDLL_MPMD_SHUTDOWN_BARRIER=1)", flush=True)
            world_comm.Barrier()
            print(f"{prefix} solver teardown barrier complete", flush=True)
            time.sleep(5)
        prefix = f"[DL {world_comm.rank}]" if world_comm is not None else "[DL]"
        print(f"{prefix} main() returning", flush=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        if HAS_SCOREP:
            try:
                print("[DL] Forcing Score-P finalization...", flush=True)
                scorep.user.force_finalize()
                print("[DL] Score-P finalization complete.", flush=True)
            except Exception as e:
                print(f"[DL] Warning: force_finalize failed: {e}", flush=True)
        if MPI.Is_initialized() and not MPI.Is_finalized():
            print("[DL] Entering MPI.Finalize()", flush=True)
            MPI.Finalize()
            print("[DL] Exited MPI.Finalize()", flush=True)
