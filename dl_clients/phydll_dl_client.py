#!/usr/bin/env python3
import os
import sys
import struct
import numpy as np
import mpi4py
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
import torch

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
    # struct BcastMetaHeader {
    #     int32_t magic;           // 0
    #     int32_t version;         // 4
    #     int32_t model_len;       // 8
    #     int32_t backend_len;     // 12
    #     int32_t device_len;      // 16
    #     int32_t num_inputs;      // 20
    #     int32_t num_outputs;     // 24
    #     // 4 bytes padding for 8-byte alignment of next field
    #     int64_t total_input;     // 32
    #     int64_t total_output;    // 40
    #     int32_t dtype;           // 48
    #     int32_t layout;          // 52
    # } (Total: 56 bytes)
    
    header_size = 56
    header_buf = bytearray(header_size)
    status = MPI.Status()
    # Tag is source_rank to match C++ provider
    comm.Recv([header_buf, MPI.BYTE], source=source_rank, tag=source_rank, status=status)
    
    magic, version, m_len, b_len, d_len, n_in, n_out, t_in, t_out, dtype, layout = struct.unpack("=7i 4x 2q 2i", header_buf)
    
    if magic != 0x4D4C434D or version != 1:
        return {'valid': False}
        
    payload_size = m_len + b_len + d_len + (n_in + n_out) * 8
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
    
    in_sizes = []
    if n_in > 0:
        in_sizes = list(struct.unpack(f"={n_in}q", payload_buf[offset:offset+n_in*8]))
        offset += n_in*8
    out_sizes = []
    if n_out > 0:
        out_sizes = list(struct.unpack(f"={n_out}q", payload_buf[offset:offset+n_out*8]))
        
    return {
        'valid': True,
        'model_path': model_path,
        'backend': backend,
        'device': device,
        'total_input': t_in,
        'total_output': t_out,
        'in_sizes': in_sizes,
        'out_sizes': out_sizes
    }

def main():
    world_comm = MPI.COMM_WORLD
    
    # Handle MPMD split (identical logic to C++ client)
    # Handle MPMD split
    # We no longer rely on MPI_APPNUM, because Slurm srun with OpenMPI 5 assigns appnum 0 to both components!
    # Since this script is ALWAYS the DL client, we unconditionally assign it color MPI_UNDEFINED.
    color = MPI.UNDEFINED
    local_comm = world_comm.Split(color, 0)
    
    # Check if we are in the DL application group

    dl_count = int(os.environ.get("PHYDLL_DL_COUNT", "1"))
    
    dll = PhyDLL()
    dll.init("dl")
    dll.define_dl(count=dl_count)
    
    dist_info = dll.get_distribution_info()
    ndest = dist_info["ndest"]
    dests = dist_info["dest"]
    field_size = dll.get_field_size()
    
    # C++ client has a Barrier here
    world_comm.Barrier()
    
    meta_initialized = False
    model_loaded = False
    model_path = ""
    device_name = ""
    total_input_size = 0
    total_output_size = 0
    
    # Receive metadata from each connected physical rank
    for source_rank in dests:
        p2p_meta = receive_p2p_metadata(world_comm, source_rank)
        if p2p_meta['valid']:
            if not meta_initialized:
                model_path = p2p_meta['model_path']
                device_name = p2p_meta['device']
                meta_initialized = True
            total_input_size += p2p_meta['total_input']
            total_output_size += p2p_meta['total_output']

    torch_device = torch.device('cpu')
    model = None
    
    # Main loop
    frame_id = 0
    while dll.is_phy_signal():
        # Receive fields from PhyDLL
        # With dl_count=1, pyphydll.recv() returns a dict with one entry
        fields = dll.recv()
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
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    torch_device = torch.device('cuda', 0)
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

        output = np.zeros(field_size, dtype=np.float64)
        used_model = False
        
        if model_loaded and model is not None:
            batch_size = max(1, ndest)
            input_per_rank_total = field_size // batch_size
            input_per_rank_used = total_input_size // batch_size
            output_stride = field_size // batch_size
            outputs_per_rank_used = total_output_size // batch_size
            
            # Pack batch
            input_flat = np.zeros(total_input_size, dtype=np.float32)
            for b in range(batch_size):
                input_flat[b * input_per_rank_used : (b+1) * input_per_rank_used] = \
                    combined_data[b * input_per_rank_total : b * input_per_rank_total + input_per_rank_used]
            
            input_tensor = torch.from_numpy(input_flat).reshape(batch_size, input_per_rank_used).to(torch_device)
            
            try:
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                    # Result is [batch_size, outputs_per_rank_used]
                    output_np = output_tensor.cpu().contiguous().numpy().flatten()
                    
                # Scatter back to output buffer
                for b in range(batch_size):
                    output[b * output_stride : b * output_stride + outputs_per_rank_used] = \
                        output_np[b * outputs_per_rank_used : (b+1) * outputs_per_rank_used]
                used_model = True
            except Exception as e:
                print(f"[PHYDLL:DL:PY] forward failed: {e}", file=sys.stderr)
                world_comm.Abort(1)
        
        if not used_model:
            # Fallback: negate inputs
            size = min(len(combined_data), len(output))
            output[:size] = -combined_data[:size]
            
        # Send results back
        dll.send({"DL-OUT": output})
        frame_id += 1

    dll.finalize()

if __name__ == "__main__":
    main()
    # mpi4py calls Finalize at exit unless configured otherwise
    # But we can call it explicitly
    MPI.Finalize()
