#!/usr/bin/env python3
"""
Unit test for decode_metadata_header in dl_clients/phydll_dl_client.py.

Builds an 88-byte BcastMetaHeader using the exact C++ struct offsets
(ml_coupling_provider_phydll.hpp) and verifies the Python decoder reads every
field correctly, including the readiness flag replacing the former padding.

Run locally:  python3 test/phydll_mpmd/test_metadata_decode.py
"""

import os
import struct
import sys

import ast

CLIENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dl_clients", "phydll_dl_client.py"))

# Test the pure decoder without importing Torch, MPI, or starting the MPMD split.
with open(CLIENT) as source:
    tree = ast.parse(source.read())
decoder = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "decode_metadata_header")
namespace = {'struct': struct}
exec(compile(ast.Module(body=[decoder], type_ignores=[]), CLIENT, 'exec'), namespace)
decode = namespace['decode_metadata_header']


def build_header(**overrides):
    buf = bytearray(88)

    def put_int32(off, val):
        struct.pack_into("=i", buf, off, val)

    def put_int64(off, val):
        struct.pack_into("=q", buf, off, val)

    fields = {
        'magic': 0x4D4C434D,
        'version': 4,
        'model_len': 12,
        'backend_len': 5,
        'device_len': 3,
        'batch_size': 0,
        'num_inputs': 1,
        'num_outputs': 1,
        'total_input': 90,
        'total_output': 5,
        'dtype': 1,
        'layout': 0,
        'num_input_dims': 2,
        'num_output_dims': 1,
        'layout_kind': 1,
        'phy_count': 18,
        'dl_count': 1,
        'field_size': 3,
        'solver_readiness_wait': 0,
    }
    fields.update(overrides)

    put_int32(0, fields['magic'])
    put_int32(4, fields['version'])
    put_int32(8, fields['model_len'])
    put_int32(12, fields['backend_len'])
    put_int32(16, fields['device_len'])
    put_int32(20, fields['batch_size'])
    put_int32(24, fields['num_inputs'])
    put_int32(28, fields['num_outputs'])
    put_int64(32, fields['total_input'])
    put_int64(40, fields['total_output'])
    put_int32(48, fields['dtype'])
    put_int32(52, fields['layout'])
    put_int32(56, fields['num_input_dims'])
    put_int32(60, fields['num_output_dims'])
    put_int32(64, fields['layout_kind'])
    put_int32(68, fields['phy_count'])
    put_int32(72, fields['dl_count'])
    put_int32(76, fields['solver_readiness_wait'])
    put_int64(80, fields['field_size'])
    return buf


failures = 0


def check(label, cond):
    global failures
    print(("PASS" if cond else "FAIL") + "  " + label)
    if not cond:
        failures += 1


h = decode(build_header())
check("header is 88 bytes exactly", struct.calcsize("=8i 2q 8i q") == 88)
check("valid", h['valid'])
check("magic", h['magic'] == 0x4D4C434D)
check("version", h['version'] == 4)
check("readiness disabled", h['solver_readiness_wait'] is False)
check("readiness enabled", decode(build_header(solver_readiness_wait=1))['solver_readiness_wait'] is True)
check("old runtime version rejected", not decode(build_header(version=3))['valid'])
try:
    decode(build_header(solver_readiness_wait=2))
    check("invalid readiness flag raises", False)
except RuntimeError:
    check("invalid readiness flag raises", True)
check("layout_kind uniform_chunks", h['layout_kind'] == 1)
check("phy_count", h['phy_count'] == 18)
check("dl_count", h['dl_count'] == 1)
check("field_size (after readiness flag)", h['field_size'] == 3)
check("total_input", h['total_input'] == 90)
check("total_output", h['total_output'] == 5)
check("num_input_dims", h['num_input_dims'] == 2)
check("num_output_dims", h['num_output_dims'] == 1)

# Round-trip with a large field_size to make sure the pad is really skipped.
h2 = decode(build_header(field_size=50000, dl_count=18, phy_count=1,
                         total_input=3, total_output=900000))
check("large field_size round-trip", h2['field_size'] == 50000)
check("dl_count=18 round-trip", h2['dl_count'] == 18)
check("total_output round-trip", h2['total_output'] == 900000)

# Invalid magic -> not valid.
h3 = decode(build_header(magic=0))
check("bad magic rejected", not h3['valid'])

# Bad layout -> raises.
try:
    decode(build_header(layout_kind=7))
    check("bad layout raises", False)
except RuntimeError:
    check("bad layout raises", True)

print()
if failures:
    print(f"{failures} FAILURE(S)")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
