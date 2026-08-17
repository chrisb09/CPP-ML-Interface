#!/usr/bin/env python3
"""
Generate deterministic TorchScript models for the PhyDLL uniform_chunks
MPMD regression test.

  model_18to1.pt : input [B,18] -> output [B]      (sum over dim 1)
  model_1to18.pt : input [B,1]  -> output [B,18]   (expand row value)

Usage:
  python3 scripts/create_phydll_asym_models.py <out_dir>
"""

import sys

import torch


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    class Sum18(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1)

    class Expand18(torch.nn.Module):
        def forward(self, x):
            return x.expand(-1, 18)

    torch.jit.script(Sum18()).save(f"{out_dir}/model_18to1.pt")
    torch.jit.script(Expand18()).save(f"{out_dir}/model_1to18.pt")

    # Quick CPU sanity check of both models.
    m1 = torch.jit.load(f"{out_dir}/model_18to1.pt")
    m2 = torch.jit.load(f"{out_dir}/model_1to18.pt")
    x1 = torch.arange(3 * 18, dtype=torch.float32).view(3, 18)
    y1 = m1(x1)
    assert y1.shape == (3,), y1.shape
    assert torch.equal(y1, x1.sum(dim=1))
    x2 = torch.arange(3, dtype=torch.float32).view(3, 1)
    y2 = m2(x2)
    assert y2.shape == (3, 18), y2.shape
    assert torch.equal(y2, x2.expand(-1, 18))
    print(f"wrote {out_dir}/model_18to1.pt and {out_dir}/model_1to18.pt")


if __name__ == "__main__":
    main()
