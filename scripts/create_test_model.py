#!/usr/bin/env python3
"""Creates a minimal TorchScript model for integration testing.

The model computes element-wise addition of the two input features:
  input  shape: [B, 2]
  output shape: [B, 1]   where output[b] = input[b, 0] + input[b, 1]

Usage:
  python3 create_test_model.py <output_path.pt>
"""

import os
import sys
import torch


class AdditionModel(torch.nn.Module):
    """Trivial model whose output is analytically verifiable: y = x0 + x1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:1] + x[:, 1:2]


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "test_addition_model.pt"

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = AdditionModel()
    model.eval()
    scripted = torch.jit.script(model)

    # Quick sanity-check before saving
    dummy = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = scripted(dummy)
    assert out.shape == (2, 1), f"Unexpected output shape: {out.shape}"
    assert abs(out[0, 0].item() - 3.0) < 1e-5, "Sanity check failed"
    assert abs(out[1, 0].item() - 7.0) < 1e-5, "Sanity check failed"

    scripted.save(output_path)
    print(f"Saved test model to: {output_path}")


if __name__ == "__main__":
    main()
