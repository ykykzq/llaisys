import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark
from llaisys.libllaisys.triton.setup_kernels import llaisysSoftmax


def torch_softmax(out, inp, dim=-1):
    out.copy_(torch.softmax(inp, dim=dim))


def test_op_softmax(
    shape,
    dim=-1,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dim={dim} dtype <{dtype_name}>")
    inp, inp_ = random_tensor(shape, dtype_name, device_name)

    out, out_ = random_tensor(shape, dtype_name, device_name)
    torch_softmax(out, inp, dim)
    llaisysSoftmax(out_, inp_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_softmax(out, inp, dim),
            lambda: llaisysSoftmax(out_, inp_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    testShapes = [
        (2, 3),
        (16, 64),
        (128, 512),
    ]
    testDtypePrec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.softmax on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_softmax(shape, -1, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")

