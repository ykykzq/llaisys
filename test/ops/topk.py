import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import (
    random_tensor,
    check_equal,
    benchmark,
    llaisys_device,
    llaisys_dtype,
)
from llaisys.libllaisys.triton.setup_kernels import llaisysTopKMask


def torch_topk_mask(out, inp, k):
    vals, idx = torch.topk(inp, k)
    out.fill_(float("-inf"))
    out.scatter_(0, idx, vals)


def test_op_topk(
    length,
    k,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   len {length} k={k} dtype <{dtype_name}>")
    shape = (length,)
    inp, inp_ = random_tensor(shape, dtype_name, device_name)
    out = torch.empty_like(inp)
    out_ = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=0,
    )

    torch_topk_mask(out, inp, k)
    llaisysTopKMask(out_, inp_, k)
    assert check_equal(out_, out, strict=True)

    if profile:
        inp_base = inp.clone()
        api = llaisys.RuntimeAPI(llaisys_device(device_name))
        bytes_ = inp_base.numel() * inp_base.element_size()

        def torch_func():
            tmp = torch.empty_like(inp_base)
            torch_topk_mask(tmp, inp_base, k)

        def llaisys_func():
            api.memcpy_sync(
                inp_.data_ptr(), inp_base.data_ptr(), bytes_, llaisys.MemcpyKind.D2D
            )
            llaisysTopKMask(out_, inp_, k)

        benchmark(torch_func, llaisys_func, device_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    test_lengths = [64, 257]
    test_k = [ 3, 5]
    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.topk_mask on {args.device}")
    for length in test_lengths:
        for k in test_k:
            if k > length:
                continue
            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_topk(length, k, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")

