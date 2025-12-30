import os
from ctypes import c_float, c_int, c_void_p

import torch

from .libllaisys import LIB_LLAISYS
from .libllaisys import NINETOOTHED
from .libllaisys import DeviceType, DataType, MemcpyKind
from .tensor import Tensor
from .runtime import RuntimeAPI


if os.environ.get("ENABLE_NT") == "True":
    _CURRENT_LIB = NINETOOTHED
    # print("LLAISYS Ops: Using NINETOOTHED for accelerated kernels.")
else:
    _CURRENT_LIB = LIB_LLAISYS
    # print("LLAISYS Ops: Using default LIB_LLAISYS kernels.")

try:
    from .libllaisys.triton import (
        llaisysAdd as triton_add,
        llaisysArgmax as triton_argmax,
        llaisysScalarDiv as triton_scalar_div,
        llaisysEmbedding as triton_embedding,
        llaisysLinear as triton_linear,
        llaisysRMSNorm as triton_rms_norm,
        llaisysROPE as triton_rope,
        llaisysSelfAttention as triton_self_attention,
        llaisysSwiGLU as triton_swiglu,
        llaisysSoftmax as triton_softmax,
        llaisysTopKMask as triton_topk_mask,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _use_triton(tensor: Tensor) -> bool:
    """Check if we should use Triton kernels for this tensor."""
    return TRITON_AVAILABLE and tensor.device_type() == DeviceType.NVIDIA


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        if _use_triton(c):
            triton_add(a, b, c)
        else:
            _CURRENT_LIB.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def scalar_div(c: Tensor, a: Tensor, b: float):
        if _use_triton(c):
            triton_scalar_div(a, b, c)
        else:
            raise NotImplementedError("Scalar division is not implemented for this device.")

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        if _use_triton(vals):
            triton_argmax(vals, max_idx, max_val)
        else:
            _CURRENT_LIB.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        if _use_triton(out):
            triton_embedding(out, index, weight)
        else:
            _CURRENT_LIB.llaisysEmbedding(
                out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
            )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        if _use_triton(out):
            triton_linear(out, inp, weight, bias)
        else:
            _CURRENT_LIB.llaisysLinear(
                out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), bias.lib_tensor()
            )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        _CURRENT_LIB.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        if _use_triton(out):
            triton_rms_norm(out, inp, weight, eps)
        else:
            _CURRENT_LIB.llaisysRmsNorm(
                out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
            )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        if _use_triton(out):
            triton_rope(out, inp, pos_ids, theta)
        else:
            _CURRENT_LIB.llaisysROPE(
                out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
            )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float):
        if _use_triton(attn_val):
            triton_self_attention(attn_val, q, k, v, scale)
        else:
            _CURRENT_LIB.llaisysSelfAttention(
                attn_val.lib_tensor(),
                q.lib_tensor(),
                k.lib_tensor(),
                v.lib_tensor(),
                c_float(scale),
            )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        if _use_triton(out):
            triton_swiglu(out, gate, up)
        else:
            _CURRENT_LIB.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())

    @staticmethod
    def softmax(out: Tensor, inp: Tensor):
        if _use_triton(out):
            triton_softmax(out, inp)
        else:
            raise NotImplementedError("Softmax is not implemented for this device.")

    @staticmethod
    def topk_mask(out: Tensor, inp: Tensor, k: int):
        if _use_triton(out):
            triton_topk_mask(out, inp, k)
        else:
            raise NotImplementedError("TopK mask is not implemented for this device.")