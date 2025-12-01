import itertools

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": block_size_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, num_stages, num_warps in itertools.product(
            (32, 64, 128, 256), (32, 64, 128), (2, 3, 4, 5), (4, 8)
        )
    ],
    key=["out_stride_m", "out_stride_n"],
)


@triton.jit
def kernel(
    out_ptr,
    gate_ptr,
    up_ptr,
    out_stride_m,
    out_stride_n,
    gate_stride_m,
    gate_stride_n, 
    up_stride_m,
    up_stride_n,
    len_m, len_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # 将整数指针转换为指定数据类型的指针
    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    gate_ptr = gate_ptr.to(tl.pointer_type(dtype))
    up_ptr = up_ptr.to(tl.pointer_type(dtype))
    
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    offset_m = m_idx * BLOCK_SIZE_M
    offset_n = n_idx * BLOCK_SIZE_N
    
    gate_block_ptr = tl.make_block_ptr(
        base=gate_ptr,
        shape=(len_m, len_n),
        strides=(gate_stride_m, gate_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    up_block_ptr = tl.make_block_ptr(
        base=up_ptr,
        shape=(len_m, len_n),
        strides=(up_stride_m, up_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(len_m, len_n),
        strides=(out_stride_m, out_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    # 加载数据
    gate = tl.load(gate_block_ptr, boundary_check=(0, 1))
    up = tl.load(up_block_ptr, boundary_check=(0, 1))
    
    # 计算 SwiGLU
    gate_fp32 = gate.to(tl.float32)
    exp_neg_gate = tl.exp(-gate_fp32)
    denominator = 1.0 + exp_neg_gate
    # gate / (1 + exp(-gate))
    gate_sigmoid = (gate_fp32 / denominator).to(dtype)
    
    # out = up * (gate / (1 + exp(-gate)))
    out = up * gate_sigmoid
    
    tl.store(out_block_ptr, out.to(dtype), boundary_check=(0, 1))

