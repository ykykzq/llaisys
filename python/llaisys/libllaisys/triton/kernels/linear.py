import itertools

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, block_size_k, num_stages, num_warps in itertools.product(
            (64, 128), (64, 128), (128,), (2, 3), (4, 8)
        )
    ],
    key=["out_stride_m", "out_stride_n"],
)


@triton.jit
def kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_stride_m,
    out_stride_n,
    x_stride_m,
    x_stride_k,
    weight_stride_n,
    weight_stride_k,
    len_m: tl.constexpr, len_n: tl.constexpr, len_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):

    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    x_ptr = x_ptr.to(tl.pointer_type(dtype))
    weight_ptr = weight_ptr.to(tl.pointer_type(dtype))
    if bias_ptr is not None:
        bias_ptr = bias_ptr.to(tl.pointer_type(dtype)) 

    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)

    offset_m = m_idx * BLOCK_SIZE_M
    offset_n = n_idx * BLOCK_SIZE_N


    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(len_m, len_k),
        strides=(x_stride_m, x_stride_k),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(len_k, len_n),
        strides=(weight_stride_k, weight_stride_n),
        offsets=(0, offset_n),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
        
    for k_idx in range(0, tl.cdiv(len_k, BLOCK_SIZE_K)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        weight_block = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(tl.float32) 
        
        acc += tl.dot(x_block, weight_block, input_precision="ieee").to(tl.float64)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_K))
        weight_block_ptr = tl.advance(weight_block_ptr, (BLOCK_SIZE_K, 0))

    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(
            base=bias_ptr ,
            shape=(len_n,),
            strides=(1,),
            offsets=(offset_n,),
            block_shape=(BLOCK_SIZE_N,),
            order=(0,),
        )
        bias_block = tl.load(bias_block_ptr, boundary_check=(0))
        acc += bias_block.to(tl.float64)[None, :]
    
    # i hate triton......
    if dtype == tl.bfloat16:
        acc = acc.to(tl.float32).to(dtype)
    elif dtype == tl.float16:
        acc = acc.to(tl.float32).to(dtype)
    else:
        acc = acc.to(dtype)

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(len_m, len_n),
        strides=(out_stride_m, out_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(out_block_ptr, acc, boundary_check=(0, 1))