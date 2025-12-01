import itertools

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m,},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, num_stages, num_warps in itertools.product(
            (32, 64, 128, 256), (2, 3, 4, 5), (4, 8)
        )
    ],
    key=["out_stride_m"],
)


@triton.jit
def kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    eps,
    out_stride_m,
    out_stride_n,
    x_stride_m,
    x_stride_n,
    weight_stride_n,
    len_m: tl.constexpr,
    len_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr,
):  
    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    x_ptr = x_ptr.to(tl.pointer_type(dtype))
    weight_ptr = weight_ptr.to(tl.pointer_type(dtype))

    m_idx = tl.program_id(0)
    offset_m = m_idx * BLOCK_SIZE_M

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(len_m, len_n),
        strides=(x_stride_m, x_stride_n),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, len_n),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(len_n,),
        strides=(weight_stride_n,),
        offsets=(0,),
        block_shape=(len_n,),
        order=(0,),
    )

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(len_m, len_n),
        strides=(out_stride_m, out_stride_n),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, len_n),
        order=(1, 0),
    )

    x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
    weight_block = tl.load(weight_block_ptr, boundary_check=(0,))


    rms = tl.sqrt(tl.sum(x_block * x_block, 1) / len_n + eps)
    rms = rms[:, None] 
    x_block = x_block / rms

    out_block = (x_block * weight_block[None, :]).to(dtype)
    tl.store(out_block_ptr, out_block, boundary_check=(0, 1))