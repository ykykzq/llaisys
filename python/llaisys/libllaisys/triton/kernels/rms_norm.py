import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128},
            num_stages=2,
            num_warps=4,
        )
    ],
    key=["out_stride_m", "out_stride_n"],
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
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE: tl.constexpr,
):  
    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    x_ptr = x_ptr.to(tl.pointer_type(dtype))
    weight_ptr = weight_ptr.to(tl.pointer_type(dtype))

    m_idx = tl.program_id(0)
    offset_m = m_idx * BLOCK_SIZE_M

    sum_sq = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(len_m, len_n),
        strides=(x_stride_m, x_stride_n),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    for n_idx in range(0, tl.cdiv(len_n, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
        sum_sq += tl.sum(x_block * x_block, axis=1)
        
        if n_idx < tl.cdiv(len_n, BLOCK_SIZE_N) - 1:
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_N))
    
    rms = tl.sqrt(sum_sq / len_n + eps)
    rms = rms[:, None]

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(len_m, len_n),
        strides=(x_stride_m, x_stride_n),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(len_n,),
        strides=(weight_stride_n,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(len_m, len_n),
        strides=(out_stride_m, out_stride_n),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    for n_idx in range(0, tl.cdiv(len_n, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
        weight_block = tl.load(weight_block_ptr, boundary_check=(0,))

        x_normalized = x_block / rms
        out_block = (x_normalized * weight_block[None, :]).to(dtype)
        tl.store(out_block_ptr, out_block, boundary_check=(0, 1))
        
        if n_idx < tl.cdiv(len_n, BLOCK_SIZE_N) - 1:
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_N))
            weight_block_ptr = tl.advance(weight_block_ptr, (BLOCK_SIZE_N,))
            out_block_ptr = tl.advance(out_block_ptr, (0, BLOCK_SIZE_N))