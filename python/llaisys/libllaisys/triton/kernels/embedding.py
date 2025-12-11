import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_D": 128},
            num_stages=5,
            num_warps=4,
        )
    ],
    key=["out_stride_n", "out_stride_d"],
)


@triton.jit
def kernel(
    out_ptr,
    index_ptr,
    weight_ptr,
    out_stride_n,
    out_stride_d,
    index_stride_n,
    weight_stride_v,
    weight_stride_d,
    len_n: tl.constexpr,
    len_d: tl.constexpr,
    len_v: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    weight_ptr = weight_ptr.to(tl.pointer_type(dtype))
    index_ptr = index_ptr.to(tl.pointer_type(tl.int64))
    
    n_idx = tl.program_id(0)
    d_idx = tl.program_id(1)
    
    offset_n = n_idx * BLOCK_SIZE_N
    offset_d = d_idx * BLOCK_SIZE_D
    
    out_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=dtype)
    
    for i in range(BLOCK_SIZE_N):
        n_pos = offset_n + i
        if n_pos < len_n:
            idx_val = tl.load(index_ptr + n_pos * index_stride_n)
            
            weight_row_offset = idx_val * weight_stride_v
            weight_row_block_ptr = tl.make_block_ptr(
                base=weight_ptr + weight_row_offset,
                shape=(len_d,),
                strides=(weight_stride_d,),
                offsets=(offset_d,),
                block_shape=(BLOCK_SIZE_D,),
                order=(0,),
            )
            weight_row = tl.load(weight_row_block_ptr, boundary_check=(0))
            
            row_mask = (tl.arange(0, BLOCK_SIZE_N)[:, None] == i)
            out_block = tl.where(row_mask, weight_row[None, :], out_block)
    
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(len_n, len_d),
        strides=(out_stride_n, out_stride_d),
        offsets=(offset_n, offset_d),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(out_block_ptr, out_block, boundary_check=(0, 1))

