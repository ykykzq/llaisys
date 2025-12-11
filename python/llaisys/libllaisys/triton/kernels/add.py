import itertools
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m,"BLOCK_SIZE_N": block_size_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, num_stages, num_warps in itertools.product(
            (32, 64, 128, 256), (32, 64, 128, 256), (2, 3, 4, 5), (4, 8)
        )
    ],
    key=["output_stride_m", "output_stride_n"]
)

@triton.jit
def kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    x_stride_m, x_stride_n,
    y_stride_m, y_stride_n,
    output_stride_m, output_stride_n,
    len_m, len_n,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE: tl.constexpr
):
    # 将整数指针转换为指定数据类型的指针
    dtype = DTYPE
    x_ptr = x_ptr.to(tl.pointer_type(dtype))
    y_ptr = y_ptr.to(tl.pointer_type(dtype))
    output_ptr = output_ptr.to(tl.pointer_type(dtype))
    
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)

    offset_m = m_idx * BLOCK_SIZE_M
    offset_n = n_idx * BLOCK_SIZE_N
    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(len_m, len_n),
        strides=(x_stride_m, x_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(len_m, len_n),
        strides=(y_stride_m, y_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(len_m, len_n),
        strides=(output_stride_m, output_stride_n),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )   

    x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
    y_block = tl.load(y_block_ptr, boundary_check=(0, 1))
    output_block = x_block + y_block
    tl.store(output_block_ptr, output_block,boundary_check=(0, 1))
    
