import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 32},
            num_stages=2,
            num_warps=4,
        )
    ],
    key=["output_stride_m"]
)

@triton.jit
def scalar_div_kernel(
    x_ptr, 
    output_ptr, 
    y, 
    x_stride_m,
    output_stride_m,
    len_m,
    BLOCK_SIZE_M: tl.constexpr, 
    DTYPE: tl.constexpr
):
    dtype = DTYPE
    x_ptr = x_ptr.to(tl.pointer_type(dtype))
    output_ptr = output_ptr.to(tl.pointer_type(dtype))
    
    m_idx = tl.program_id(0)

    offset_m = m_idx * BLOCK_SIZE_M
    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(len_m,),
        strides=(x_stride_m,),
        offsets=(offset_m,),
        block_shape=(BLOCK_SIZE_M, ),
        order=(0,),
    )
    
    
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(len_m, ),
        strides=(output_stride_m,),
        offsets=(offset_m,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,),
    )   

    x_block = tl.load(x_block_ptr, boundary_check=(0,))
    output_block = x_block / y
    tl.store(output_block_ptr, output_block.to(dtype), boundary_check=(0,))
    
