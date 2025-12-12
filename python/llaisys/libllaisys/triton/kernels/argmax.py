import triton
import triton.language as tl



@triton.jit
def argmax_kernel(
    vals_ptr,
    max_idx_ptr,
    max_val_ptr,
    len: tl.constexpr,
    DTYPE: tl.constexpr,
    IDX_DTYPE: tl.constexpr = tl.int64,
):
    dtype = DTYPE
    idx_dtype = IDX_DTYPE
    vals_ptr = vals_ptr.to(tl.pointer_type(dtype))
    max_idx_ptr = max_idx_ptr.to(tl.pointer_type(idx_dtype))
    max_val_ptr = max_val_ptr.to(tl.pointer_type(dtype))

    vals_ptr = tl.make_block_ptr(
        base=vals_ptr,
        shape=(len,),
        strides=(1,),
        offsets=(0,),
        block_shape=(len,),
        order=(0,),
    )
    max_idx_ptr = tl.make_block_ptr(
        base=max_idx_ptr,
        shape=(1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )
    max_val_ptr = tl.make_block_ptr(
        base=max_val_ptr,
        shape=(1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )

    vals_block = tl.load(vals_ptr, boundary_check=(0,))

    max_val = tl.max(vals_block, axis=0).to(dtype)
    max_idx = tl.argmax(vals_block, axis=0).to(idx_dtype)

    tl.store(max_idx_ptr, max_idx,boundary_check=(0,))
    tl.store(max_val_ptr, max_val,boundary_check=(0,))


