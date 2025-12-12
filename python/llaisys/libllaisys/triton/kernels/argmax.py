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
    BLOCK_SIZE: tl.constexpr = 128
):
    dtype = DTYPE
    idx_dtype = IDX_DTYPE
    vals_ptr = vals_ptr.to(tl.pointer_type(dtype))
    max_idx_ptr = max_idx_ptr.to(tl.pointer_type(idx_dtype))
    max_val_ptr = max_val_ptr.to(tl.pointer_type(dtype))

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    max_val = tl.full([], float('-inf'), dtype=dtype)
    max_idx = tl.full([], 0, dtype=idx_dtype)
    
    for i in range(0, len, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < len
        
        vals = tl.load(vals_ptr + idx, mask=mask, other=float('-inf'))
        
        block_max_val = tl.max(vals, axis=0).to(dtype)
        block_max_idx = tl.argmax(vals, axis=0).to(idx_dtype)
        
        condition = block_max_val > max_val
        max_val = tl.where(condition, block_max_val, max_val).to(dtype)
        max_idx = tl.where(condition, i + block_max_idx, max_idx).to(idx_dtype)
    
    tl.store(max_idx_ptr, max_idx)
    tl.store(max_val_ptr, max_val)


