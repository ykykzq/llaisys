import triton
import triton.language as tl


@triton.jit
def topk_mask_kernel(
    out_ptr,
    vals_ptr,
    k: tl.constexpr,
    len: tl.constexpr,
    DTYPE: tl.constexpr,
    IDX_DTYPE: tl.constexpr = tl.int64,
    BLOCK_SIZE: tl.constexpr = 64
):
    dtype = DTYPE
    idx_dtype = IDX_DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    vals_ptr = vals_ptr.to(tl.pointer_type(dtype))

    # 初始化输出为负无穷
    for i in range(0, len, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < len
        tl.store(out_ptr + idx, tl.full([BLOCK_SIZE], float('-inf'), dtype=dtype), mask=mask)
    
    # 使用 out_ptr 来跟踪已选择的索引：如果 out_ptr[i] != -inf，说明已选中
    for k_idx in range(k):
        max_val = tl.full([], float('-inf'), dtype=dtype)
        max_idx = tl.full([], 0, dtype=idx_dtype)
        
        # 遍历所有值找到当前最大值
        for i in range(0, len, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            mask = idx < len
            
            vals = tl.load(vals_ptr + idx, mask=mask, other=float('-inf')).to(dtype)
            outs = tl.load(out_ptr + idx, mask=mask, other=float(0)).to(dtype)
            
            # 只考虑尚未被选中的位置
            not_selected = outs == float('-inf')
            vals = tl.where(not_selected, vals, float('-inf'))
            
            # 找到当前块中的最大值
            block_max_val = tl.max(vals, axis=0)
            block_max_pos = tl.argmax(vals, axis=0)
            block_max_idx = i + block_max_pos
            
            # 更新全局最大值
            condition = block_max_val > max_val
            max_val = tl.where(condition, block_max_val, max_val).to(dtype)
            max_idx = tl.where(condition, block_max_idx, max_idx).to(idx_dtype)
        
        # 将找到的最大值存储到输出
        original_val = tl.load(vals_ptr + max_idx).to(dtype)
        tl.store(out_ptr + max_idx, original_val)
