import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr,
    inp_ptr,
    out_stride_m, out_stride_n,
    inp_stride_m, inp_stride_n,
    len_m,
    len_n: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE_N: tl.constexpr = 128,
):
    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    inp_ptr = inp_ptr.to(tl.pointer_type(dtype))

    pid = tl.program_id(0)
    offset_m = pid * BLOCK_SIZE_M

    rows = offset_m + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < len_m
    cols = tl.arange(0, BLOCK_SIZE_N)

    max_vals = tl.full([BLOCK_SIZE_M], float("-inf"), dtype=tl.float32)
    for block_n in range(0, tl.cdiv(len_n, BLOCK_SIZE_N)):
        col_idx = block_n * BLOCK_SIZE_N + cols
        col_mask = col_idx < len_n
        mask = row_mask[:, None] & col_mask[None, :]
        inp_ptrs = inp_ptr + rows[:, None] * inp_stride_m + col_idx[None, :] * inp_stride_n
        vals = tl.load(inp_ptrs, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(vals, axis=1)
        max_vals = tl.maximum(max_vals, block_max)
    max_vals = tl.where(row_mask, max_vals, 0.0)

    sum_exp = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for block_n in range(0, tl.cdiv(len_n, BLOCK_SIZE_N)):
        col_idx = block_n * BLOCK_SIZE_N + cols
        col_mask = col_idx < len_n
        mask = row_mask[:, None] & col_mask[None, :]
        inp_ptrs = inp_ptr + rows[:, None] * inp_stride_m + col_idx[None, :] * inp_stride_n
        vals = tl.load(inp_ptrs, mask=mask, other=float("-inf")).to(tl.float32)
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)
    sum_exp = tl.where(row_mask, sum_exp, 1.0)

    for block_n in range(0, tl.cdiv(len_n, BLOCK_SIZE_N)):
        col_idx = block_n * BLOCK_SIZE_N + cols
        col_mask = col_idx < len_n
        mask = row_mask[:, None] & col_mask[None, :]
        inp_ptrs = inp_ptr + rows[:, None] * inp_stride_m + col_idx[None, :] * inp_stride_n
        out_ptrs = out_ptr + rows[:, None] * out_stride_m + col_idx[None, :] * out_stride_n
        vals = tl.load(inp_ptrs, mask=mask, other=float("-inf")).to(tl.float32)
        exp_vals = tl.exp(vals - max_vals[:, None])
        softmax_vals = exp_vals / sum_exp[:, None]
        tl.store(out_ptrs, softmax_vals.to(dtype), mask=mask)