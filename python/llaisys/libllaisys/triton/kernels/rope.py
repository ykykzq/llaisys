import triton
import triton.language as tl
import itertools
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_D": block_size_d},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_d, num_stages, num_warps in itertools.product(
            (32, 64, 128), (32, 64, 128), (2, 3, 4, 5), (4, 8)
        )
    ],
    key=["len_seq", "len_h", "len_d"],
)


@triton.jit
def kernel(
    out_ptr, 
    x_ptr, 
    pos_ids_ptr, 
    theta,
    out_stride_m, out_stride_h, out_stride_d,
    x_stride_m, x_stride_h, x_stride_d,
    pos_ids_stride_m,
    len_seq: tl.constexpr, len_h: tl.constexpr, len_d: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    dtype = DTYPE
    out_ptr = out_ptr.to(tl.pointer_type(dtype))
    x_ptr = x_ptr.to(tl.pointer_type(dtype))
    pos_ids_ptr = pos_ids_ptr.to(tl.pointer_type(tl.int64))

    block_id = tl.program_id(0)
    d_id = tl.program_id(1)
    head_id = tl.program_id(2)

    offset_m = block_id * BLOCK_SIZE_M
    offset_d = d_id * BLOCK_SIZE_D

    x_offset = head_id * x_stride_h
    # 加载 x_a 部分
    x_a_block_ptr = tl.make_block_ptr(
        base=x_ptr + x_offset,
        shape=(len_seq, len_d),
        strides=(x_stride_m, x_stride_d),
        offsets=(offset_m, offset_d),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1,0),
    )
    
    # 加载 x_b 部分
    x_b_block_ptr = tl.make_block_ptr(
        base=x_ptr + x_offset,
        shape=(len_seq, len_d),
        strides=(x_stride_m, x_stride_d),
        offsets=(offset_m, len_d // 2 + offset_d),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1,0),
    )

    out_offset = head_id * out_stride_h
    out_a_block_ptr = tl.make_block_ptr(
        base=out_ptr + out_offset,
        shape=(len_seq, len_d),
        strides=(out_stride_m, out_stride_d),
        offsets=(offset_m, offset_d),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1,0)
    )
    out_b_block_ptr = tl.make_block_ptr(
        base=out_ptr + out_offset,
        shape=(len_seq, len_d),
        strides=(out_stride_m, out_stride_d),
        offsets=(offset_m, len_d // 2 + offset_d),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1,0)
    )

    pos_block_ptr = tl.make_block_ptr(
        base=pos_ids_ptr,
        shape=(len_seq,),
        strides=(pos_ids_stride_m,),
        offsets=(offset_m,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,),
    )

    # 加载 x_a 和 x_b 块
    x_a_block = tl.load(x_a_block_ptr, boundary_check=(0,1)).to(tl.float64)
    x_b_block = tl.load(x_b_block_ptr, boundary_check=(0,1)).to(tl.float64)

    pos_block = tl.load(pos_block_ptr, boundary_check=(0,)).to(tl.float64)

    # j 索引：全局索引，从 offset_d 开始
    j = offset_d + tl.arange(0, BLOCK_SIZE_D)
    theta_f = tl.full((), theta.to(tl.float64), tl.float64)
    # 计算：phi = p / theta^(2j/d)
    inv_freq = tl.exp((-2.0 * j / len_d) * tl.log(theta_f))  # (BLOCK_SIZE_D,)
    freqs = pos_block.to(tl.float64)[:, None] * inv_freq.to(tl.float64)[None, :]  # (BLOCK_SIZE_M, BLOCK_SIZE_D)

    sin = tl.sin(freqs)
    cos = tl.cos(freqs)

    new_x_a_block = (x_a_block * cos - x_b_block * sin).to(tl.float32).to(dtype)
    new_x_b_block = (x_b_block * cos + x_a_block * sin).to(tl.float32).to(dtype)

    tl.store(out_a_block_ptr, new_x_a_block, boundary_check=(0,1))
    tl.store(out_b_block_ptr, new_x_b_block, boundary_check=(0,1))