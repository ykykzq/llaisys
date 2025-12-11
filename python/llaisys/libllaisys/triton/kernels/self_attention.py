import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32},
            num_stages=3,
            num_warps=8,
        )
    ],
    key=["EMB_DIM"],
)


@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_stride_m,
    q_stride_h,
    q_stride_k,
    k_stride_n,
    k_stride_h,
    k_stride_k,
    v_stride_n,
    v_stride_h,
    v_stride_k,
    o_stride_m,
    o_stride_h,
    o_stride_n,
    scale,
    seq_len_q,
    seq_len_k_v,
    group_size,
    EMB_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    q_ptr = q_ptr.to(tl.pointer_type(DTYPE))
    k_ptr = k_ptr.to(tl.pointer_type(DTYPE))
    v_ptr = v_ptr.to(tl.pointer_type(DTYPE))
    o_ptr = o_ptr.to(tl.pointer_type(DTYPE))

    offs_m_start = block_id * BLOCK_SIZE_M

    # get ptr of q/k/v/o block
    q_offset = head_id * q_stride_h
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(seq_len_q, EMB_DIM),
        strides=(q_stride_m, q_stride_k),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )
    k_offset = (head_id // group_size) * k_stride_h
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(EMB_DIM, seq_len_k_v),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, 0),
        block_shape=(EMB_DIM, BLOCK_SIZE_N),
        order=(0, 1),
    )
    v_offset = (head_id // group_size) * v_stride_h
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_offset,
        shape=(seq_len_k_v, EMB_DIM),
        strides=(v_stride_n, v_stride_k),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, EMB_DIM),
        order=(1, 0),
    )
    o_offset = head_id * o_stride_h
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(seq_len_q, EMB_DIM),
        strides=(o_stride_m, o_stride_n),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )

    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    q = (q * scale).to(q_block_ptr.type.element_ty)

    # initialize
    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=DTYPE)
    l_i = tl.full((BLOCK_SIZE_M,), 1, dtype=DTYPE)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=DTYPE)

    for i in range(0, tl.cdiv(seq_len_k_v, BLOCK_SIZE_N)):

        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        qk = tl.dot(q, k, input_precision="ieee")
        

        m_global = tl.arange(0, BLOCK_SIZE_M) + offs_m_start
        n_global = tl.arange(0, BLOCK_SIZE_N) + i * BLOCK_SIZE_N
        diagonal_offset = seq_len_k_v - seq_len_q
        mask = (n_global[None, :] <= m_global[:, None] + diagonal_offset) & (n_global[None, :] < seq_len_k_v) & (m_global[:, None] + diagonal_offset >= 0)
        qk = tl.where(mask, qk, float("-inf")).to(DTYPE)


        m_ij = tl.maximum(m_i, tl.max(qk, 1)).to(DTYPE)
        p = tl.exp((qk - m_ij[:, None]).to(tl.float32)).to(DTYPE)
        l_ij = tl.sum(p, 1).to(DTYPE)

        v = tl.load(v_block_ptr, boundary_check=(0, 1))
        alpha = tl.exp((m_i - m_ij).to(tl.float32)).to(DTYPE)
        pv = tl.dot(p.to(v_block_ptr.type.element_ty), v, input_precision="ieee").to(DTYPE)
        acc = (acc * alpha[:, None] + pv).to(DTYPE)
        m_i = m_ij.to(DTYPE)
        l_i = (l_i * alpha + l_ij).to(DTYPE)

        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))

    acc /= l_i[:, None]

    tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty), boundary_check=(0, 1))