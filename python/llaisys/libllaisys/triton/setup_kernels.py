import math
import triton
import triton.language as tl
from .kernels import add
from .kernels import argmax
from .kernels import self_attention
from .kernels import swiglu
from .kernels import linear
from .kernels import embedding
from .kernels import rms_norm
from .kernels import rope
from .kernels import scalar_div
from ...libllaisys.llaisys_types import DataType


def _llaisys_dtype_to_triton_dtype(llaisys_dtype: DataType):
    mapping = {
        DataType.F16: tl.float16,
        DataType.BF16: tl.bfloat16,
        DataType.F32: tl.float32,
        DataType.F64: tl.float64,
        DataType.I8: tl.int8,
        DataType.I16: tl.int16,
        DataType.I32: tl.int32,
        DataType.I64: tl.int64,
        DataType.U8: tl.uint8,
        DataType.U16: tl.uint16,
        DataType.U32: tl.uint32,
        DataType.U64: tl.uint64,
    }
    if llaisys_dtype not in mapping:
        raise ValueError(f"Unsupported dtype for triton kernel: {llaisys_dtype}")
    return mapping[llaisys_dtype]


def llaisysAdd(x, y, output):
    
    assert x.shape() == y.shape() == output.shape()
    assert len(x.shape()) == 2
    
    # 检查所有输入的数据类型是否一致
    x_dtype = x.dtype()
    y_dtype = y.dtype()
    output_dtype = output.dtype()
    assert x_dtype == y_dtype == output_dtype, \
        f"All tensors must have the same dtype, got x={x_dtype}, y={y_dtype}, output={output_dtype}"

    len_m = x.shape()[0]
    len_n = x.shape()[1]
    
    # 获取数据指针
    x_ptr = x.data_ptr()
    y_ptr = y.data_ptr()
    output_ptr = output.data_ptr()
    
    # 将 llaisys DataType 转换为 triton dtype
    triton_dtype = _llaisys_dtype_to_triton_dtype(x_dtype)
    
    def grid(meta):
        return (
            triton.cdiv(len_m, meta["BLOCK_SIZE_M"]),
            triton.cdiv(len_n, meta["BLOCK_SIZE_N"]),
        )

    add.add_kernel[grid](
        x_ptr, y_ptr, output_ptr,
        *x.strides(),
        *y.strides(),
        *output.strides(),
        len_m, len_n,
        DTYPE=triton_dtype,
    )

    return output

def llaisysScalarDiv(x, y, output):
    
    assert x.shape() == output.shape()
    assert len(x.shape()) == 1
    
    # 检查所有输入的数据类型是否一致
    x_dtype = x.dtype()
    output_dtype = output.dtype()
    assert x_dtype == output_dtype, \
        f"All tensors must have the same dtype, got x={x_dtype}, output={output_dtype}"

    len_m = x.shape()[0]
    
    # 获取数据指针
    x_ptr = x.data_ptr()
    output_ptr = output.data_ptr()
    
    # 将 llaisys DataType 转换为 triton dtype
    triton_dtype = _llaisys_dtype_to_triton_dtype(x_dtype)
    
    def grid(meta):
        return (
            triton.cdiv(len_m, meta["BLOCK_SIZE_M"]),
        )

    scalar_div.scalar_div_kernel[grid](
        x_ptr, output_ptr, y,
        *x.strides(),
        *output.strides(),
        len_m,
        DTYPE=triton_dtype,
    )

    return output


# 目前只支持1D tensor
def llaisysArgmax(vals, max_idx, max_val):
    assert len(vals.shape()) == len(max_idx.shape()) == len(max_val.shape()) == 1

    len_vals = vals.shape()[0]
    val_dtype = _llaisys_dtype_to_triton_dtype(vals.dtype())
    idx_dtype = _llaisys_dtype_to_triton_dtype(max_idx.dtype())

    def grid(meta):
        return (1, 1)

    vals_ptr = vals.data_ptr()
    max_idx_ptr = max_idx.data_ptr()
    max_val_ptr = max_val.data_ptr()

    argmax.argmax_kernel[grid](vals_ptr, max_idx_ptr, max_val_ptr, len_vals, DTYPE=val_dtype, IDX_DTYPE=idx_dtype)
    return max_idx, max_val


# 暂时不考虑batch维度
# require embed_dim >= 16
# causal + group query attention
def llaisysSelfAttention(o, q, k, v, scale=None):
    seq_len_q, num_q_heads, emb_dim = q.shape()
    seq_len_k_v, num_kv_heads, _ = k.shape()

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)

    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads

    o_ptr = o.data_ptr()
    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()
    v_ptr = v.data_ptr()
    dtype = _llaisys_dtype_to_triton_dtype(o.dtype())

    def grid(meta):
        return (
            triton.cdiv(seq_len_q, meta["BLOCK_SIZE_M"]),
            num_q_heads,
        )

    self_attention.self_attention_kernel[grid](
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        *q.strides(),
        *k.strides(),
        *v.strides(),
        *o.strides(),
        scale=scale,
        seq_len_q=seq_len_q,
        seq_len_k_v=seq_len_k_v,
        group_size=group_size,
        EMB_DIM=emb_dim,
        DTYPE=dtype,
    )

    return o

def llaisysSwiGLU(out, gate, up):
    seqlen, intermediate_size = out.shape()
    out_ptr = out.data_ptr()
    gate_ptr = gate.data_ptr()
    up_ptr = up.data_ptr()
    dtype = _llaisys_dtype_to_triton_dtype(out.dtype())

    def grid(meta):
        return (
            triton.cdiv(seqlen, meta["BLOCK_SIZE_M"]),
            triton.cdiv(intermediate_size, meta["BLOCK_SIZE_N"]),
        )

    swiglu.swiglu_kernel[grid](
        out_ptr, 
        gate_ptr, 
        up_ptr, 
        *out.strides(), 
        *gate.strides(), 
        *up.strides(), 
        seqlen, 
        intermediate_size, 
        DTYPE=dtype,
    )
    return out

def llaisysLinear(out, x, weight, bias):
    len_m, len_n = out.shape()
    _, len_k = weight.shape()

    out_ptr = out.data_ptr()
    x_ptr = x.data_ptr()
    weight_ptr = weight.data_ptr() 
    if bias is not None:
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = None
    dtype = _llaisys_dtype_to_triton_dtype(out.dtype())

    def grid(meta):
        return (
            triton.cdiv(len_m, meta["BLOCK_SIZE_M"]),
            triton.cdiv(len_n, meta["BLOCK_SIZE_N"]),
        )


    linear.linear_kernel[grid](
        out_ptr, x_ptr, 
        weight_ptr, bias_ptr, 
        *out.strides(), *x.strides(), *weight.strides(), 
        len_m, len_n, len_k, 
        DTYPE=dtype,
    )
    return out

# the ligality of the index element should be ensured by the user
def llaisysEmbedding(out, index, weight):
    assert index.dtype() == DataType.I64

    len_n, len_d = out.shape()
    len_v, _ = weight.shape()

    out_ptr = out.data_ptr()
    index_ptr = index.data_ptr()
    weight_ptr = weight.data_ptr()
    dtype = _llaisys_dtype_to_triton_dtype(out.dtype())

    def grid(meta):
        return (
        triton.cdiv(len_n, meta["BLOCK_SIZE_N"]),
        triton.cdiv(len_d, meta["BLOCK_SIZE_D"]),
    )

    embedding.embedding_kernel[grid](
        out_ptr, index_ptr, weight_ptr,
        *out.strides(), *index.strides(), *weight.strides(),
        len_n, len_d, len_v,
        DTYPE=dtype,
    )
    return out

def llaisysRMSNorm(out, x, weight, eps):
    len_m, len_n = out.shape()

    out_ptr = out.data_ptr()
    x_ptr = x.data_ptr()
    weight_ptr = weight.data_ptr()
    dtype = _llaisys_dtype_to_triton_dtype(out.dtype())

    def grid(meta):
        return (
            triton.cdiv(len_m, meta["BLOCK_SIZE_M"]),
        )

    rms_norm.rms_norm_kernel[grid](
        out_ptr, x_ptr, weight_ptr, eps,
        *out.strides(), *x.strides(), *weight.strides(),
        len_m, len_n, DTYPE=dtype,
        )

def llaisysROPE(out, x, pos_ids, theta):
    assert pos_ids.dtype() == DataType.I64
    assert out.shape() == x.shape()
    assert len(out.shape()) == 3
    assert x.shape()[2] % 2 == 0 # embed_dim
    len_seq, len_h, len_d = out.shape()
    assert pos_ids.shape()[0] == len_seq

    out_ptr = out.data_ptr()
    x_ptr = x.data_ptr()
    pos_ids_ptr = pos_ids.data_ptr()
    dtype = _llaisys_dtype_to_triton_dtype(out.dtype())

    # 在 seq_len, d 维度和 head 上并行
    def grid(meta):
        return(
            triton.cdiv(len_seq, meta["BLOCK_SIZE_M"]),
            triton.cdiv(len_d // 2, meta["BLOCK_SIZE_D"]),
            len_h,
        )

    rope.rope_kernel[grid](
        out_ptr, x_ptr, pos_ids_ptr, theta,
        *out.strides(), *x.strides(), *pos_ids.strides(),
        len_seq, len_h, len_d,
        DTYPE=dtype,
    )
    return out