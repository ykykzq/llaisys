import triton
import triton.language as tl
from .kernels import add
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

    add.kernel[grid](
        x_ptr, y_ptr, output_ptr,
        *x.strides(),
        *y.strides(),
        *output.strides(),
        len_m, len_n,
        DTYPE=triton_dtype,
    )

    return output
