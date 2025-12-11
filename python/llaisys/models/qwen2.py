from typing import Sequence, List, Optional
from pathlib import Path
from ctypes import c_void_p, c_char
import ctypes
import json
import math

import numpy as np
import safetensors
import torch

from ..runtime import RuntimeAPI
from ..tensor import Tensor
from ..ops import Ops
from ..libllaisys import DeviceType, DataType, MemcpyKind


def _prod(shape: Sequence[int]) -> int:
    """Compute the number of elements for a given shape."""
    val = 1
    for dim in shape:
        val *= int(dim)
    return val


def _dtype_size(dtype: DataType) -> int:
    mapping = {
        DataType.F16: 2,
        DataType.BF16: 2,
        DataType.F32: 4,
        DataType.F64: 8,
        DataType.I32: 4,
        DataType.I64: 8,
        DataType.U32: 4,
        DataType.U64: 8,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for size lookup: {dtype}")
    return mapping[dtype]


def _np_to_llaisys(dtype) -> DataType:
    dtype = np.dtype(dtype)
    name = dtype.name
    if name == "float16":
        return DataType.F16
    if name == "bfloat16":
        return DataType.BF16
    if name == "float32":
        return DataType.F32
    if name == "float64":
        return DataType.F64
    if name in ("int32", "intc"):
        return DataType.I32
    if name in ("int64", "intp"):
        return DataType.I64
    if name == "uint32":
        return DataType.U32
    if name == "uint64":
        return DataType.U64
    raise ValueError(f"Unsupported numpy dtype: {dtype}")


def _torch_dtype_from_llaisys(dtype: DataType):
    mapping = {
        DataType.F16: torch.float16,
        DataType.BF16: torch.bfloat16,
        DataType.F32: torch.float32,
        DataType.F64: torch.float64,
        DataType.I32: torch.int32,
        DataType.I64: torch.int64,
        DataType.U32: torch.uint32,
        DataType.U64: torch.uint64,
    }
    return mapping.get(dtype, torch.float32)


def _llaisys_dtype_from_torch(dtype: torch.dtype) -> DataType:
    mapping = {
        torch.float16: DataType.F16,
        torch.bfloat16: DataType.BF16,
        torch.float32: DataType.F32,
        torch.float64: DataType.F64,
        torch.int32: DataType.I32,
        torch.int64: DataType.I64,
        torch.uint32: DataType.U32,
        torch.uint64: DataType.U64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


class LayerWeights:
    def __init__(self):
        self.input_norm: Optional[Tensor] = None
        self.post_norm: Optional[Tensor] = None

        self.q_weight: Optional[Tensor] = None
        self.k_weight: Optional[Tensor] = None
        self.v_weight: Optional[Tensor] = None
        self.o_weight: Optional[Tensor] = None

        self.q_bias: Optional[Tensor] = None
        self.k_bias: Optional[Tensor] = None
        self.v_bias: Optional[Tensor] = None
        self.o_bias: Optional[Tensor] = None

        self.gate_weight: Optional[Tensor] = None
        self.up_weight: Optional[Tensor] = None
        self.down_weight: Optional[Tensor] = None

        self.gate_bias: Optional[Tensor] = None
        self.up_bias: Optional[Tensor] = None
        self.down_bias: Optional[Tensor] = None


class KVCache:
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: DataType,
        device: DeviceType,
        device_id: int,
        runtime: RuntimeAPI,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device
        self.device_id = device_id
        self.runtime = runtime
        self.dtype = dtype
        self.current_len = 0

        self._k: List[Tensor] = []
        self._v: List[Tensor] = []
        for _ in range(num_layers):
            self._k.append(
                Tensor(
                    (max_seq_len, num_kv_heads, head_dim),
                    dtype=dtype,
                    device=device,
                    device_id=device_id,
                )
            )
            self._v.append(
                Tensor(
                    (max_seq_len, num_kv_heads, head_dim),
                    dtype=dtype,
                    device=device,
                    device_id=device_id,
                )
            )

    def append(self, layer_idx: int, k_new: Tensor, v_new: Tensor, start_pos: int):
        length = k_new.shape()[0]
        end = start_pos + length
        if end > self.max_seq_len:
            raise ValueError("KV cache overflow: requested position exceeds cache size.")

        dst_k = self._k[layer_idx].slice(0, start_pos, end)
        dst_v = self._v[layer_idx].slice(0, start_pos, end)

        bytes_k = _dtype_size(k_new.dtype()) * _prod(k_new.shape())
        bytes_v = _dtype_size(v_new.dtype()) * _prod(v_new.shape())
        kind = MemcpyKind.D2D if self.device == DeviceType.NVIDIA else MemcpyKind.H2H

        self.runtime.memcpy_sync(dst_k.data_ptr(), k_new.data_ptr(), bytes_k, kind)
        self.runtime.memcpy_sync(dst_v.data_ptr(), v_new.data_ptr(), bytes_v, kind)

    def get(self, layer_idx: int, length: int) -> tuple[Tensor, Tensor]:
        end = min(length, self.max_seq_len)
        return (
            self._k[layer_idx].slice(0, 0, end),
            self._v[layer_idx].slice(0, 0, end),
        )


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        model_path = Path(model_path)

        self.device = device
        self.device_id = device_id
        self.runtime = RuntimeAPI(device)
        try:
            self.runtime.set_device(device_id)
        except Exception:
            pass

        self.config = self._load_config(model_path)
        self.hidden_size = int(self.config.get("hidden_size", 0))
        self.intermediate_size = int(self.config.get("intermediate_size", 0))
        self.num_attention_heads = int(self.config.get("num_attention_heads", 0))
        self.num_key_value_heads = int(
            self.config.get("num_key_value_heads", self.config.get("num_kv_heads", 0))
        )
        self.num_layers_config = int(self.config.get("num_hidden_layers", 0))
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))
        self.rms_norm_eps = float(self.config.get("rms_norm_eps", 1e-6))
        eos_token = self.config.get("eos_token_id")
        if isinstance(eos_token, list):
            eos_token = eos_token[0] if eos_token else None
        self.eos_token_id = eos_token if isinstance(eos_token, int) else None

        self.model_dtype: Optional[DataType] = None
        self.embed_tokens: Optional[Tensor] = None
        self.lm_head_weight: Optional[Tensor] = None
        self.lm_head_bias: Optional[Tensor] = None
        self.final_norm: Optional[Tensor] = None
        self.vocab_size: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.attn_scale: Optional[float] = None
        self.kv_cache: Optional[KVCache] = None

        initial_layers = max(self.num_layers_config, 1)
        self.layers: List[LayerWeights] = [LayerWeights() for _ in range(initial_layers)]

        self._load_weights(model_path)
        self._finalize_model()

    def _load_config(self, model_path: Path) -> dict:
        cfg_path = model_path / "config.json"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _ensure_layer(self, idx: int):
        while len(self.layers) <= idx:
            self.layers.append(LayerWeights())

    def _tensor_from_array(self, array, dtype_override: Optional[DataType] = None) -> Tensor:
        if torch.is_tensor(array):
            t = array.cpu()
            if not t.is_contiguous():
                t = t.contiguous()
            dtype = dtype_override or _llaisys_dtype_from_torch(t.dtype)
            tensor = Tensor(
                tuple(t.shape),
                dtype=dtype,
                device=self.device,
                device_id=self.device_id,
            )
            kind = MemcpyKind.H2D if self.device == DeviceType.NVIDIA else MemcpyKind.H2H
            size_bytes = t.numel() * t.element_size()
            self.runtime.memcpy_sync(tensor.data_ptr(), c_void_p(t.data_ptr()), size_bytes, kind)
            if self.model_dtype is None and dtype in (
                DataType.F16,
                DataType.BF16,
                DataType.F32,
                DataType.F64,
            ):
                self.model_dtype = dtype
            return tensor

        arr = np.asarray(array)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        dtype = dtype_override or _np_to_llaisys(arr.dtype)
        tensor = Tensor(
            arr.shape,
            dtype=dtype,
            device=self.device,
            device_id=self.device_id,
        )
        kind = MemcpyKind.H2D if self.device == DeviceType.NVIDIA else MemcpyKind.H2H
        self.runtime.memcpy_sync(tensor.data_ptr(), c_void_p(arr.ctypes.data), arr.nbytes, kind)
        if self.model_dtype is None and dtype in (
            DataType.F16,
            DataType.BF16,
            DataType.F32,
            DataType.F64,
        ):
            self.model_dtype = dtype
        return tensor

    def _zero_tensor(self, shape: Sequence[int], dtype: DataType) -> Tensor:
        tensor = Tensor(shape, dtype=dtype, device=self.device, device_id=self.device_id)
        size_bytes = _dtype_size(dtype) * _prod(shape)
        zero_buf = (c_char * size_bytes)()
        kind = MemcpyKind.H2D if self.device == DeviceType.NVIDIA else MemcpyKind.H2H
        self.runtime.memcpy_sync(
            tensor.data_ptr(),
            c_void_p(ctypes.addressof(zero_buf)),
            size_bytes,
            kind,
        )
        return tensor

    def _load_weights(self, model_path: Path):
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                array = data_.get_tensor(name_)
                self._assign_weight(name_, array)
    
    def _assign_weight(self, name: str, array):
        if name == "model.embed_tokens.weight":
            self.embed_tokens = self._tensor_from_array(array)
            self.vocab_size = array.shape[0]
            if self.hidden_size == 0:
                self.hidden_size = int(array.shape[1])
            return

        if name == "lm_head.weight":
            self.lm_head_weight = self._tensor_from_array(array)
            self.vocab_size = array.shape[0]
            return

        if name == "model.norm.weight":
            self.final_norm = self._tensor_from_array(array)
            return

        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            return

        layer_idx = int(parts[2])
        self._ensure_layer(layer_idx)
        layer = self.layers[layer_idx]
        sub_name = ".".join(parts[3:])

        if sub_name == "input_layernorm.weight":
            layer.input_norm = self._tensor_from_array(array)
        elif sub_name == "post_attention_layernorm.weight":
            layer.post_norm = self._tensor_from_array(array)
        elif sub_name == "self_attn.q_proj.weight":
            layer.q_weight = self._tensor_from_array(array)
        elif sub_name == "self_attn.q_proj.bias":
            layer.q_bias = self._tensor_from_array(array)
        elif sub_name == "self_attn.k_proj.weight":
            layer.k_weight = self._tensor_from_array(array)
        elif sub_name == "self_attn.k_proj.bias":
            layer.k_bias = self._tensor_from_array(array)
        elif sub_name == "self_attn.v_proj.weight":
            layer.v_weight = self._tensor_from_array(array)
        elif sub_name == "self_attn.v_proj.bias":
            layer.v_bias = self._tensor_from_array(array)
        elif sub_name == "self_attn.o_proj.weight":
            layer.o_weight = self._tensor_from_array(array)
        elif sub_name == "mlp.gate_proj.weight":
            layer.gate_weight = self._tensor_from_array(array)
        elif sub_name == "mlp.up_proj.weight":
            layer.up_weight = self._tensor_from_array(array)
        elif sub_name == "mlp.down_proj.weight":
            layer.down_weight = self._tensor_from_array(array)

    def _finalize_model(self):
        self.num_layers = len(self.layers)
        if self.hidden_size == 0 and self.embed_tokens is not None:
            self.hidden_size = self.embed_tokens.shape()[1]
        if self.num_attention_heads == 0:
            raise ValueError("num_attention_heads is missing in config.")
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads

        if self.model_dtype is None:
            self.model_dtype = DataType.F16

        if self.intermediate_size == 0:
            for layer in self.layers:
                if layer.gate_weight is not None:
                    self.intermediate_size = layer.gate_weight.shape()[0]
                    break

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        if self.final_norm is None:
            raise ValueError("Final RMSNorm weight (model.norm.weight) not found.")
        if self.embed_tokens is None:
            raise ValueError("Embedding weight (model.embed_tokens.weight) not found.")
        if self.vocab_size is None:
            self.vocab_size = self.embed_tokens.shape()[0]

        for layer in self.layers:
            for required in [
                ("q_weight", layer.q_weight),
                ("k_weight", layer.k_weight),
                ("v_weight", layer.v_weight),
                ("o_weight", layer.o_weight),
                ("gate_weight", layer.gate_weight),
                ("up_weight", layer.up_weight),
                ("down_weight", layer.down_weight),
                ("input_norm", layer.input_norm),
                ("post_norm", layer.post_norm),
            ]:
                if required[1] is None:
                    raise ValueError(f"Missing parameter {required[0]} for Qwen2.")

            if layer.q_bias is None:
                layer.q_bias = self._zero_tensor((self.hidden_size,), self.model_dtype)
            if layer.k_bias is None:
                layer.k_bias = self._zero_tensor((self.hidden_size,), self.model_dtype)
            if layer.v_bias is None:
                layer.v_bias = self._zero_tensor((self.hidden_size,), self.model_dtype)
            if layer.o_bias is None:
                layer.o_bias = self._zero_tensor((self.hidden_size,), self.model_dtype)

            if layer.gate_bias is None:
                layer.gate_bias = self._zero_tensor((self.intermediate_size,), self.model_dtype)
            if layer.up_bias is None:
                layer.up_bias = self._zero_tensor((self.intermediate_size,), self.model_dtype)
            if layer.down_bias is None:
                layer.down_bias = self._zero_tensor((self.hidden_size,), self.model_dtype)

        if self.lm_head_weight is None:
            self.lm_head_weight = self.embed_tokens
        if self.lm_head_bias is None:
            self.lm_head_bias = self._zero_tensor((self.vocab_size,), self.model_dtype)

    def _init_cache(self, total_len: int):
        self.kv_cache = KVCache(
            num_layers=self.num_layers,
            max_seq_len=total_len,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
            runtime=self.runtime,
        )

    def _build_pos_ids(self, start: int, length: int) -> Tensor:
        pos = np.arange(start, start + length, dtype=np.int64)
        return self._tensor_from_array(pos, dtype_override=DataType.I64)

    def _embed(self, token_ids: Sequence[int]) -> Tensor:
        token_array = np.array(token_ids, dtype=np.int64)
        idx_tensor = self._tensor_from_array(token_array, dtype_override=DataType.I64)
        out = Tensor(
            (len(token_ids), self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.embedding(out, idx_tensor, self.embed_tokens)
        return out

    def _decoder_layer(
        self,
        layer_idx: int,
        hidden_states: Tensor,
        pos_ids: Tensor,
        start_pos: int,
    ) -> Tensor:
        layer = self.layers[layer_idx]
        seq_len = hidden_states.shape()[0]

        normed = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.rms_norm(normed, hidden_states, layer.input_norm, self.rms_norm_eps)

        q_proj = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        k_proj = Tensor(
            (seq_len, self.num_key_value_heads * self.head_dim),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        v_proj = Tensor(
            (seq_len, self.num_key_value_heads * self.head_dim),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.linear(q_proj, normed, layer.q_weight, layer.q_bias)
        Ops.linear(k_proj, normed, layer.k_weight, layer.k_bias)
        Ops.linear(v_proj, normed, layer.v_weight, layer.v_bias)

        q = q_proj.view(seq_len, self.num_attention_heads, self.head_dim)
        k = k_proj.view(seq_len, self.num_key_value_heads, self.head_dim)
        v = v_proj.view(seq_len, self.num_key_value_heads, self.head_dim)
        
        q_rope = Tensor(
            q.shape(),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        k_rope = Tensor(
            k.shape(),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.rope(q_rope, q, pos_ids, self.rope_theta)
        Ops.rope(k_rope, k, pos_ids, self.rope_theta)
        
        self.kv_cache.append(layer_idx, k_rope, v, start_pos)
        total_len = start_pos + seq_len
        k_full, v_full = self.kv_cache.get(layer_idx, total_len)

        attn_out = Tensor(
            (seq_len, self.num_attention_heads, self.head_dim),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.self_attention(attn_out, q_rope, k_full, v_full, self.attn_scale)
        attn_flat = attn_out.view(seq_len, self.hidden_size)

        attn_proj = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.linear(attn_proj, attn_flat, layer.o_weight, layer.o_bias)
        
        attn_res = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.add(attn_res, hidden_states, attn_proj)

        mlp_norm = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.rms_norm(mlp_norm, attn_res, layer.post_norm, self.rms_norm_eps)
        
        gate = Tensor(
            (seq_len, self.intermediate_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        up = Tensor(
            (seq_len, self.intermediate_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.linear(gate, mlp_norm, layer.gate_weight, layer.gate_bias)
        Ops.linear(up, mlp_norm, layer.up_weight, layer.up_bias)

        swiglu_out = Tensor(
            (seq_len, self.intermediate_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.swiglu(swiglu_out, gate, up)
        
        down = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.linear(down, swiglu_out, layer.down_weight, layer.down_bias)
        
        output = Tensor(
            (seq_len, self.hidden_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.add(output, attn_res, down)
        return output

    def _run_decoder(self, hidden_states: Tensor, start_pos: int) -> Tensor:
        if self.kv_cache is None:
            raise RuntimeError("KV cache is not initialized.")
        seq_len = hidden_states.shape()[0]
        if start_pos + seq_len > self.kv_cache.max_seq_len:
            raise ValueError("Requested sequence exceeds allocated KV cache length.")

        pos_ids = self._build_pos_ids(start_pos, seq_len)
        for idx in range(self.num_layers):
            hidden_states = self._decoder_layer(idx, hidden_states, pos_ids, start_pos)
        self.kv_cache.current_len = start_pos + seq_len
        return hidden_states

    def _compute_logits(self, hidden: Tensor) -> Tensor:
        normed = Tensor(
            hidden.shape(),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.rms_norm(normed, hidden, self.final_norm, self.rms_norm_eps)
        logits = Tensor(
            (hidden.shape()[0], self.vocab_size),
            dtype=self.model_dtype,
            device=self.device,
            device_id=self.device_id,
        )
        Ops.linear(logits, normed, self.lm_head_weight, self.lm_head_bias)
        return logits

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        shape = tensor.shape()
        dtype = tensor.dtype()
        torch_dtype = _torch_dtype_from_llaisys(dtype)
        buf = torch.empty(shape, dtype=torch_dtype, device="cpu")
        kind = MemcpyKind.D2H if self.device == DeviceType.NVIDIA else MemcpyKind.H2H
        size_bytes = _dtype_size(dtype) * _prod(shape)
        self.runtime.memcpy_sync(c_void_p(buf.data_ptr()), tensor.data_ptr(), size_bytes, kind)
        if torch_dtype == torch.bfloat16:
            # numpy does not support bfloat16; cast to float32 for sampling.
            return buf.float().numpy()
        return buf.numpy()

    def _sample_token(
        self, logits: Tensor, top_k: int, top_p: float, temperature: float
    ) -> int:
        scores = self._tensor_to_numpy(logits)[-1].astype(np.float64)
        if temperature <= 0:
            temperature = 1.0
        if temperature != 1.0:
            scores = scores / temperature

        vocab = scores.shape[-1]
        top_k = max(1, min(top_k, vocab))

        if top_k == 1:
            return int(np.argmax(scores))

        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx_scores = scores[idx]
        order = np.argsort(-idx_scores)
        idx = idx[order]
        idx_scores = idx_scores[order]

        max_score = idx_scores[0]
        probs = np.exp(idx_scores - max_score)
        probs = probs / probs.sum()

        if top_p < 1.0:
            cumulative = np.cumsum(probs)
            keep = cumulative <= top_p
            if not keep.any():
                keep[0] = True
            idx = idx[keep]
            probs = probs[keep]
            probs = probs / probs.sum()

        choice = np.random.choice(idx, p=probs)
        return int(choice)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        tokens = list(inputs)
        if max_new_tokens is None:
            max_new_tokens = 128
        
        total_len = len(tokens) + max_new_tokens
        self._init_cache(total_len)
        
        hidden = self._embed(tokens)
        hidden = self._run_decoder(hidden, start_pos=0)
        last_hidden = hidden.slice(0, hidden.shape()[0] - 1, hidden.shape()[0])

        generated = 0
        while generated < max_new_tokens:
            logits = self._compute_logits(last_hidden)
            next_token = self._sample_token(logits, top_k, top_p, temperature)
            tokens.append(next_token)
            generated += 1

            if self.eos_token_id is not None and next_token == self.eos_token_id:
                break
            if generated >= max_new_tokens:
                break

            new_hidden = self._embed([next_token])
            new_hidden = self._run_decoder(new_hidden, start_pos=self.kv_cache.current_len)
            last_hidden = new_hidden

        return tokens
