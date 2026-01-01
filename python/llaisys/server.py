import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .models.qwen2 import Qwen2
from .libllaisys import DeviceType

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "llaisys"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class LlaisysServer:

    def __init__(self, model_path: str, device: str = "cpu", device_id: int = 0):
        self.model_path = model_path
        self.device_type = DeviceType.NVIDIA if device.lower() in ("gpu", "nvidia") else DeviceType.CPU
        self.device_id = device_id
        self.model: Optional[Qwen2] = None
        self.tokenizer = None
        self.model_name = "qwen2"
        
    def load_model(self):
        print(f"正在加载模型: {self.model_path}")
        print(f"设备: {self.device_type}, 设备ID: {self.device_id}")
        
        # 加载分词器
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            print("分词器加载完成")
        except Exception as e:
            raise RuntimeError(f"加载分词器失败: {e}")
        
        # 加载模型
        try:
            self.model = Qwen2(
                self.model_path,
                device=self.device_type,
                device_id=self.device_id
            )
            print("模型加载完成")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def format_chat_messages(self, messages: List[ChatMessage]) -> str:
        # 使用 Qwen2 的聊天模板
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 回退到简单格式
            text = ""
            for msg in messages:
                if msg.role == "system":
                    text += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
                elif msg.role == "user":
                    text += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
                elif msg.role == "assistant":
                    text += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
        
        return text
    
    def generate(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
    ) -> tuple[str, int, int]:
        prompt = self.format_chat_messages(messages)
        input_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(input_ids)
        
        output_ids = self.model.generate(
            inputs=input_ids,
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        
        # 提取生成的新 token
        new_tokens = output_ids[prompt_tokens:]
        completion_tokens = len(new_tokens)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text, prompt_tokens, completion_tokens
    
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        prompt = self.format_chat_messages(messages)
        input_ids = self.tokenizer.encode(prompt)
        
        # 初始化 KV cache
        total_len = len(input_ids) + max_tokens
        self.model._init_cache(total_len)
        
        # 处理 prompt
        hidden = self.model._embed(input_ids)
        hidden = self.model._run_decoder(hidden, start_pos=0)
        last_hidden = hidden.slice(0, hidden.shape()[0] - 1, hidden.shape()[0])
        
        generated = 0
        tokens = list(input_ids)
        
        while generated < max_tokens:
            logits = self.model._compute_logits(last_hidden)
            next_token = self.model._sample_token(logits, top_k, top_p, temperature)
            tokens.append(next_token)
            generated += 1
            
            # 检查是否遇到结束符
            is_eos = (self.model.eos_token_id is not None and 
                      next_token == self.model.eos_token_id)
            
            # 解码单个 token
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            
            yield token_text, is_eos
            
            if is_eos or generated >= max_tokens:
                break
            
            # 准备下一轮生成
            new_hidden = self.model._embed([next_token])
            new_hidden = self.model._run_decoder(
                new_hidden, 
                start_pos=self.model.kv_cache.current_len
            )
            last_hidden = new_hidden
            
            # 让出控制权以支持异步
            await asyncio.sleep(0)


app = FastAPI(
    title="Llaisys OpenAI-Compatible API",
    description="OpenAI Chat-Completion API Server",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务器实例
server: Optional[LlaisysServer] = None


def get_server() -> LlaisysServer:
    if server is None:
        raise HTTPException(status_code=500, detail="服务器未初始化")
    return server


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    srv = get_server()
    return ModelListResponse(
        data=[
            ModelInfo(
                id=srv.model_name,
                created=int(time.time()),
            )
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    srv = get_server()
    if model_id != srv.model_name:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
    return ModelInfo(
        id=srv.model_name,
        created=int(time.time()),
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    srv = get_server()
    
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    
    if request.stream:
        # 流式响应
        async def generate_stream_response() -> AsyncGenerator[str, None]:
            try:
                async for token_text, is_eos in srv.generate_stream(
                    messages=request.messages,
                    max_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.8,
                    top_p=request.top_p or 0.8,
                    top_k=request.top_k or 50,
                ):
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": token_text} if not is_eos else {},
                                finish_reason="stop" if is_eos else None,
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_data = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    else:
        # 非流式响应
        try:
            generated_text, prompt_tokens, completion_tokens = srv.generate(
                messages=request.messages,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.8,
                top_p=request.top_p or 0.8,
                top_k=request.top_k or 50,
            )
            
            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def create_app(model_path: str, device: str = "cpu", device_id: int = 0) -> FastAPI:
    global server
    server = LlaisysServer(model_path, device, device_id)
    server.load_model()
    return app


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cpu",
    device_id: int = 0,
):
    create_app(model_path, device, device_id)
    uvicorn.run(app, host=host, port=port)

