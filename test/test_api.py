#!/usr/bin/env python3
"""
Llaisys API 测试客户端

测试 OpenAI-Compatible API 是否正常工作
"""

import argparse
import json
import httpx
import sys


def test_health(base_url: str):
    """测试健康检查端点"""
    print("\n[1/4] 测试健康检查 (/health)...")
    try:
        response = httpx.get(f"{base_url}/health", timeout=10.0)
        response.raise_for_status()
        print(f"  ✓ 状态: {response.json()}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False
    return True


def test_list_models(base_url: str):
    """测试列出模型端点"""
    print("\n[2/4] 测试列出模型 (/v1/models)...")
    try:
        response = httpx.get(f"{base_url}/v1/models", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        print(f"  ✓ 可用模型: {[m['id'] for m in data['data']]}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False
    return True


def test_chat_completion(base_url: str, prompt: str):
    """测试聊天补全 (非流式)"""
    print("\n[3/4] 测试聊天补全 (非流式)...")
    try:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "qwen2",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.8,
                "stream": False
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        usage = data["usage"]
        
        print(f"  ✓ 回复: {content[:100]}..." if len(content) > 100 else f"  ✓ 回复: {content}")
        print(f"  ✓ Token 使用: 输入={usage['prompt_tokens']}, 输出={usage['completion_tokens']}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False
    return True


def test_chat_completion_stream(base_url: str, prompt: str):
    """测试聊天补全 (流式)"""
    print("\n[4/4] 测试聊天补全 (流式)...")
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": "qwen2",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.8,
                "stream": True
            },
            timeout=120.0
        ) as response:
            response.raise_for_status()
            
            print("  ✓ 流式回复: ", end="", flush=True)
            full_content = ""
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data["choices"][0]["delta"].get("content"):
                            token = data["choices"][0]["delta"]["content"]
                            print(token, end="", flush=True)
                            full_content += token
                    except json.JSONDecodeError:
                        pass
            
            print()  # 换行
            print(f"  ✓ 完整回复长度: {len(full_content)} 字符")
            
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="测试 Llaisys OpenAI-Compatible API")
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="http://localhost:8000",
        help="API 服务器地址 (默认: http://localhost:8000)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="你好，请介绍一下你自己。",
        help="测试提示词"
    )
    
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    
    print("=" * 60)
    print("  Llaisys API 测试客户端")
    print("=" * 60)
    print(f"  服务器地址: {base_url}")
    print(f"  测试提示词: {args.prompt}")
    print("=" * 60)
    
    results = []
    results.append(test_health(base_url))
    results.append(test_list_models(base_url))
    results.append(test_chat_completion(base_url, args.prompt))
    results.append(test_chat_completion_stream(base_url, args.prompt))
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"  测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

