import sys
import os

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
python_dir = os.path.join(project_root, "python")
sys.path.insert(0, python_dir)

from llaisys.server import run_server
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Llaisys OpenAI-Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str, 
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="服务器监听地址 "
    )
    parser.add_argument(
        "--port", "-p",
        type=int, 
        default=8000,
        help="服务器监听端口 "
    )
    parser.add_argument(
        "--device", "-d",
        type=str, 
        default="cpu",
        choices=["cpu", "gpu", "nvidia"],
        help="运行设备 "
    )
    parser.add_argument(
        "--device-id", 
        type=int, 
        default=0,
        help="GPU 设备 ID"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Llaisys OpenAI-Compatible API Server")
    print("=" * 60)
    print(f"  模型路径: {args.model_path}")
    print(f"  设备:     {args.device} (ID: {args.device_id})")
    print(f"  地址:     http://{args.host}:{args.port}")
    print("=" * 60)
    
    run_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()

