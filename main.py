import os
import json
import argparse
import torch
import features_semantic
import semantic_semantic
import time
import numpy as np
from datetime import datetime
from aliyun_mcp_feature_extraction import run as run_features
from aliyun_mcp_semantic_analysis import run as run_semantic
from aliyun_mcp_trajectory_prediction import run as run_multi_predict
from dashscope import Generation
import semantic_communication


def load_api_key_from_file(key_file_path: str = None) -> str:
    """
    从API密钥文件中读取密钥

    Args:
        key_file_path: API密钥文件路径，默认为当前目录下的 'api_key.txt'

    Returns:
        API密钥字符串
    """
    if key_file_path is None:
        key_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'api_key.txt')

    try:
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
                else:
                    print(f"警告: API密钥文件 {key_file_path} 为空")
        else:
            print(f"警告: API密钥文件 {key_file_path} 不存在")
    except Exception as e:
        print(f"读取API密钥文件时出错: {e}")

    # 如果文件读取失败，尝试从环境变量获取
    env_key = os.getenv("DASHSCOPE_API_KEY")
    if env_key:
        print("使用环境变量中的API密钥")
        return env_key
    else:
        print("错误: 无法获取API密钥，请检查密钥文件或环境变量")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Main orchestrator for Aliyun MCP traffic pipeline with multi-trajectory prediction")
    parser.add_argument("--csv", help="Fallback CSV path if features not provided")
    parser.add_argument("--vehicle_id", default=None, help="Vehicle ID for prediction; defaults to last in CSV")
    parser.add_argument("--k", type=int, default=1, help="Number of different trajectories to predict (default: 5)")
    parser.add_argument("--steps", type=int, default=1, help="Future steps to predict")
    parser.add_argument("--model", default="qwen-turbo", help="DashScope model name")
    parser.add_argument("--api_key_file", default="../api_key", help="Path to API key file")
    parser.add_argument("--api_key", default=None, help="DashScope API key (overrides key file)")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    # 加载API密钥
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = load_api_key_from_file(args.api_key_file)
        if not api_key:
            print("错误: 无法获取API密钥，程序退出")
            return

    os.makedirs(args.output, exist_ok=True)
    features_path = os.path.join(args.output, "features_mcp.json")
    semantic_path = os.path.join(args.output, "semantic_mcp.json")

    print("[1/3] Feature extraction...")
    _ = run_features(args.csv, api_key=api_key, model=args.model, output_dir=args.output)
    print(f"Saved: {features_path}")

    print("[2/3] Semantic analysis...")
    _ = run_semantic(csv_path=None, api_key=api_key, model=args.model, feature_json=features_path,
                     output_dir=args.output)
    print(f"Saved: {semantic_path}")

    # 加载特征和语义分析结果
    features_results = features_semantic.load_and_convert_structured_json(features_path)
    semantic_results = semantic_semantic.load_and_convert_json(semantic_path)

    # 初始化语义通信模型
    model = semantic_communication.SemCom_KAN()

    # 设置模型为评估模式
    model.eval()

    # 进行推理
    with torch.no_grad():
        features_output = model(features_results, snr=20.0)
        semantic_output = model(semantic_results, snr=20.0)

    print(f"特征输出形状: {features_output.shape if hasattr(features_output, 'shape') else 'N/A'}")
    print(f"语义输出形状: {semantic_output.shape if hasattr(semantic_output, 'shape') else 'N/A'}")

    print(f"[3/3] Multi-trajectory prediction (k={args.k})...")

    # 使用多轨迹预测器
    _ = run_multi_predict(
        csv_path=args.csv,
        api_key=api_key,
        model=args.model,
        output_dir=args.output,
        vehicle_id=args.vehicle_id,
        k=args.k,  # 预测k条不同轨迹
        steps=args.steps,
        semantic_json=semantic_output,
        features_json=features_output
    )

    print(f"Saved: {os.path.join(args.output, 'multi_trajectory_predictions.json')}")
    print(f"预测完成: 生成了 {args.k} 条不同的轨迹，每条轨迹 {args.steps} 步")

    print("Pipeline finished at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
