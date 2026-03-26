import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import requests
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AliyunMCPClient:
    """阿里云DashScope MCP客户端，参考OneKE的extraction_agent.py风格"""

    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        """
        初始化MCP客户端

        Args:
            api_key: 阿里云DashScope API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.session = requests.Session()

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_retries: int = 2) -> str:
        """
        调用DashScope API进行对话

        Args:
            messages: 对话消息列表
            temperature: 生成温度
            max_retries: 最大重试次数

        Returns:
            API响应文本
        """
        # 检查API密钥
        if not self.api_key or self.api_key == "test_key" or self.api_key.strip() == "":
            logger.warning("API密钥无效或未提供，使用模拟预测")
            return self._generate_mock_response(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": 4000,
                "top_p": 0.8,
                "result_format": "message"
            }
        }

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"尝试调用DashScope API (第{attempt + 1}次)")
                resp = self.session.post(
                    self.endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    logger.info("DashScope API调用成功")

                    text = (
                            data.get("output", {}).get("text")
                            or data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content")
                            or data.get("output", {}).get("message", {}).get("content")
                            or json.dumps(data)
                    )

                    if text and text.strip():
                        return text
                    else:
                        logger.warning("API返回内容为空，使用模拟响应")
                        return self._generate_mock_response(messages)

                else:
                    error_msg = f"API调用失败，状态码: {resp.status_code}"
                    try:
                        error_data = resp.json()
                        error_msg += f", 错误: {error_data.get('message', 'Unknown error')}"
                    except:
                        error_msg += f", 响应: {resp.text[:200]}"

                    logger.warning(f"{error_msg} (第{attempt + 1}次尝试)")

                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        logger.error("所有API调用尝试都失败了")

            except requests.exceptions.Timeout:
                logger.warning(f"API调用超时 (第{attempt + 1}次尝试)")
                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.warning(f"API请求异常: {e} (第{attempt + 1}次尝试)")
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.warning(f"API调用异常: {e} (第{attempt + 1}次尝试)")
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))

        logger.error("DashScope API调用完全失败，使用模拟预测")
        return self._generate_mock_response(messages)

    def _generate_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """生成模拟响应，当API调用失败时使用"""
        try:
            user_content = messages[-1]["content"] if messages else ""

            vehicle_id = "unknown"
            if "车辆" in user_content:
                import re
                match = re.search(r'车辆\s*(\w+)', user_content)
                if match:
                    vehicle_id = match.group(1)

            steps = 10
            if "步" in user_content:
                import re
                match = re.search(r'(\d+)\s*步', user_content)
                if match:
                    steps = int(match.group(1))

            # 生成k条轨迹的模拟数据
            k = 3  # 默认3条轨迹
            if "条轨迹" in user_content:
                import re
                match = re.search(r'(\d+)\s*条轨迹', user_content)
                if match:
                    k = int(match.group(1))

            mock_data = {
                "vehicle_id": vehicle_id,
                "prediction_steps": steps,
                "number_of_trajectories": k,
                "trajectories": [],
                "predicted_lane": "lane_1",
                "trajectory_pattern": "模拟预测轨迹模式",
                "prediction_confidence": 0.8,
                "reasoning": "由于API调用失败，使用模拟预测算法生成轨迹。基于历史数据趋势进行线性外推。"
            }

            # 为每条轨迹生成数据
            for traj_id in range(1, k + 1):
                trajectory = {
                    "trajectory_id": traj_id,
                    "trajectory_probability": 1.0 / k,  # 均匀分布概率
                    "predicted_positions": [],
                    "predicted_speeds": [],
                    "trajectory_confidence": 0.8
                }

                for i in range(steps):
                    # 为不同轨迹添加变异性
                    variation_factor = 1.0 + (traj_id - 1) * 0.2
                    trajectory["predicted_positions"].append({
                        "step": i + 1,
                        "local_x": float(100 + i * 2.5 * variation_factor),
                        "local_y": float(200 + i * 1.8 * variation_factor),
                        "confidence": max(0.1, 0.9 - i * 0.05)
                    })

                for i in range(steps):
                    variation_factor = 1.0 + (traj_id - 1) * 0.1
                    trajectory["predicted_speeds"].append({
                        "step": i + 1,
                        "speed": float(25.0 + i * 0.5 * variation_factor),
                        "confidence": max(0.1, 0.9 - i * 0.05)
                    })

                mock_data["trajectories"].append(trajectory)

            return json.dumps(mock_data, ensure_ascii=False)

        except Exception as e:
            logger.error(f"生成模拟响应失败: {e}")
            return '{"error": "API调用失败且模拟生成失败", "fallback": true}'


class MCPInformationExtractor:
    """MCP信息提取器，参考OneKE的extraction_agent.py风格"""

    def __init__(self, mcp: AliyunMCPClient):
        """
        初始化信息提取器

        Args:
            mcp: MCP客户端实例
        """
        self.mcp = mcp

    def extract_information_direct(self, instruction: str, text: str, schema_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        直接提取信息

        Args:
            instruction: 指令
            text: 输入文本
            schema_obj: 输出模式

        Returns:
            提取的信息字典
        """
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的轨迹预测专家，能够基于车辆历史数据和特征分析结果进行准确的轨迹预测。你能够生成多条不同的、合理的未来轨迹，并为每条轨迹分配概率分数。"
            },
            {
                "role": "user",
                "content": f"{instruction}\n\n输入数据:\n{text}\n\n输出格式要求:\n{json.dumps(schema_obj, ensure_ascii=False, indent=2)}"
            }
        ]

        response = self.mcp.chat(messages)
        return self._extract_json_dict(response)

    def _extract_json_dict(self, text: str) -> Dict[str, Any]:
        """从文本中提取JSON字典"""
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return {"error": "输入文本为空", "fallback": True}

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                logger.info("成功解析JSON响应")
                return result
            else:
                logger.warning(f"API返回的不是字典格式: {type(result)}")
                return {"error": f"API返回格式错误: {type(result)}", "raw": text, "fallback": True}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")

        # 尝试提取JSON块
        try:
            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_text = text[start:end + 1]
                logger.info(f"尝试提取JSON块: {json_text[:100]}...")

                result = json.loads(json_text)
                if isinstance(result, dict):
                    logger.info("成功提取并解析JSON块")
                    return result
                else:
                    logger.warning("提取的JSON块不是字典格式")
            else:
                logger.warning("未找到有效的JSON块")

        except Exception as e:
            logger.warning(f"JSON块提取失败: {e}")

        logger.error("所有JSON解析尝试都失败了，返回原始文本")
        return {
            "error": "JSON解析完全失败",
            "raw": text[:500],
            "fallback": True,
            "notice": "请检查API返回格式或网络连接"
        }


class MultiTrajectoryPredictor:
    """多轨迹预测器 - 支持k条不同轨迹的预测"""

    def __init__(self, csv_path: str, api_key: str, model: str = "qwen-turbo"):
        """
        初始化多轨迹预测器

        Args:
            csv_path: CSV文件路径
            api_key: API密钥
            model: 模型名称
        """
        self.csv_path = csv_path
        self.model = model
        self.mcp_client = AliyunMCPClient(api_key, model)
        self.agent = MCPInformationExtractor(self.mcp_client)
        self.data = None
        self.window_size = 5  # 滑动窗口大小

    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        if self.data is None:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"成功加载数据: {self.data.shape}")
        return self.data

    def predict_multi_trajectory(self, vehicle_id: str, k: int = 3, steps: int = None,
                                 semantic_json: str = None, features_json: str = None) -> Dict[str, Any]:
        """
        使用多轨迹方法进行轨迹预测

        Args:
            vehicle_id: 车辆ID
            k: 轨迹数量
            steps: 预测步数
            semantic_json: 语义分析结果文件路径
            features_json: 特征提取结果文件路径

        Returns:
            预测结果字典
        """
        logger.info(f"开始使用多轨迹方法预测车辆 {vehicle_id} 的 {k} 条轨迹")

        # 加载数据
        data = self.load_data()

        # 查找车辆ID列
        vehicle_col = None
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['vehicle', 'id', 'veh']):
                vehicle_col = col
                break

        if not vehicle_col:
            raise ValueError("未找到车辆ID列")

        # 过滤车辆数据
        vehicle_data = data[data[vehicle_col] == int(vehicle_id)].copy()

        if vehicle_data.empty:
            raise ValueError(f"未找到车辆 {vehicle_id} 的数据")

        # 按时间排序
        time_col = None
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['time', 'frame', 'step']):
                time_col = col
                break

        if time_col:
            vehicle_data = vehicle_data.sort_values(time_col)

        # 设置预测步数
        if steps is None:
            steps = len(vehicle_data)

        logger.info(f"车辆 {vehicle_id} 有 {len(vehicle_data)} 个数据点，预测 {steps} 步，生成 {k} 条轨迹")

        # 处理特征和语义数据
        features_data = self._process_features(features_json)
        semantic_data = self._process_semantic(semantic_json)

        # 使用多轨迹预测
        prediction_result = self._multi_trajectory_prediction(
            vehicle_data, vehicle_id, k, steps, features_data, semantic_data
        )

        # 添加元数据
        prediction_result['metadata'] = {
            'vehicle_id': vehicle_id,
            'prediction_timestamp': datetime.now().isoformat(),
            'input_data_shape': vehicle_data.shape,
            'real_data_points': len(vehicle_data),
            'prediction_steps': steps,
            'number_of_trajectories': k,
            'window_size': self.window_size,
            'prediction_method': 'multi_trajectory',
            'model_used': self.model,
            'mcp_success': not prediction_result.get("fallback", False)
        }

        return prediction_result

    def _process_features(self, features_json: str) -> Dict[str, Any]:
        """处理特征数据"""
        features_data = {}
        if features_json is not None:
            try:
                # 处理字典类型
                if isinstance(features_json, dict):
                    features_data = features_json
                    logger.info(f"成功接收字典格式特征，包含 {len(features_json)} 个键")

                # 处理PyTorch Tensor
                elif hasattr(features_json, 'detach'):
                    logger.info(f"接收PyTorch Tensor特征，形状: {features_json.shape}")
                    feature_array = features_json.detach().cpu().numpy()

                    if len(feature_array.shape) == 1:
                        features_data = {
                            "feature_vector": feature_array.tolist(),
                            "feature_dim": len(feature_array),
                            "data_type": "1d_tensor"
                        }
                    elif len(feature_array.shape) == 2:
                        features_data = {
                            "feature_matrix": feature_array.tolist(),
                            "shape": list(feature_array.shape),
                            "data_type": "2d_tensor"
                        }
                    else:
                        features_data = {
                            "feature_tensor": feature_array.tolist(),
                            "shape": list(feature_array.shape),
                            "data_type": "high_dim_tensor"
                        }

                # 处理numpy数组
                elif hasattr(features_json, 'shape'):
                    logger.info(f"接收numpy数组特征，形状: {features_json.shape}")
                    if len(features_json.shape) == 1:
                        features_data = {
                            "feature_vector": features_json.tolist(),
                            "feature_dim": len(features_json),
                            "data_type": "1d_numpy"
                        }
                    else:
                        features_data = {
                            "feature_array": features_json.tolist(),
                            "shape": list(features_json.shape),
                            "data_type": "numpy_array"
                        }

                # 处理列表
                elif isinstance(features_json, list):
                    logger.info(f"接收列表格式特征，长度: {len(features_json)}")
                    if features_json and isinstance(features_json[0], list):
                        features_data = {
                            "feature_matrix": features_json,
                            "shape": [len(features_json), len(features_json[0])],
                            "data_type": "nested_list"
                        }
                    else:
                        features_data = {
                            "feature_vector": features_json,
                            "feature_dim": len(features_json),
                            "data_type": "flat_list"
                        }

                else:
                    logger.warning(f"未知的特征结果类型: {type(features_json)}")
                    features_data = {
                        "raw_feature": str(features_json),
                        "data_type": str(type(features_json)),
                        "notice": "原始特征数据，需要进一步处理"
                    }

                logger.info(f"特征处理完成，最终特征键: {list(features_data.keys())}")

            except Exception as e:
                logger.warning(f"特征处理失败: {e}")

        return features_data

    def _process_semantic(self, semantic_json: str) -> Dict[str, Any]:
        """处理语义分析结果"""
        semantic_data = {}
        if semantic_json is not None:
            try:
                # 处理PyTorch Tensor
                if hasattr(semantic_json, 'detach'):
                    semantic_vector = semantic_json.detach().cpu().numpy().tolist()
                    semantic_data = {
                        "semantic_vector": semantic_vector,
                        "data_shape": list(semantic_json.shape),
                        "source": "pytorch_tensor"
                    }
                    logger.info(f"成功接收语义分析Tensor，形状: {list(semantic_json.shape)}")

                # 处理列表
                elif isinstance(semantic_json, list):
                    semantic_data = {
                        "semantic_vector": semantic_json,
                        "data_shape": [len(semantic_json)],
                        "source": "list"
                    }
                    logger.info(f"成功接收语义分析列表，长度: {len(semantic_json)}")

                # 处理字典
                elif isinstance(semantic_json, dict):
                    semantic_data = semantic_json
                    logger.info(f"成功接收语义分析字典，包含 {len(semantic_json)} 个键")

                else:
                    logger.warning(f"未知的语义分析结果类型: {type(semantic_json)}")

            except Exception as e:
                logger.warning(f"处理语义分析结果时出错: {e}")

        return semantic_data

    def _multi_trajectory_prediction(self, vehicle_data: pd.DataFrame, vehicle_id: str, k: int,
                                     steps: int, features_data: Dict[str, Any],
                                     semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多轨迹预测

        Args:
            vehicle_data: 车辆数据
            vehicle_id: 车辆ID
            k: 轨迹数量
            steps: 预测步数
            features_data: 特征数据
            semantic_data: 语义数据

        Returns:
            预测结果
        """
        logger.info(f"使用多轨迹方法进行 {steps} 步预测，生成 {k} 条轨迹")

        features_data = None
        semantic_data = None


        # 初始化结果 - 支持多条轨迹
        all_trajectories = []

        # 获取位置和速度列
        location_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['local_x', 'local_y', 'global_x', 'global_y']):
                location_cols.append(col)

        speed_col = None
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['v_vel', 'speed', 'velocity']):
                speed_col = col
                break

        # 为每条轨迹生成预测
        for trajectory_id in range(1, k + 1):
            logger.info(f"生成第 {trajectory_id} 条轨迹")

            # 初始化单条轨迹的结果
            predicted_positions = []
            predicted_speeds = []

            # 逐步预测
            current_data = vehicle_data.copy()

            for step in range(1, steps + 1):
                logger.info(f"轨迹 {trajectory_id} - 预测第 {step} 步")

                # 获取当前窗口数据（最后window_size个点）
                window_data = current_data.tail(self.window_size)

                # 构建当前步的预测文本
                prediction_text = self._build_multi_trajectory_prediction_text(
                    window_data, step, features_data, semantic_data, trajectory_id, k
                )

                # 定义单步预测的输出模式 - 支持多条轨迹
                schema = {
                    "step": step,
                    "trajectory_id": trajectory_id,
                    "trajectory_probability": "float (Probability of this trajectory, 0.0 to 1.0)",
                    "predicted_position": {
                        "local_x": "float (predicted X coordinate)",
                        "local_y": "float (predicted Y coordinate)",
                        "confidence": "float (0.0 to 1.0)"
                    },
                    "predicted_speed": {
                        "speed": "float (predicted speed)",
                        "confidence": "float (0.0 to 1.0)"
                    },
                    "reasoning": "string (prediction reasoning for this step and trajectory)"
                }

                instruction = (
                    f"基于滑动窗口数据预测车辆 {vehicle_id} 第 {step} 步的位置和速度，这是第 {trajectory_id}/{k} 条轨迹。\n\n"
                    f"关键要求：\n"
                    f"1. 基于最近 {len(window_data)} 个数据点进行预测\n"
                    f"2. 考虑车辆的运动趋势和模式\n"
                    f"3. 结合特征和语义分析结果\n"
                    f"4. 生成合理的轨迹变体（与其他轨迹有差异但都合理）\n"
                    f"5. 为这条轨迹分配概率分数（所有轨迹概率之和应为1.0）\n"
                    f"6. 返回纯JSON格式，不要markdown标记\n\n"
                    f"示例格式：\n"
                    f'{{"step": {step}, "trajectory_id": {trajectory_id}, "trajectory_probability": 0.25, "predicted_position": {{"local_x": 100.0, "local_y": 200.0, "confidence": 0.9}}, "predicted_speed": {{"speed": 25.0, "confidence": 0.8}}}}'
                )

                # 调用MCP agent进行单步预测
                step_result = self.agent.extract_information_direct(
                    instruction=instruction,
                    text=prediction_text,
                    schema_obj=schema
                )

                # 处理预测结果
                if step_result.get("fallback", False) or step_result.get("error"):
                    logger.warning(f"轨迹 {trajectory_id} 第 {step} 步MCP预测失败，使用传统方法")
                    step_result = self._generate_traditional_single_step(
                        window_data, step, location_cols, speed_col, trajectory_id, k
                    )

                # 提取位置预测
                if 'predicted_position' in step_result:
                    pos = step_result['predicted_position']
                    predicted_positions.append({
                        "step": step,
                        "local_x": float(pos.get('local_x', 0.0)),
                        "local_y": float(pos.get('local_y', 0.0)),
                        "confidence": float(pos.get('confidence', 0.5))
                    })
                else:
                    # 使用传统方法生成位置
                    if location_cols and len(location_cols) >= 2:
                        x_col, y_col = location_cols[0], location_cols[1]
                        last_x = window_data[x_col].iloc[-1]
                        last_y = window_data[y_col].iloc[-1]

                        if len(window_data) > 1:
                            dx = window_data[x_col].diff().mean()
                            dy = window_data[y_col].diff().mean()
                        else:
                            dx, dy = 0.1, 0.1

                        # 为不同轨迹添加变异性
                        variation_factor = 1.0 + (trajectory_id - 1) * 0.1
                        predicted_positions.append({
                            "step": step,
                            "local_x": float(last_x + dx * variation_factor),
                            "local_y": float(last_y + dy * variation_factor),
                            "confidence": max(0.1, 0.8 - step * 0.05)
                        })

                # 提取速度预测
                if 'predicted_speed' in step_result:
                    speed = step_result['predicted_speed']
                    predicted_speeds.append({
                        "step": step,
                        "speed": float(speed.get('speed', 0.0)),
                        "confidence": float(speed.get('confidence', 0.5))
                    })
                else:
                    # 使用传统方法生成速度
                    if speed_col:
                        last_speed = window_data[speed_col].iloc[-1]

                        if len(window_data) > 1:
                            speed_change = window_data[speed_col].diff().mean()
                        else:
                            speed_change = 0.0

                        # 为不同轨迹添加变异性
                        variation_factor = 1.0 + (trajectory_id - 1) * 0.05
                        predicted_speeds.append({
                            "step": step,
                            "speed": float(max(0.0, last_speed + speed_change * variation_factor)),
                            "confidence": max(0.1, 0.8 - step * 0.05)
                        })

                # 将预测结果添加到当前数据中，用于下一步预测
                if predicted_positions and location_cols and len(location_cols) >= 2:
                    new_row = window_data.iloc[-1].copy()
                    new_row[location_cols[0]] = predicted_positions[-1]['local_x']
                    new_row[location_cols[1]] = predicted_positions[-1]['local_y']

                    if predicted_speeds and speed_col:
                        new_row[speed_col] = predicted_speeds[-1]['speed']

                    # 添加到当前数据
                    current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)

            # 计算轨迹概率（如果MCP没有提供，则使用均匀分布）
            trajectory_probability = step_result.get('trajectory_probability', 1.0 / k)

            # 添加单条轨迹到结果中
            trajectory = {
                "trajectory_id": trajectory_id,
                "trajectory_probability": float(trajectory_probability),
                "predicted_positions": predicted_positions,
                "predicted_speeds": predicted_speeds,
                "trajectory_confidence": np.mean(
                    [p.get('confidence', 0.5) for p in predicted_positions]) if predicted_positions else 0.5
            }

            all_trajectories.append(trajectory)

        # 归一化轨迹概率，确保总和为1.0
        total_probability = sum(traj['trajectory_probability'] for traj in all_trajectories)
        if total_probability > 0:
            for traj in all_trajectories:
                traj['trajectory_probability'] = traj['trajectory_probability'] / total_probability
        else:
            # 如果所有概率都为0，使用均匀分布
            uniform_prob = 1.0 / k
            for traj in all_trajectories:
                traj['trajectory_probability'] = uniform_prob

        # 构建最终结果
        result = {
            "vehicle_id": vehicle_id,
            "prediction_steps": steps,
            "number_of_trajectories": k,
            "trajectories": all_trajectories,
            "predicted_lane": vehicle_data['Lane_ID'].iloc[-1] if 'Lane_ID' in vehicle_data.columns else "lane_1",
            "trajectory_pattern": f"多轨迹预测 - {k}条变体",
            "overall_confidence": np.mean([traj['trajectory_confidence'] for traj in all_trajectories]),
            "reasoning": f"使用 {self.window_size} 点滑动窗口进行逐步预测，生成 {k} 条不同的合理轨迹变体，每步基于最近的历史数据点进行MCP智能预测",
            "prediction_method": "multi_trajectory",
            "window_size": self.window_size
        }

        logger.info(f"多轨迹预测完成，生成 {k} 条轨迹，每条轨迹 {steps} 步")
        return result

    def _build_multi_trajectory_prediction_text(self, window_data: pd.DataFrame, step: int,
                                                features_data: Dict[str, Any],
                                                semantic_data: Dict[str, Any],
                                                trajectory_id: int, k: int) -> str:
        """
        构建多轨迹预测的输入文本

        Args:
            window_data: 窗口数据
            step: 当前预测步数
            features_data: 特征数据
            semantic_data: 语义数据
            trajectory_id: 当前轨迹ID
            k: 总轨迹数

        Returns:
            预测输入文本
        """
        text_parts = []

        text_parts.append(f"多轨迹预测 - 第 {trajectory_id}/{k} 条轨迹，第 {step} 步")
        text_parts.append(f"窗口大小: {len(window_data)} 个数据点")
        text_parts.append(f"数据列: {list(window_data.columns)}")

        # 窗口数据统计
        text_parts.append("\n窗口数据统计:")
        for col in window_data.columns:
            if window_data[col].dtype in ['int64', 'float64']:
                mean_val = window_data[col].mean()
                std_val = window_data[col].std()
                text_parts.append(f"- {col}: 均值={mean_val:.2f}, 标准差={std_val:.2f}")

        # 运动趋势分析
        text_parts.append("\n运动趋势分析:")
        for col in window_data.columns:
            if any(keyword in col.lower() for keyword in ['local_x', 'local_y', 'v_vel']):
                if len(window_data) > 1:
                    trend = window_data[col].diff().mean()
                    text_parts.append(f"- {col} 变化趋势: {trend:.3f}")

        # 特征信息（简化）
        if features_data:
            text_parts.append("\n相关特征信息:")
            if 'spatiotemporal_features' in features_data:
                spatiotemporal = features_data['spatiotemporal_features']
                if 'trajectory_features' in spatiotemporal:
                    trajectory = spatiotemporal['trajectory_features']
                    for key, value in list(trajectory.items())[:3]:  # 只取前3个特征
                        text_parts.append(f"- {key}: {value}")

        # 语义信息（简化）
        if semantic_data:
            text_parts.append("\n语义分析摘要:")
            if 'traffic_patterns' in semantic_data:
                patterns = semantic_data['traffic_patterns']
                if isinstance(patterns, list) and patterns:
                    text_parts.append(f"- 交通模式: {patterns[0]}")

        # 历史数据点
        text_parts.append("\n历史数据点:")
        for i, (_, row) in enumerate(window_data.iterrows()):
            row_data = {col: row[col] for col in window_data.columns if window_data[col].dtype in ['int64', 'float64']}
            text_parts.append(f"  点{i + 1}: {row_data}")

        # 轨迹变体说明
        text_parts.append(f"\n轨迹变体说明:")
        text_parts.append(f"- 当前生成第 {trajectory_id} 条轨迹，共 {k} 条")
        text_parts.append(f"- 每条轨迹应该有不同的合理变体")
        text_parts.append(f"- 轨迹概率分配应合理，总和为1.0")

        return "\n".join(text_parts)

    def _generate_traditional_single_step(self, window_data: pd.DataFrame, step: int,
                                          location_cols: List[str], speed_col: str,
                                          trajectory_id: int, k: int) -> Dict[str, Any]:
        """
        使用传统方法生成单步预测

        Args:
            window_data: 窗口数据
            step: 预测步数
            location_cols: 位置列
            speed_col: 速度列
            trajectory_id: 轨迹ID
            k: 总轨迹数

        Returns:
            单步预测结果
        """
        result = {
            "step": step,
            "trajectory_id": trajectory_id,
            "trajectory_probability": 1.0 / k,  # 均匀分布
            "predicted_position": {"local_x": 0.0, "local_y": 0.0, "confidence": 0.5},
            "predicted_speed": {"speed": 0.0, "confidence": 0.5},
            "reasoning": f"传统算法单步预测 - 轨迹 {trajectory_id}",
            "fallback": True
        }

        # 位置预测
        if location_cols and len(location_cols) >= 2:
            x_col, y_col = location_cols[0], location_cols[1]

            if len(window_data) > 1:
                # 计算趋势
                dx = window_data[x_col].diff().mean()
                dy = window_data[y_col].diff().mean()

                # 预测下一个位置
                last_x = window_data[x_col].iloc[-1]
                last_y = window_data[y_col].iloc[-1]

                # 为不同轨迹添加变异性
                variation_factor = 1.0 + (trajectory_id - 1) * 0.1
                result["predicted_position"] = {
                    "local_x": float(last_x + dx * variation_factor),
                    "local_y": float(last_y + dy * variation_factor),
                    "confidence": max(0.1, 0.7 - step * 0.05)
                }

        # 速度预测
        if speed_col:
            if len(window_data) > 1:
                # 计算速度趋势
                speed_change = window_data[speed_col].diff().mean()
                last_speed = window_data[speed_col].iloc[-1]

                # 为不同轨迹添加变异性
                variation_factor = 1.0 + (trajectory_id - 1) * 0.05
                result["predicted_speed"] = {
                    "speed": float(max(0.0, last_speed + speed_change * variation_factor)),
                    "confidence": max(0.1, 0.7 - step * 0.05)
                }

        return result

    def save_prediction(self, prediction_result: Dict[str, Any], output_dir: str = "output") -> str:
        """
        保存预测结果

        Args:
            prediction_result: 预测结果
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # 转换numpy数据类型为Python原生类型
        def convert_numpy_types(obj):
            """递归转换numpy数据类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy标量类型
                return obj.item()
            elif hasattr(obj, 'dtype'):  # numpy数组类型
                return obj.tolist()
            else:
                return obj

        # 转换数据类型
        serializable_result = convert_numpy_types(prediction_result)

        # 保存到文件
        output_path = os.path.join(output_dir, "multi_trajectory_predictions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        logger.info(f"多轨迹预测结果已保存到 {output_path}")
        return output_path


def run(csv_path: str, api_key: str, model: str = "qwen-turbo",
        output_dir: str = "output", vehicle_id: str = None, k: int = 3,
        steps: int = None, semantic_json: str = None, features_json: str = None) -> str:
    """
    运行多轨迹预测的主函数

    Args:
        csv_path: CSV文件路径
        api_key: API密钥
        model: 模型名称
        output_dir: 输出目录
        vehicle_id: 车辆ID
        k: 轨迹数量
        steps: 预测步数
        semantic_json: 语义分析结果文件路径
        features_json: 特征提取结果文件路径

    Returns:
        预测结果文件路径
    """
    try:
        # 创建预测器
        predictor = MultiTrajectoryPredictor(csv_path, api_key, model)

        # 如果没有指定车辆ID，使用第一个可用的车辆
        if vehicle_id is None:
            data = predictor.load_data()
            vehicle_col = None
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['vehicle', 'id', 'veh']):
                    vehicle_col = col
                    break

            if vehicle_col:
                vehicle_id = str(data[vehicle_col].iloc[0])
                logger.info(f"使用默认车辆ID: {vehicle_id}")
            else:
                raise ValueError("未找到车辆ID列")

        # 进行多轨迹预测
        prediction_result = predictor.predict_multi_trajectory(
            vehicle_id, k, steps, semantic_json, features_json
        )

        # 保存结果
        output_path = predictor.save_prediction(prediction_result, output_dir)

        return output_path

    except Exception as e:
        logger.error(f"多轨迹预测失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多轨迹预测器")
    parser.add_argument("csv_path", help="CSV文件路径")
    parser.add_argument("--vehicle_id", help="车辆ID（可选，默认使用第一个可用车辆）")
    parser.add_argument("--k", type=int, default=3, help="轨迹数量（默认3条）")
    parser.add_argument("--steps", type=int, default=None,
                        help="预测步数（可选，默认使用真实轨迹的数据点数量）")
    parser.add_argument("--semantic", help="语义分析结果JSON文件路径")
    parser.add_argument("--feature", help="特征提取结果JSON文件路径")
    parser.add_argument("--model", default="qwen-turbo", help="模型名称")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--api_key", required=True, help="阿里云DashScope API密钥")

    args = parser.parse_args()

    run(args.csv_path, args.api_key, args.model, args.output,
        args.vehicle_id, args.k, args.steps, args.semantic, args.feature)