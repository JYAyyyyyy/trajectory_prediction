import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import requests
import time
from datetime import datetime
import warnings
from scipy import stats

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
                "max_tokens": 4000,  # 增加token限制
                "top_p": 0.8,
                "result_format": "message"  # 指定返回格式
            }
        }

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"尝试调用DashScope API (第{attempt + 1}次)")
                resp = self.session.post(
                    self.endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120  # 增加超时时间
                )

                if resp.status_code == 200:
                    data = resp.json()
                    logger.info("DashScope API调用成功")

                    # 尝试从不同路径提取生成文本
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
                    logger.warning(f"API调用失败，状态码: {resp.status_code}, 响应: {resp.text}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # 指数退避
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        logger.error("达到最大重试次数，使用模拟响应")
                        return self._generate_mock_response(messages)

            except Exception as e:
                logger.error(f"API调用异常: {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("达到最大重试次数，使用模拟响应")
                    return self._generate_mock_response(messages)

        # 如果所有尝试都失败，返回模拟响应
        return self._generate_mock_response(messages)

    def _generate_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """生成模拟响应"""
        logger.warning("使用模拟响应")
        # 提取用户消息
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # 检查是否包含JSON schema
        if "JSON Schema" in user_message:
            # 尝试提取schema并生成模拟JSON
            try:
                schema_start = user_message.find("JSON Schema")
                schema_text = user_message[schema_start:]
                # 简单模拟一个JSON响应
                return '{"vehicle_id": "mock_vehicle", "prediction_steps": 5, "predicted_positions": [{"step": 1, "local_x": 100.0, "local_y": 200.0, "confidence": 0.9}], "predicted_speeds": [{"step": 1, "speed": 25.0, "confidence": 0.9}], "predicted_lane": "1", "trajectory_pattern": "直线行驶", "prediction_confidence": 0.8, "reasoning": "这是一个模拟预测结果"}'
            except Exception as e:
                logger.error(f"生成模拟JSON响应失败: {e}")
                return '{"error": "模拟响应生成失败", "fallback": true}'
        else:
            # 一般性回复
            return "这是一个模拟响应，API密钥无效或API调用失败。请提供有效的API密钥。"


class MCPInformationExtractor:
    """MCP信息提取器"""

    def __init__(self, mcp: AliyunMCPClient):
        """
        初始化信息提取器

        Args:
            mcp: MCP客户端实例
        """
        self.mcp = mcp

    def _extract_json_dict(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取JSON字典

        Args:
            text: 输入文本

        Returns:
            提取的JSON字典
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON部分
        try:
            # 查找可能的JSON开始和结束位置
            start_pos = text.find('{')
            end_pos = text.rfind('}')

            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                json_text = text[start_pos:end_pos + 1]
                return json.loads(json_text)
        except json.JSONDecodeError:
            pass

        # 尝试使用正则表达式提取
        try:
            import re
            json_pattern = r'\{[^\{\}]*((\{[^\{\}]*\})[^\{\}]*)*\}'
            matches = re.findall(json_pattern, text)
            if matches:
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            match = match[0]
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        # 所有尝试都失败，返回错误信息
        logger.error(f"无法从文本中提取JSON: {text[:100]}...")
        return {"error": "无法提取JSON", "fallback": True}

    def extract_information(self, instruction: str, text: str, examples: str, schema: str,
                           additional_info: str = "") -> Dict[str, Any]:
        """
        提取信息

        Args:
            instruction: 指令
            text: 输入文本
            examples: 示例
            schema: 输出模式
            additional_info: 附加信息

        Returns:
            提取的信息
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert information extraction agent. "
                    "Extract information from the provided text according to the instruction. "
                    "Return a valid JSON object matching the schema."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{instruction}\n\n"
                    f"Text:\n{text}\n\n"
                    f"Examples:\n{examples}\n\n"
                    f"Additional Info:\n{additional_info}\n\n"
                    f"JSON Schema (keys & types hint):\n{schema}\n\n"
                    "Return a valid JSON object matching the schema. Do not include any explanations."
                )
            }
        ]

        resp = self.mcp.chat(messages)
        parsed = self._extract_json_dict(resp)
        # 若失败，进行一次纠错重试（更严格的格式约束）
        if parsed.get("fallback") or parsed.get("error"):
            logger.info("第一次解析失败，发起纠错重试请求")
            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert traffic trajectory prediction MCP agent. "
                        "Return strictly valid, minified JSON matching the schema. No markdown, no extra text."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Instruction (strict):\n{instruction}\n\n"
                        f"Text:\n{text}\n\n"
                        f"Additional Info:\n{additional_info}\n\n"
                        f"JSON Schema (keys & types hint):\n{schema}\n\n"
                        "Output only pure JSON."
                    )
                }
            ]
            resp2 = self.mcp.chat(retry_messages, temperature=0.1)
            parsed2 = self._extract_json_dict(resp2)
            return parsed2
        return parsed


class MCPExtractionAgent:
    """MCP提取代理，参考OneKE的extraction_agent.py风格"""

    def __init__(self, mcp: AliyunMCPClient):
        """
        初始化提取代理

        Args:
            mcp: MCP客户端实例
        """
        self.mcp = mcp
        self.module = MCPInformationExtractor(mcp)

    def extract_information_direct(self, instruction: str, text: str, schema_obj: Dict[str, Any],
                                   examples: Optional[str] = None, constraint: Optional[Any] = None) -> Dict[str, Any]:
        """
        直接提取信息

        Args:
            instruction: 指令
            text: 输入文本
            schema_obj: 输出模式对象
            examples: 示例
            constraint: 约束条件

        Returns:
            提取的信息
        """
        schema = json.dumps(schema_obj, ensure_ascii=False)
        return self.module.extract_information(
            instruction=instruction,
            text=text,
            examples=examples or "",
            schema=schema,
            additional_info=json.dumps({"constraint": constraint}, ensure_ascii=False) if constraint else ""
        )


class US101TrajectoryDistributionPredictor:
    """US101数据集轨迹概率分布预测器"""

    def __init__(self, csv_path: str, api_key: str, model: str = "qwen-turbo"):
        """
        初始化轨迹概率分布预测器

        Args:
            csv_path: US101 CSV文件路径
            api_key: API密钥
            model: 模型名称
        """
        self.csv_path = csv_path
        self.api_key = api_key
        self.model = model
        self.data = None
        self.mcp = AliyunMCPClient(api_key, model)
        self.agent = MCPExtractionAgent(self.mcp)
        self.feature_results = {}

    def load_data(self) -> pd.DataFrame:
        """加载US101数据集，处理编码问题"""
        try:
            # 尝试多种编码方式读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.csv_path, encoding=encoding)
                    logger.info(f"成功使用 {encoding} 编码加载数据，共 {len(self.data)} 行，{len(self.data.columns)} 列")
                    return self.data
                except UnicodeDecodeError:
                    continue

            # 如果所有编码都失败，尝试使用默认编码
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"成功使用默认编码加载数据，共 {len(self.data)} 行，{len(self.data.columns)} 列")
            return self.data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            # 创建一个空的DataFrame作为备用
            self.data = pd.DataFrame()
            return self.data

    def prepare_vehicle_data(self, vehicle_id: str) -> pd.DataFrame:
        """准备指定车辆的数据"""
        if self.data is None:
            self.load_data()

        if self.data.empty:
            logger.warning("数据为空，无法准备车辆数据")
            return pd.DataFrame()

        # 检查是否有Vehicle_ID列
        vehicle_id_col = None
        for col in self.data.columns:
            if 'vehicle' in col.lower() and 'id' in col.lower():
                vehicle_id_col = col
                break

        if not vehicle_id_col:
            logger.warning("找不到Vehicle_ID列，尝试使用第一列作为ID")
            vehicle_id_col = self.data.columns[0]

        # 如果vehicle_id是'random'，随机选择一个车辆ID
        if vehicle_id.lower() == 'random':
            unique_ids = self.data[vehicle_id_col].unique()
            if len(unique_ids) > 0:
                vehicle_id = str(np.random.choice(unique_ids))
                logger.info(f"随机选择车辆ID: {vehicle_id}")
            else:
                logger.warning("没有可用的车辆ID")
                return pd.DataFrame()

        # 提取指定车辆的数据
        try:
            # 尝试不同的方式匹配车辆ID
            vehicle_data = self.data[self.data[vehicle_id_col] == vehicle_id]
            if vehicle_data.empty:
                # 尝试将vehicle_id转换为整数
                try:
                    vehicle_id_int = int(vehicle_id)
                    vehicle_data = self.data[self.data[vehicle_id_col] == vehicle_id_int]
                except ValueError:
                    pass

            if vehicle_data.empty:
                # 尝试将vehicle_id转换为浮点数
                try:
                    vehicle_id_float = float(vehicle_id)
                    vehicle_data = self.data[self.data[vehicle_id_col] == vehicle_id_float]
                except ValueError:
                    pass

            if vehicle_data.empty:
                logger.warning(f"找不到车辆ID为 {vehicle_id} 的数据")
                return pd.DataFrame()

            logger.info(f"成功提取车辆 {vehicle_id} 的数据，共 {len(vehicle_data)} 行")
            return vehicle_data

        except Exception as e:
            logger.error(f"准备车辆数据失败: {e}")
            return pd.DataFrame()

    def extract_trajectory_features(self, vehicle_data: pd.DataFrame) -> Dict[str, Any]:
        """提取轨迹特征

        Args:
            vehicle_data: 车辆数据

        Returns:
            轨迹特征字典
        """
        features = {}

        # 位置特征
        location_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['local_x', 'local_y', 'global_x', 'global_y', 'lat', 'lon']):
                location_cols.append(col)

        if location_cols:
            features['location_columns'] = location_cols
            for col in location_cols:
                features[f'{col}_mean'] = float(vehicle_data[col].mean())
                features[f'{col}_std'] = float(vehicle_data[col].std())
                features[f'{col}_range'] = float(vehicle_data[col].max() - vehicle_data[col].min())

        # 速度特征
        speed_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['v_vel', 'speed', 'velocity']):
                speed_cols.append(col)

        if speed_cols:
            features['speed_columns'] = speed_cols
            for col in speed_cols:
                features[f'{col}_mean'] = float(vehicle_data[col].mean())
                features[f'{col}_std'] = float(vehicle_data[col].std())
                features[f'{col}_max'] = float(vehicle_data[col].max())
                features[f'{col}_min'] = float(vehicle_data[col].min())

        # 加速度特征
        accel_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['a_vel', 'accel', 'acceleration']):
                accel_cols.append(col)

        if accel_cols:
            features['acceleration_columns'] = accel_cols
            for col in accel_cols:
                features[f'{col}_mean'] = float(vehicle_data[col].mean())
                features[f'{col}_std'] = float(vehicle_data[col].std())

        # 车道特征
        lane_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['lane', 'road']):
                lane_cols.append(col)

        if lane_cols:
            features['lane_columns'] = lane_cols
            for col in lane_cols:
                features[f'{col}_unique'] = list(vehicle_data[col].unique())

        # 时间特征
        time_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['time', 'frame', 'timestamp']):
                time_cols.append(col)

        if time_cols:
            features['time_columns'] = time_cols
            features['total_frames'] = len(vehicle_data)

        return features

    def _build_enhanced_prediction_text(self, vehicle_data: pd.DataFrame, vehicle_id: str, predict_neighbors: bool = False) -> str:
        """
        构建增强的预测输入文本，包含车辆数据和相邻车辆信息

        Args:
            vehicle_data: 车辆数据
            vehicle_id: 车辆ID
            predict_neighbors: 是否预测相邻车辆

        Returns:
            增强的预测文本
        """
        text_parts = []

        # 车辆基本信息
        text_parts.append(f"车辆轨迹预测分析")
        text_parts.append(f"车辆ID: {vehicle_id}")
        text_parts.append(f"数据点数量: {len(vehicle_data)}")
        text_parts.append(f"数据列: {list(vehicle_data.columns)}")

        # 添加车辆轨迹数据摘要
        if not vehicle_data.empty:
            # 获取位置和速度列
            x_col = next((col for col in vehicle_data.columns if 'local_x' in col.lower()), None)
            y_col = next((col for col in vehicle_data.columns if 'local_y' in col.lower()), None)
            v_col = next((col for col in vehicle_data.columns if 'v' == col.lower() or 'velocity' in col.lower() or 'speed' in col.lower()), None)
            lane_col = next((col for col in vehicle_data.columns if 'lane' in col.lower()), None)

            text_parts.append(f"\n轨迹数据摘要:")
            if x_col and y_col:
                text_parts.append(f"- 起始位置: ({vehicle_data[x_col].iloc[0]:.2f}, {vehicle_data[y_col].iloc[0]:.2f})")
                text_parts.append(f"- 结束位置: ({vehicle_data[x_col].iloc[-1]:.2f}, {vehicle_data[y_col].iloc[-1]:.2f})")
            if v_col:
                text_parts.append(f"- 平均速度: {vehicle_data[v_col].mean():.2f}")
                text_parts.append(f"- 最大速度: {vehicle_data[v_col].max():.2f}")
                text_parts.append(f"- 最小速度: {vehicle_data[v_col].min():.2f}")
            if lane_col:
                text_parts.append(f"- 车道信息: {vehicle_data[lane_col].unique().tolist()}")

        # 添加相邻车辆信息（如果需要）
        if predict_neighbors and hasattr(self, 'feature_results') and 'neighbor_info' in self.feature_results:
            neighbor_info = self.feature_results['neighbor_info']
            if 'relationships' in neighbor_info and neighbor_info['relationships']:
                text_parts.append(f"\n相邻车辆信息:")
                for neighbor_id, relation in neighbor_info['relationships'].items():
                    text_parts.append(f"- 车辆 {neighbor_id}: {relation}")

        return "\n".join(text_parts)

    def _generate_traditional_prediction_with_distribution(self, vehicle_data: pd.DataFrame, vehicle_id: str, steps: int = 10) -> Dict[str, Any]:
        """
        使用传统算法生成带概率分布的预测结果

        Args:
            vehicle_data: 车辆数据
            vehicle_id: 车辆ID
            steps: 预测步数

        Returns:
            带概率分布的预测结果
        """
        result = {
            "vehicle_id": vehicle_id,
            "prediction_steps": steps,
            "total_real_data_points": len(vehicle_data),
            "predicted_positions": [],
            "predicted_speeds": [],
            "position_distributions": [],
            "speed_distributions": [],
            "predicted_lane": "1",  # 默认车道
            "trajectory_pattern": "直线行驶",  # 默认模式
            "prediction_confidence": 0.8,  # 默认置信度
            "reasoning": "基于历史轨迹数据的统计预测"
        }

        # 获取位置列
        x_col = None
        y_col = None
        for col in vehicle_data.columns:
            if 'local_x' in col.lower():
                x_col = col
            elif 'local_y' in col.lower():
                y_col = col

        # 获取速度列
        speed_col = None
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['v_vel', 'speed', 'velocity']):
                speed_col = col
                break

        # 获取车道列
        lane_col = None
        for col in vehicle_data.columns:
            if 'lane' in col.lower():
                lane_col = col
                break

        # 生成位置预测
        if x_col and y_col and len(vehicle_data) > 1:
            # 计算历史位置变化
            x_changes = vehicle_data[x_col].diff().dropna().values
            y_changes = vehicle_data[y_col].diff().dropna().values
            
            # 计算位置变化的均值和标准差
            x_mean_change = np.mean(x_changes)
            y_mean_change = np.mean(y_changes)
            x_std_change = np.std(x_changes) if len(x_changes) > 1 else 0.5
            y_std_change = np.std(y_changes) if len(y_changes) > 1 else 0.5
            
            # 确保标准差不为零（避免概率分布计算问题）
            x_std_change = max(0.1, x_std_change)
            y_std_change = max(0.1, y_std_change)
            
            # 获取最后的位置作为起点
            last_x = vehicle_data[x_col].iloc[-1]
            last_y = vehicle_data[y_col].iloc[-1]
            
            # 生成预测点和分布
            for i in range(steps):
                # 预测步数从1开始
                step_num = i + 1
                
                # 预测位置（均值）
                predicted_x = last_x + x_mean_change * step_num
                predicted_y = last_y + y_mean_change * step_num
                
                # 计算置信度（随步数增加而降低）
                confidence = max(0.1, 0.9 - i * 0.05)
                
                # 添加预测位置
                result["predicted_positions"].append({
                    "step": step_num,
                    "local_x": float(predicted_x),
                    "local_y": float(predicted_y),
                    "confidence": confidence
                })
                
                # 生成位置概率分布参数
                # 标准差随预测步数增加而增大（表示不确定性增加）
                x_std = x_std_change * (1 + 0.2 * step_num)
                y_std = y_std_change * (1 + 0.2 * step_num)
                
                # 生成采样点（用于可视化分布）
                num_samples = 10  # 每个分布的采样点数量
                x_samples = np.linspace(predicted_x - 3*x_std, predicted_x + 3*x_std, num_samples)
                y_samples = np.linspace(predicted_y - 3*y_std, predicted_y + 3*y_std, num_samples)
                
                # 计算每个采样点的概率密度
                x_pdf = stats.norm.pdf(x_samples, predicted_x, x_std)
                y_pdf = stats.norm.pdf(y_samples, predicted_y, y_std)
                
                # 归一化PDF值
                x_pdf = x_pdf / np.max(x_pdf)
                y_pdf = y_pdf / np.max(y_pdf)
                
                # 添加位置分布
                result["position_distributions"].append({
                    "step": step_num,
                    "distribution_type": "gaussian",
                    "x_distribution": {
                        "mean": float(predicted_x),
                        "std": float(x_std),
                        "samples": [float(x) for x in x_samples.tolist()],
                        "densities": [float(p) for p in x_pdf.tolist()]
                    },
                    "y_distribution": {
                        "mean": float(predicted_y),
                        "std": float(y_std),
                        "samples": [float(y) for y in y_samples.tolist()],
                        "densities": [float(p) for p in y_pdf.tolist()]
                    }
                })
        else:
            # 如果数据不足，使用默认值
            for i in range(steps):
                step_num = i + 1
                result["predicted_positions"].append({
                    "step": step_num,
                    "local_x": float(100 + i * 2.0),
                    "local_y": float(200 + i * 1.5),
                    "confidence": max(0.1, 0.8 - i * 0.02)
                })
                
                # 默认分布
                x_std = 0.5 * (1 + 0.2 * step_num)
                y_std = 0.5 * (1 + 0.2 * step_num)
                predicted_x = 100 + i * 2.0
                predicted_y = 200 + i * 1.5
                
                # 生成采样点
                num_samples = 10
                x_samples = np.linspace(predicted_x - 3*x_std, predicted_x + 3*x_std, num_samples)
                y_samples = np.linspace(predicted_y - 3*y_std, predicted_y + 3*y_std, num_samples)
                
                # 计算概率密度
                x_pdf = stats.norm.pdf(x_samples, predicted_x, x_std)
                y_pdf = stats.norm.pdf(y_samples, predicted_y, y_std)
                
                # 归一化
                x_pdf = x_pdf / np.max(x_pdf)
                y_pdf = y_pdf / np.max(y_pdf)
                
                result["position_distributions"].append({
                    "step": step_num,
                    "distribution_type": "gaussian",
                    "x_distribution": {
                        "mean": float(predicted_x),
                        "std": float(x_std),
                        "samples": [float(x) for x in x_samples.tolist()],
                        "densities": [float(p) for p in x_pdf.tolist()]
                    },
                    "y_distribution": {
                        "mean": float(predicted_y),
                        "std": float(y_std),
                        "samples": [float(y) for y in y_samples.tolist()],
                        "densities": [float(p) for p in y_pdf.tolist()]
                    }
                })

        # 生成速度预测
        if speed_col and len(vehicle_data) > 1:
            # 计算历史速度变化
            speed_changes = vehicle_data[speed_col].diff().dropna().values
            
            # 计算速度变化的均值和标准差
            speed_mean_change = np.mean(speed_changes)
            speed_std_change = np.std(speed_changes) if len(speed_changes) > 1 else 0.5
            
            # 确保标准差不为零
            speed_std_change = max(0.1, speed_std_change)
            
            # 获取最后的速度作为起点
            last_speed = vehicle_data[speed_col].iloc[-1]
            
            # 生成预测点和分布
            for i in range(steps):
                step_num = i + 1
                
                # 预测速度（均值）
                predicted_speed = last_speed + speed_mean_change * step_num
                # 确保速度不为负数
                predicted_speed = max(0.0, predicted_speed)
                
                # 计算置信度
                confidence = max(0.1, 0.9 - i * 0.05)
                
                # 添加预测速度
                result["predicted_speeds"].append({
                    "step": step_num,
                    "speed": float(predicted_speed),
                    "confidence": confidence
                })
                
                # 生成速度概率分布参数
                # 标准差随预测步数增加而增大
                speed_std = speed_std_change * (1 + 0.2 * step_num)
                
                # 生成采样点
                num_samples = 10
                speed_samples = np.linspace(max(0, predicted_speed - 3*speed_std), predicted_speed + 3*speed_std, num_samples)
                
                # 计算概率密度
                speed_pdf = stats.norm.pdf(speed_samples, predicted_speed, speed_std)
                
                # 归一化
                speed_pdf = speed_pdf / np.max(speed_pdf)
                
                # 添加速度分布
                result["speed_distributions"].append({
                    "step": step_num,
                    "distribution_type": "gaussian",
                    "speed_distribution": {
                        "mean": float(predicted_speed),
                        "std": float(speed_std),
                        "samples": [float(s) for s in speed_samples.tolist()],
                        "densities": [float(p) for p in speed_pdf.tolist()]
                    }
                })
        else:
            # 如果数据不足，使用默认值
            for i in range(steps):
                step_num = i + 1
                predicted_speed = 25.0 + i * 0.3
                
                result["predicted_speeds"].append({
                    "step": step_num,
                    "speed": float(predicted_speed),
                    "confidence": max(0.1, 0.8 - i * 0.02)
                })
                
                # 默认分布
                speed_std = 1.0 * (1 + 0.1 * step_num)
                
                # 生成采样点
                num_samples = 10
                speed_samples = np.linspace(max(0, predicted_speed - 3*speed_std), predicted_speed + 3*speed_std, num_samples)
                
                # 计算概率密度
                speed_pdf = stats.norm.pdf(speed_samples, predicted_speed, speed_std)
                
                # 归一化
                speed_pdf = speed_pdf / np.max(speed_pdf)
                
                result["speed_distributions"].append({
                    "step": step_num,
                    "distribution_type": "gaussian",
                    "speed_distribution": {
                        "mean": float(predicted_speed),
                        "std": float(speed_std),
                        "samples": [float(s) for s in speed_samples.tolist()],
                        "densities": [float(p) for p in speed_pdf.tolist()]
                    }
                })

        # 设置车道预测
        if lane_col and not vehicle_data.empty:
            # 使用最后的车道作为预测车道
            result["predicted_lane"] = str(vehicle_data[lane_col].iloc[-1])

        logger.info(
            f"传统算法成功生成 {len(result['predicted_positions'])} 个位置预测点和 {len(result['predicted_speeds'])} 个速度预测点，"  
            f"以及对应的概率分布")
        return result

    def _predict_trajectory_with_mcp_distribution(self, vehicle_data: pd.DataFrame, vehicle_id: str, steps: int = 10, predict_neighbors: bool = False) -> Dict[str, Any]:
        """
        使用MCP Agent预测轨迹并生成概率分布

        Args:
            vehicle_data: 车辆数据
            vehicle_id: 车辆ID
            steps: 预测步数
            predict_neighbors: 是否预测相邻车辆

        Returns:
            带概率分布的预测结果
        """
        # 构建预测文本
        prediction_text = self._build_enhanced_prediction_text(vehicle_data, vehicle_id, predict_neighbors)
        
        # 获取相邻车辆ID（如果需要）
        neighbor_vehicles = []
        if predict_neighbors and hasattr(self, 'feature_results') and 'neighbor_info' in self.feature_results:
            neighbor_vehicles = list(self.feature_results['neighbor_info'].get('relationships', {}).keys())
        
        # 定义增强的输出模式 - 包含概率分布
        schema = {
            "vehicle_id": "string",
            "prediction_steps": steps,
            "total_real_data_points": len(vehicle_data),
            "predicted_positions": [
                {
                    "step": f"integer (1 to {steps})",
                    "local_x": "float (predicted X coordinate)",
                    "local_y": "float (predicted Y coordinate)",
                    "confidence": "float (0.0 to 1.0, prediction confidence)"
                }
            ],
            "position_distributions": [
                {
                    "step": f"integer (1 to {steps})",
                    "distribution_type": "string (gaussian, mixture, etc.)",
                    "x_distribution": {
                        "mean": "float (mean value)",
                        "std": "float (standard deviation)",
                        "samples": "array of float (sample points for visualization)",
                        "densities": "array of float (probability density at each sample point)"
                    },
                    "y_distribution": {
                        "mean": "float (mean value)",
                        "std": "float (standard deviation)",
                        "samples": "array of float (sample points for visualization)",
                        "densities": "array of float (probability density at each sample point)"
                    }
                }
            ],
            "predicted_speeds": [
                {
                    "step": f"integer (1 to {steps})",
                    "speed": "float (predicted speed)",
                    "confidence": "float (0.0 to 1.0, prediction confidence)"
                }
            ],
            "speed_distributions": [
                {
                    "step": f"integer (1 to {steps})",
                    "distribution_type": "string (gaussian, mixture, etc.)",
                    "speed_distribution": {
                        "mean": "float (mean value)",
                        "std": "float (standard deviation)",
                        "samples": "array of float (sample points for visualization)",
                        "densities": "array of float (probability density at each sample point)"
                    }
                }
            ],
            "predicted_lane": "string (predicted lane ID)",
            "trajectory_pattern": "string (description of predicted trajectory pattern)",
            "prediction_confidence": "float (overall prediction confidence 0.0 to 1.0)",
            "reasoning": "string (explanation of prediction logic)"
        }
        
        # 如果需要预测相邻车辆，添加相应的schema
        if predict_neighbors and neighbor_vehicles:
            schema["neighbor_predictions"] = {
                "type": "object",
                "description": "相邻车辆的轨迹预测结果",
                "properties": {}
            }
            for neighbor_id in neighbor_vehicles:
                schema["neighbor_predictions"]["properties"][neighbor_id] = {
                    "vehicle_id": "string",
                    "relationship": "string (与主车辆的关系描述)",
                    "predicted_positions": [
                        {
                            "step": f"integer (1 to {steps})",
                            "local_x": "float (predicted X coordinate)",
                            "local_y": "float (predicted Y coordinate)",
                            "confidence": "float (0.0 to 1.0, prediction confidence)"
                        }
                    ],
                    "position_distributions": [
                        {
                            "step": f"integer (1 to {steps})",
                            "distribution_type": "string (gaussian, mixture, etc.)",
                            "x_distribution": {
                                "mean": "float (mean value)",
                                "std": "float (standard deviation)",
                                "samples": "array of float (sample points for visualization)",
                                "densities": "array of float (probability density at each sample point)"
                            },
                            "y_distribution": {
                                "mean": "float (mean value)",
                                "std": "float (standard deviation)",
                                "samples": "array of float (sample points for visualization)",
                                "densities": "array of float (probability density at each sample point)"
                            }
                        }
                    ],
                    "predicted_speeds": [
                        {
                            "step": f"integer (1 to {steps})",
                            "speed": "float (predicted speed)",
                            "confidence": "float (0.0 to 1.0, prediction confidence)"
                        }
                    ],
                    "speed_distributions": [
                        {
                            "step": f"integer (1 to {steps})",
                            "distribution_type": "string (gaussian, mixture, etc.)",
                            "speed_distribution": {
                                "mean": "float (mean value)",
                                "std": "float (standard deviation)",
                                "samples": "array of float (sample points for visualization)",
                                "densities": "array of float (probability density at each sample point)"
                            }
                        }
                    ]
                }
        
        instruction = (
            f"基于提供的车辆轨迹数据，使用MCP Agent智能预测车辆 {vehicle_id} 未来 {steps} 步的轨迹，并生成概率分布。\n\n"
            f"关键要求：\n"
            f"1. 真实轨迹有 {len(vehicle_data)} 个数据点，请生成恰好 {steps} 个预测点\n"
            f"2. 对每个预测点，生成位置和速度的概率分布（默认使用高斯分布）\n"
            f"3. 考虑车辆的历史运动模式、速度变化、加速度特征、车道信息\n"
            f"4. 给出合理的位置、速度和车道预测，并说明预测依据\n"
            f"5. 随着预测步数增加，增大分布的标准差以反映不确定性增加\n"
            f"6. 对于每个分布，提供采样点和对应的概率密度值用于可视化\n"
        )
        
        # 如果需要预测相邻车辆，添加相应的指令
        if predict_neighbors and neighbor_vehicles:
            instruction += (
                f"\n7. 基于车辆 {vehicle_id} 的预测轨迹和与相邻车辆的关系，预测以下相邻车辆的轨迹：\n"
                f"   - 相邻车辆IDs: {', '.join(neighbor_vehicles)}\n"
                f"8. 考虑车辆间的相对位置、距离、车道关系等因素\n"
                f"9. 确保相邻车辆的预测轨迹与主车辆的预测轨迹在空间和时间上保持合理的关系\n"
                f"10. 对于每个相邻车辆，提供与主车辆相同步数的轨迹预测和概率分布\n"
            )
            
        instruction += ("\n\n"
            f"CRITICAL: 返回格式要求：\n"
            f"- 只返回纯JSON，不要任何其他文本\n"
            f"- 不要markdown标记（如```json）\n"
            f"- 确保JSON格式完整，不要截断\n"
            f"- 所有字符串用双引号包围\n"
            f"- 数组格式正确，包含完整的 {steps} 个预测点\n"
            f"- 如果无法生成完整预测，至少返回基本字段（vehicle_id, prediction_steps等）\n\n"
            f"示例格式：\n"
            f'{{"vehicle_id": "{vehicle_id}", "prediction_steps": {steps}, ...}}'
        )

        # 调用MCP Agent进行预测
        try:
            logger.info(f"使用MCP Agent预测车辆 {vehicle_id} 的轨迹和概率分布...")
            result = self.agent.extract_information_direct(
                instruction=instruction,
                text=prediction_text,
                schema_obj=schema
            )
            
            # 检查MCP agent是否成功
            if result.get("fallback", False) or result.get("error"):
                logger.warning(f"MCP agent提取失败: {result.get('error', 'Unknown error')}")
                logger.info("使用传统算法生成预测结果和概率分布")
                result = self._generate_traditional_prediction_with_distribution(vehicle_data, vehicle_id, steps)
            else:
                logger.info("MCP agent成功生成预测结果和概率分布")
                
                # 如果MCP没有生成概率分布，使用传统方法补充
                if 'position_distributions' not in result or not result['position_distributions']:
                    logger.warning("MCP agent未生成位置概率分布，使用传统方法补充")
                    self._add_position_distributions(result, steps)
                    
                if 'speed_distributions' not in result or not result['speed_distributions']:
                    logger.warning("MCP agent未生成速度概率分布，使用传统方法补充")
                    self._add_speed_distributions(result, steps)
            
            return result
            
        except Exception as e:
            logger.error(f"MCP预测失败: {e}")
            # 使用传统算法作为备选
            logger.info("使用传统算法生成预测结果和概率分布")
            return self._generate_traditional_prediction_with_distribution(vehicle_data, vehicle_id, steps)

    def _add_position_distributions(self, result: Dict[str, Any], steps: int) -> Dict[str, Any]:
        """
        为预测结果添加位置概率分布

        Args:
            result: 预测结果
            steps: 预测步数

        Returns:
            添加了位置概率分布的预测结果
        """
        if 'position_distributions' not in result:
            result['position_distributions'] = []
            
        # 确保predicted_positions存在
        if 'predicted_positions' not in result or not result['predicted_positions']:
            return result
            
        # 获取现有分布步数
        existing_steps = set(dist['step'] for dist in result['position_distributions']) if result['position_distributions'] else set()
        
        # 为每个缺失的步数添加分布
        for i in range(steps):
            step_num = i + 1
            if step_num in existing_steps:
                continue
                
            # 查找对应步数的位置预测
            position = next((pos for pos in result['predicted_positions'] if pos['step'] == step_num), None)
            if not position:
                continue
                
            # 获取预测位置
            predicted_x = position['local_x']
            predicted_y = position['local_y']
            
            # 计算标准差（随步数增加而增大）
            x_std = 0.5 * (1 + 0.2 * step_num)
            y_std = 0.5 * (1 + 0.2 * step_num)
            
            # 生成采样点
            num_samples = 10
            x_samples = np.linspace(predicted_x - 3*x_std, predicted_x + 3*x_std, num_samples)
            y_samples = np.linspace(predicted_y - 3*y_std, predicted_y + 3*y_std, num_samples)
            
            # 计算概率密度
            x_pdf = stats.norm.pdf(x_samples, predicted_x, x_std)
            y_pdf = stats.norm.pdf(y_samples, predicted_y, y_std)
            
            # 归一化
            x_pdf = x_pdf / np.max(x_pdf)
            y_pdf = y_pdf / np.max(y_pdf)
            
            # 添加分布
            result['position_distributions'].append({
                "step": step_num,
                "distribution_type": "gaussian",
                "x_distribution": {
                    "mean": float(predicted_x),
                    "std": float(x_std),
                    "samples": [float(x) for x in x_samples.tolist()],
                    "densities": [float(p) for p in x_pdf.tolist()]
                },
                "y_distribution": {
                    "mean": float(predicted_y),
                    "std": float(y_std),
                    "samples": [float(y) for y in y_samples.tolist()],
                    "densities": [float(p) for p in y_pdf.tolist()]
                }
            })
            
        return result

    def _add_speed_distributions(self, result: Dict[str, Any], steps: int) -> Dict[str, Any]:
        """
        为预测结果添加速度概率分布

        Args:
            result: 预测结果
            steps: 预测步数

        Returns:
            添加了速度概率分布的预测结果
        """
        if 'speed_distributions' not in result:
            result['speed_distributions'] = []
            
        # 确保predicted_speeds存在
        if 'predicted_speeds' not in result or not result['predicted_speeds']:
            return result
            
        # 获取现有分布步数
        existing_steps = set(dist['step'] for dist in result['speed_distributions']) if result['speed_distributions'] else set()
        
        # 为每个缺失的步数添加分布
        for i in range(steps):
            step_num = i + 1
            if step_num in existing_steps:
                continue
                
            # 查找对应步数的速度预测
            speed_pred = next((spd for spd in result['predicted_speeds'] if spd['step'] == step_num), None)
            if not speed_pred:
                continue
                
            # 获取预测速度
            predicted_speed = speed_pred['speed']
            
            # 计算标准差（随步数增加而增大）
            speed_std = 1.0 * (1 + 0.1 * step_num)
            
            # 生成采样点
            num_samples = 10
            speed_samples = np.linspace(max(0, predicted_speed - 3*speed_std), predicted_speed + 3*speed_std, num_samples)
            
            # 计算概率密度
            speed_pdf = stats.norm.pdf(speed_samples, predicted_speed, speed_std)
            
            # 归一化
            speed_pdf = speed_pdf / np.max(speed_pdf)
            
            # 添加分布
            result['speed_distributions'].append({
                "step": step_num,
                "distribution_type": "gaussian",
                "speed_distribution": {
                    "mean": float(predicted_speed),
                    "std": float(speed_std),
                    "samples": [float(s) for s in speed_samples.tolist()],
                    "densities": [float(p) for p in speed_pdf.tolist()]
                }
            })
            
        return result

    def predict_trajectory_with_distribution(self, vehicle_id: str, steps: int = None,
                                 semantic_json: str = None, features_json: str = None,
                                 predict_neighbors: bool = False) -> Dict[str, Any]:
        """
        预测车辆轨迹并生成概率分布

        Args:
            vehicle_id: 车辆ID
            steps: 预测步数（如果为None，则使用真实轨迹的数据点数量）
            semantic_json: 语义分析结果文件路径
            features_json: 特征提取结果文件路径
            predict_neighbors: 是否预测相邻车辆

        Returns:
            带概率分布的预测结果
        """
        # 准备车辆数据
        vehicle_data = self.prepare_vehicle_data(vehicle_id)
        
        # 如果未指定steps，使用真实轨迹的数据点数量
        if steps is None:
            steps = len(vehicle_data)
            logger.info(f"未指定预测步数，使用真实轨迹的数据点数量: {steps}")

        logger.info(f"车辆 {vehicle_id} 真实轨迹有 {len(vehicle_data)} 个数据点，将生成 {steps} 个预测点")

        # 加载特征提取结果
        if features_json and os.path.exists(features_json):
            try:
                with open(features_json, 'r', encoding='utf-8') as f:
                    self.feature_results = json.load(f)
                logger.info(f"成功加载特征提取结果: {features_json}")
            except Exception as e:
                logger.warning(f"加载特征提取结果失败: {e}")
                self.feature_results = {}

        # 使用MCP Agent预测轨迹并生成概率分布
        result = self._predict_trajectory_with_mcp_distribution(vehicle_data, vehicle_id, steps, predict_neighbors)
        
        # 验证预测点数量
        if 'predicted_positions' not in result or len(result['predicted_positions']) != steps:
            logger.warning(f"预测位置点数量不匹配，期望 {steps} 个点，实际 {len(result.get('predicted_positions', []))} 个点")
            # 使用传统方法生成完整预测
            if 'predicted_positions' not in result or not result['predicted_positions']:
                result = self._generate_traditional_prediction_with_distribution(vehicle_data, vehicle_id, steps)
            else:
                # 补充或截断预测点
                self._adjust_prediction_points(result, steps)
                
        return result
    
    def _adjust_prediction_points(self, result: Dict[str, Any], steps: int) -> None:
        """调整预测点数量，确保与要求的步数一致

        Args:
            result: 预测结果
            steps: 预测步数
        """
        # 调整位置预测点
        if 'predicted_positions' in result:
            current_steps = len(result['predicted_positions'])
            if current_steps < steps:
                # 补充预测点
                logger.info(f"补充位置预测点: {current_steps} -> {steps}")
                last_pos = result['predicted_positions'][-1] if result['predicted_positions'] else {
                    "step": 0, "local_x": 0.0, "local_y": 0.0, "confidence": 0.5
                }
                
                for i in range(current_steps, steps):
                    # 简单外推
                    if current_steps > 1 and i > 0:
                        prev_pos = result['predicted_positions'][i-1]
                        if i > 1:
                            prev_prev_pos = result['predicted_positions'][i-2]
                            dx = prev_pos['local_x'] - prev_prev_pos['local_x']
                            dy = prev_pos['local_y'] - prev_prev_pos['local_y']
                        else:
                            # 使用最后两个点的差值
                            dx = result['predicted_positions'][-1]['local_x'] - result['predicted_positions'][-2]['local_x']
                            dy = result['predicted_positions'][-1]['local_y'] - result['predicted_positions'][-2]['local_y']
                            
                        new_pos = {
                            "step": i + 1,
                            "local_x": float(prev_pos['local_x'] + dx),
                            "local_y": float(prev_pos['local_y'] + dy),
                            "confidence": max(0.1, prev_pos['confidence'] - 0.05)
                        }
                    else:
                        # 简单线性外推
                        new_pos = {
                            "step": i + 1,
                            "local_x": float(last_pos['local_x'] + (i - current_steps + 1) * 2.0),
                            "local_y": float(last_pos['local_y'] + (i - current_steps + 1) * 1.5),
                            "confidence": max(0.1, 0.5 - (i - current_steps) * 0.05)
                        }
                        
                    result['predicted_positions'].append(new_pos)
            elif current_steps > steps:
                # 截断预测点
                logger.info(f"截断位置预测点: {current_steps} -> {steps}")
                result['predicted_positions'] = result['predicted_positions'][:steps]
                
        # 调整速度预测点
        if 'predicted_speeds' in result:
            current_steps = len(result['predicted_speeds'])
            if current_steps < steps:
                # 补充预测点
                logger.info(f"补充速度预测点: {current_steps} -> {steps}")
                last_speed = result['predicted_speeds'][-1] if result['predicted_speeds'] else {
                    "step": 0, "speed": 0.0, "confidence": 0.5
                }
                
                for i in range(current_steps, steps):
                    # 简单外推
                    if current_steps > 1 and i > 0:
                        prev_speed = result['predicted_speeds'][i-1]
                        if i > 1:
                            prev_prev_speed = result['predicted_speeds'][i-2]
                            dv = prev_speed['speed'] - prev_prev_speed['speed']
                        else:
                            # 使用最后两个点的差值
                            dv = result['predicted_speeds'][-1]['speed'] - result['predicted_speeds'][-2]['speed']
                            
                        new_speed = {
                            "step": i + 1,
                            "speed": max(0.0, float(prev_speed['speed'] + dv)),
                            "confidence": max(0.1, prev_speed['confidence'] - 0.05)
                        }
                    else:
                        # 简单线性外推
                        new_speed = {
                            "step": i + 1,
                            "speed": max(0.0, float(last_speed['speed'] + (i - current_steps + 1) * 0.5)),
                            "confidence": max(0.1, 0.5 - (i - current_steps) * 0.05)
                        }
                        
                    result['predicted_speeds'].append(new_speed)
            elif current_steps > steps:
                # 截断预测点
                logger.info(f"截断速度预测点: {current_steps} -> {steps}")
                result['predicted_speeds'] = result['predicted_speeds'][:steps]
                
        # 调整位置分布
        if 'position_distributions' in result:
            current_steps = len(result['position_distributions'])
            if current_steps != steps:
                # 重新生成位置分布
                self._add_position_distributions(result, steps)
                
        # 调整速度分布
        if 'speed_distributions' in result:
            current_steps = len(result['speed_distributions'])
            if current_steps != steps:
                # 重新生成速度分布
                self._add_speed_distributions(result, steps)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='US101轨迹概率分布预测')
    parser.add_argument('--csv', type=str, required=True, help='US101 CSV文件路径')
    parser.add_argument('--vehicle_id', type=str, required=True, help='车辆ID')
    parser.add_argument('--api_key', type=str, required=True, help='阿里云DashScope API密钥')
    parser.add_argument('--model', type=str, default='qwen-turbo', help='模型名称')
    parser.add_argument('--steps', type=int, default=None, help='预测步数')
    parser.add_argument('--output', type=str, default=None, help='输出JSON文件路径')
    parser.add_argument('--features', type=str, default=None, help='特征提取结果文件路径')
    parser.add_argument('--semantic', type=str, default=None, help='语义分析结果文件路径')
    parser.add_argument('--neighbors', action='store_true', help='是否预测相邻车辆')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = US101TrajectoryDistributionPredictor(
        csv_path=args.csv,
        api_key=args.api_key,
        model=args.model
    )
    
    # 加载数据
    predictor.load_data()
    
    # 预测轨迹
    result = predictor.predict_trajectory_with_distribution(
        vehicle_id=args.vehicle_id,
        steps=args.steps,
        semantic_json=args.semantic,
        features_json=args.features,
        predict_neighbors=args.neighbors
    )
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"预测结果已保存到: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()