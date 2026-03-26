#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from random import random

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AliyunMCPClient:
    """阿里云MCP客户端"""

    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        """
        初始化MCP客户端

        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model
        self.mock_mode = True  # 模拟模式，用于开发测试

        # 尝试导入dashscope
        try:
            import dashscope
            dashscope.api_key = api_key
            from dashscope.mcp import MCP
            self.mcp = MCP(model=model)
            self.mock_mode = False
            logger.info(f"成功初始化阿里云MCP客户端，使用模型: {model}")
        except ImportError:
            logger.warning("未安装dashscope包，将使用模拟模式")
        except Exception as e:
            logger.warning(f"初始化阿里云MCP客户端失败: {e}，将使用模拟模式")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        调用MCP API进行对话

        Args:
            messages: 消息列表
            temperature: 温度参数

        Returns:
            响应文本
        """
        if self.mock_mode:
            logger.info("使用模拟模式生成响应")
            return self._generate_mock_response(messages)

        try:
            response = self.mcp.call(
                messages=messages,
                temperature=temperature
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.error(f"API调用失败: {response.code}, {response.message}")
                return json.dumps({"error": f"API调用失败: {response.code}, {response.message}", "fallback": True})
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return json.dumps({"error": f"API调用异常: {e}", "fallback": True})

    def _generate_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """
        生成模拟响应

        Args:
            messages: 消息列表

        Returns:
            模拟响应文本
        """
        # 提取用户消息
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # 提取车辆ID和步数
        vehicle_id = "2"  # 默认值
        steps = 10  # 默认值
        import re
        vehicle_match = re.search(r"车辆\s*([\d]+)", user_message)
        if vehicle_match:
            vehicle_id = vehicle_match.group(1)

        steps_match = re.search(r"(\d+)\s*步", user_message)
        if steps_match:
            steps = int(steps_match.group(1))

        # 生成模拟预测数据
        positions = []
        speeds = []
        for i in range(steps):
            positions.append({
                "step": i + 1,
                "local_x": 100.0 + i * 5.0,
                "local_y": 200.0 + i * 2.0,
                "confidence": max(0.1, 0.9 - i * 0.05)
            })
            speeds.append({
                "step": i + 1,
                "speed": 20.0 + i * 0.5,
                "confidence": max(0.1, 0.9 - i * 0.05)
            })

        # 生成模拟响应
        response = {
            "vehicle_id": vehicle_id,
            "prediction_steps": steps,
            "predicted_positions": positions,
            "predicted_speeds": speeds,
            "predicted_lane": "lane_1",
            "trajectory_pattern": "直线加速",
            "prediction_confidence": 0.85,
            "reasoning": "基于历史轨迹数据分析，车辆呈现稳定加速模式，预计将继续沿当前车道直线行驶。"
        }

        return json.dumps(response, ensure_ascii=False)


class MCPInformationExtractor:
    """MCP信息提取器"""

    def __init__(self, mcp: AliyunMCPClient):
        """
        初始化信息提取器

        Args:
            mcp: MCP客户端实例
        """
        self.mcp = mcp

    def extract_information(self, instruction: str, text: str, examples: str = "",
                           schema: str = "", additional_info: str = "") -> Dict[str, Any]:
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
                    "You are an expert traffic trajectory prediction MCP agent. "
                    "Extract information according to the instruction and schema. "
                    "Return valid JSON."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{instruction}\n\n"
                    f"Text:\n{text}\n\n"
                    f"Examples:\n{examples}\n\n"
                    f"JSON Schema:\n{schema}\n\n"
                    f"Additional Info:\n{additional_info}"
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

    def _extract_json_dict(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取JSON字典

        Args:
            text: 输入文本

        Returns:
            提取的JSON字典
        """
        try:
            # 尝试直接解析JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试从文本中提取JSON部分
            import re
            json_pattern = r'\{[\s\S]*\}'  # 匹配最外层的花括号及其内容
            match = re.search(json_pattern, text)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

            # 如果仍然失败，返回错误信息
            logger.error(f"无法解析JSON: {text[:100]}...")
            return {"error": "无法解析JSON", "fallback": True}


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


class VehicleRelationshipAnalyzer:
    """车辆关系分析器"""

    def __init__(self, data: pd.DataFrame):
        """
        初始化车辆关系分析器

        Args:
            data: 包含多车数据的DataFrame
        """
        self.data = data
        self.vehicle_ids = self._extract_vehicle_ids()
        self.relationships = {}
        self.neighbor_map = {}

    def _extract_vehicle_ids(self) -> List[str]:
        """
        从数据中提取所有车辆ID

        Returns:
            车辆ID列表
        """
        # 查找车辆ID列
        vehicle_col = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['vehicle', 'id', 'veh']):
                vehicle_col = col
                break

        if vehicle_col is None:
            raise ValueError("未找到车辆ID列")

        # 提取唯一车辆ID
        vehicle_ids = sorted(self.data[vehicle_col].unique())
        logger.info(f"数据集中共有 {len(vehicle_ids)} 辆车")
        return [str(vid) for vid in vehicle_ids]

    def analyze_relationships(self) -> Dict[str, Dict[str, Any]]:
        """
        分析所有车辆之间的关系

        Returns:
            车辆关系字典
        """
        logger.info("开始分析车辆之间的关系...")

        # 查找位置列
        location_cols = []
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['local_x', 'local_y', 'global_x', 'global_y', 'lat', 'lon']):
                location_cols.append(col)

        if len(location_cols) < 2:
            raise ValueError("未找到足够的位置列")

        x_col, y_col = location_cols[0], location_cols[1]

        # 查找时间列
        time_col = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['time', 'frame', 'timestamp']):
                time_col = col
                break

        if time_col is None:
            raise ValueError("未找到时间列")

        # 查找车辆ID列
        vehicle_col = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['vehicle', 'id', 'veh']):
                vehicle_col = col
                break

        if vehicle_col is None:
            raise ValueError("未找到车辆ID列")

        # 查找车道列
        lane_col = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['lane', 'road']):
                lane_col = col
                break

        # 分析每个时间点的车辆位置关系
        time_points = sorted(self.data[time_col].unique())
        
        # 初始化关系字典
        for vid in self.vehicle_ids:
            self.relationships[vid] = {
                "neighbors": {},
                "avg_distance": {},
                "time_overlap": {},
                "same_lane_percentage": {},
                "relative_position": {}
            }
            self.neighbor_map[vid] = []

        # 对每个时间点分析车辆位置
        for time_point in time_points:
            time_data = self.data[self.data[time_col] == time_point]
            vehicles_at_time = time_data[vehicle_col].unique()
            
            # 分析每对车辆之间的关系
            for i, vid1 in enumerate(vehicles_at_time):
                for vid2 in vehicles_at_time[i+1:]:
                    v1_data = time_data[time_data[vehicle_col] == vid1]
                    v2_data = time_data[time_data[vehicle_col] == vid2]
                    
                    if len(v1_data) == 0 or len(v2_data) == 0:
                        continue
                    
                    # 计算距离
                    x1, y1 = float(v1_data[x_col].iloc[0]), float(v1_data[y_col].iloc[0])
                    x2, y2 = float(v2_data[x_col].iloc[0]), float(v2_data[y_col].iloc[0])
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # 更新关系数据
                    vid1_str, vid2_str = str(vid1), str(vid2)
                    
                    # 更新距离统计
                    if vid2_str not in self.relationships[vid1_str]["avg_distance"]:
                        self.relationships[vid1_str]["avg_distance"][vid2_str] = []
                    self.relationships[vid1_str]["avg_distance"][vid2_str].append(distance)
                    
                    if vid1_str not in self.relationships[vid2_str]["avg_distance"]:
                        self.relationships[vid2_str]["avg_distance"][vid1_str] = []
                    self.relationships[vid2_str]["avg_distance"][vid1_str].append(distance)
                    
                    # 更新时间重叠
                    if vid2_str not in self.relationships[vid1_str]["time_overlap"]:
                        self.relationships[vid1_str]["time_overlap"][vid2_str] = 0
                    self.relationships[vid1_str]["time_overlap"][vid2_str] += 1
                    
                    if vid1_str not in self.relationships[vid2_str]["time_overlap"]:
                        self.relationships[vid2_str]["time_overlap"][vid1_str] = 0
                    self.relationships[vid2_str]["time_overlap"][vid1_str] += 1
                    
                    # 更新相对位置（前后关系）
                    if x1 > x2:  # 假设x轴是行驶方向
                        rel_pos = "前方"
                    else:
                        rel_pos = "后方"
                    
                    if vid2_str not in self.relationships[vid1_str]["relative_position"]:
                        self.relationships[vid1_str]["relative_position"][vid2_str] = {"前方": 0, "后方": 0}
                    self.relationships[vid1_str]["relative_position"][vid2_str][rel_pos] += 1
                    
                    opposite_rel_pos = "后方" if rel_pos == "前方" else "前方"
                    if vid1_str not in self.relationships[vid2_str]["relative_position"]:
                        self.relationships[vid2_str]["relative_position"][vid1_str] = {"前方": 0, "后方": 0}
                    self.relationships[vid2_str]["relative_position"][vid1_str][opposite_rel_pos] += 1
                    
                    # 更新车道关系（如果有车道信息）
                    if lane_col:
                        lane1 = v1_data[lane_col].iloc[0]
                        lane2 = v2_data[lane_col].iloc[0]
                        same_lane = lane1 == lane2
                        
                        if vid2_str not in self.relationships[vid1_str]["same_lane_percentage"]:
                            self.relationships[vid1_str]["same_lane_percentage"][vid2_str] = []
                        self.relationships[vid1_str]["same_lane_percentage"][vid2_str].append(1 if same_lane else 0)
                        
                        if vid1_str not in self.relationships[vid2_str]["same_lane_percentage"]:
                            self.relationships[vid2_str]["same_lane_percentage"][vid1_str] = []
                        self.relationships[vid2_str]["same_lane_percentage"][vid1_str].append(1 if same_lane else 0)

        # 计算平均值和百分比
        for vid1 in self.vehicle_ids:
            for vid2 in self.relationships[vid1]["avg_distance"].keys():
                # 平均距离
                distances = self.relationships[vid1]["avg_distance"][vid2]
                self.relationships[vid1]["avg_distance"][vid2] = sum(distances) / len(distances) if distances else float('inf')
                
                # 同车道百分比
                if vid2 in self.relationships[vid1]["same_lane_percentage"]:
                    same_lane_values = self.relationships[vid1]["same_lane_percentage"][vid2]
                    self.relationships[vid1]["same_lane_percentage"][vid2] = sum(same_lane_values) / len(same_lane_values) if same_lane_values else 0
                
                # 相对位置主导关系
                if vid2 in self.relationships[vid1]["relative_position"]:
                    rel_pos = self.relationships[vid1]["relative_position"][vid2]
                    if rel_pos["前方"] > rel_pos["后方"]:
                        self.relationships[vid1]["relative_position"][vid2] = "前方"
                    else:
                        self.relationships[vid1]["relative_position"][vid2] = "后方"
            
            # 确定邻居车辆（平均距离最近的几辆车）
            neighbors = sorted(
                [(vid2, dist) for vid2, dist in self.relationships[vid1]["avg_distance"].items()],
                key=lambda x: x[1]
            )
            
            # 取前5个最近的车辆作为邻居
            self.neighbor_map[vid1] = [n[0] for n in neighbors[:5] if n[1] < float('inf')]
            self.relationships[vid1]["neighbors"] = {n: {"distance": d} for n, d in neighbors[:5] if d < float('inf')}
            
            # 添加邻居的其他关系信息
            for n in self.relationships[vid1]["neighbors"]:
                if n in self.relationships[vid1]["time_overlap"]:
                    self.relationships[vid1]["neighbors"][n]["time_overlap"] = self.relationships[vid1]["time_overlap"][n]
                if n in self.relationships[vid1]["relative_position"]:
                    self.relationships[vid1]["neighbors"][n]["relative_position"] = self.relationships[vid1]["relative_position"][n]
                if n in self.relationships[vid1]["same_lane_percentage"]:
                    self.relationships[vid1]["neighbors"][n]["same_lane_percentage"] = self.relationships[vid1]["same_lane_percentage"][n]

        logger.info("车辆关系分析完成")
        return self.relationships

    def get_neighbors(self, vehicle_id: str) -> List[str]:
        """
        获取指定车辆的邻居车辆ID列表

        Args:
            vehicle_id: 车辆ID

        Returns:
            邻居车辆ID列表
        """
        if not self.neighbor_map:
            self.analyze_relationships()
            
        return self.neighbor_map.get(vehicle_id, [])

    def get_relationship_details(self, vehicle_id1: str, vehicle_id2: str) -> Dict[str, Any]:
        """
        获取两辆车之间的详细关系

        Args:
            vehicle_id1: 第一辆车ID
            vehicle_id2: 第二辆车ID

        Returns:
            关系详情字典
        """
        if not self.relationships:
            self.analyze_relationships()
            
        if vehicle_id1 not in self.relationships or vehicle_id2 not in self.relationships[vehicle_id1]["avg_distance"]:
            return {}
            
        return {
            "avg_distance": self.relationships[vehicle_id1]["avg_distance"][vehicle_id2],
            "time_overlap": self.relationships[vehicle_id1]["time_overlap"].get(vehicle_id2, 0),
            "relative_position": self.relationships[vehicle_id1]["relative_position"].get(vehicle_id2, "未知"),
            "same_lane_percentage": self.relationships[vehicle_id1]["same_lane_percentage"].get(vehicle_id2, 0)
        }


class US101TrajectoryPredictor:
    """US101数据集轨迹预测器"""

    def __init__(self, csv_path: str, api_key: str, model: str = "qwen-turbo"):
        """
        初始化轨迹预测器

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
        self.relationship_analyzer = None

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

            # 如果所有编码都失败，使用错误处理
            self.data = pd.read_csv(self.csv_path, encoding='utf-8', errors='ignore')
            logger.warning("使用错误忽略模式加载数据")
            return self.data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def prepare_vehicle_data(self, vehicle_id: str) -> pd.DataFrame:
        """
        准备指定车辆的数据

        Args:
            vehicle_id: 车辆ID

        Returns:
            车辆数据DataFrame
        """
        if self.data is None:
            self.load_data()

        # 查找车辆ID列
        vehicle_col = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['vehicle', 'id', 'veh']):
                vehicle_col = col
                break

        if vehicle_col is None:
            raise ValueError("未找到车辆ID列")

        # 筛选指定车辆的数据 - 使用多种匹配方式
        vehicle_data = None

        # 1. 尝试精确匹配
        vehicle_data = self.data[self.data[vehicle_col] == vehicle_id].copy()

        # 2. 如果精确匹配失败，尝试浮点数匹配
        if vehicle_data.empty:
            try:
                float_id = float(vehicle_id)
                vehicle_data = self.data[self.data[vehicle_col] == float_id].copy()
                logger.info(f"使用浮点数匹配找到车辆ID {vehicle_id}")
            except (ValueError, TypeError):
                pass

        # 3. 如果浮点数匹配失败，尝试字符串匹配
        if vehicle_data.empty:
            try:
                vehicle_data = self.data[self.data[vehicle_col].astype(str) == str(vehicle_id)].copy()
                logger.info(f"使用字符串匹配找到车辆ID {vehicle_id}")
            except Exception:
                pass

        # 4. 如果字符串匹配失败，尝试近似匹配（考虑浮点精度）
        if vehicle_data.empty:
            try:
                import numpy as np
                float_id = float(vehicle_id)
                vehicle_data = self.data[np.isclose(self.data[vehicle_col], float_id, atol=1e-10)].copy()
                logger.info(f"使用近似匹配找到车辆ID {vehicle_id}")
            except Exception:
                pass

        # 5. 如果所有匹配都失败，提供详细的错误信息
        if vehicle_data.empty:
            # 显示可用的车辆ID
            available_ids = sorted(self.data[vehicle_col].dropna().unique())[:10]
            error_msg = f"未找到车辆ID为 {vehicle_id} 的数据\n"
            error_msg += f"可用的车辆ID示例: {available_ids}\n"
            error_msg += f"车辆ID列 '{vehicle_col}' 的数据类型: {self.data[vehicle_col].dtype}\n"
            error_msg += f"数据范围: {self.data[vehicle_col].min()} 到 {self.data[vehicle_col].max()}"
            raise ValueError(error_msg)

        # 尝试按时间排序
        time_col = None
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['time', 'frame', 'timestamp']):
                time_col = col
                break

        if time_col:
            try:
                vehicle_data[time_col] = pd.to_datetime(vehicle_data[time_col], errors='coerce')
                vehicle_data = vehicle_data.sort_values(time_col)
            except Exception as e:
                logger.warning(f"时间排序失败: {e}")

        return vehicle_data

    def extract_trajectory_features(self, vehicle_data: pd.DataFrame) -> Dict[str, Any]:
        """
        提取轨迹特征

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

    def predict_trajectory(self, vehicle_id: str, steps: int = None,
                           semantic_json: str = None, features_json: str = None,
                           predict_neighbors: bool = False) -> Dict[str, Any]:
        """
        预测车辆轨迹，可选择是否同时预测相邻车辆的轨迹

        Args:
            vehicle_id: 车辆ID
            steps: 预测步数（如果为None，则使用真实轨迹的数据点数量）
            semantic_json: 语义分析结果文件路径
            features_json: 特征提取结果文件路径
            predict_neighbors: 是否同时预测相邻车辆的轨迹

        Returns:
            预测结果，如果predict_neighbors为True，则包含相邻车辆的预测结果
        """
        # 准备车辆数据
        vehicle_data = self.prepare_vehicle_data(vehicle_id)
        trajectory_features = self.extract_trajectory_features(vehicle_data)
        
        # 初始化关系分析器（如果尚未初始化）
        if self.relationship_analyzer is None and predict_neighbors:
            self.relationship_analyzer = VehicleRelationshipAnalyzer(self.data)

        # 如果未指定steps，使用真实轨迹的数据点数量
        if steps is None:
            steps = len(vehicle_data)
            logger.info(f"未指定预测步数，使用真实轨迹的数据点数量: {steps}")

        logger.info(f"车辆 {vehicle_id} 真实轨迹有 {len(vehicle_data)} 个数据点，将生成 {steps} 个预测点")

        # 加载特征提取结果
        features_data = {}
        if features_json and os.path.exists(features_json):
            try:
                with open(features_json, 'r', encoding='utf-8') as f:
                    features_data = json.load(f)
                logger.info(f"成功加载特征提取结果: {features_json}")
            except Exception as e:
                logger.warning(f"加载特征提取结果失败: {e}")

        # 加载语义分析结果
        semantic_data = {}
        if semantic_json and os.path.exists(semantic_json):
            try:
                with open(semantic_json, 'r', encoding='utf-8') as f:
                    semantic_data = json.load(f)
                logger.info(f"成功加载语义分析结果: {semantic_json}")
            except Exception as e:
                logger.warning(f"加载语义分析结果失败: {e}")

        # 构建增强的预测输入文本
        prediction_text = self._build_enhanced_prediction_text(
            vehicle_data, trajectory_features, features_data, semantic_data
        )
        
        # 获取相邻车辆信息（如果需要）
        neighbor_info = {}
        neighbor_vehicles = []
        if predict_neighbors and self.relationship_analyzer:
            neighbor_vehicles = self.relationship_analyzer.get_neighbors(vehicle_id)
            if neighbor_vehicles:
                neighbor_info["neighbor_vehicles"] = neighbor_vehicles
                neighbor_info["relationships"] = {}
                for neighbor_id in neighbor_vehicles:
                    relationship = self.relationship_analyzer.get_relationship_details(vehicle_id, neighbor_id)
                    if relationship:
                        neighbor_info["relationships"][neighbor_id] = relationship
                
                # 将相邻车辆信息添加到预测文本中
                prediction_text += f"\n\n相邻车辆信息:\n"
                prediction_text += f"该车辆有 {len(neighbor_vehicles)} 个相邻车辆: {', '.join(neighbor_vehicles)}\n"
                for neighbor_id in neighbor_vehicles:
                    rel = neighbor_info["relationships"].get(neighbor_id, {})
                    if rel:
                        prediction_text += f"与车辆 {neighbor_id} 的关系:\n"
                        prediction_text += f"- 平均距离: {rel.get('avg_distance', '未知')}\n"
                        prediction_text += f"- 时间重叠: {rel.get('time_overlap', '未知')}\n"
                        prediction_text += f"- 相对位置: {rel.get('relative_position', '未知')}\n"
                        prediction_text += f"- 同车道百分比: {rel.get('same_lane_percentage', '未知')}\n"

        # 定义增强的输出模式 - 确保预测点数量与steps一致
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
            "predicted_speeds": [
                {
                    "step": f"integer (1 to {steps})",
                    "speed": "float (predicted speed)",
                    "confidence": "float (0.0 to 1.0, prediction confidence)"
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
                    "predicted_speeds": [
                        {
                            "step": f"integer (1 to {steps})",
                            "speed": "float (predicted speed)",
                            "confidence": "float (0.0 to 1.0, prediction confidence)"
                        }
                    ],
                    "prediction_confidence": "float (overall prediction confidence 0.0 to 1.0)"
                }

        instruction = (
            f"基于提供的车辆轨迹数据和语义分析结果，使用MCP Agent智能预测车辆 {vehicle_id} 未来 {steps} 步的轨迹。\n\n"
            f"关键要求：\n"
            f"1. 真实轨迹有 {len(vehicle_data)} 个数据点，请生成恰好 {steps} 个预测点\n"
            f"2. 充分利用特征提取结果中的时空特征、统计特征等\n"
            f"3. 结合语义分析结果中的交通模式、异常检测、安全评估等\n"
            f"4. 考虑车辆的历史运动模式、速度变化、加速度特征、车道信息\n"
            f"5. 给出合理的位置、速度和车道预测，并说明预测依据\n"
            f"6. 分析特征和语义信息如何影响预测结果\n"
        )
        
        # 如果需要预测相邻车辆，添加相应的指令
        if predict_neighbors and neighbor_vehicles:
            instruction += (
                f"\n7. 基于车辆 {vehicle_id} 的预测轨迹和与相邻车辆的关系，预测以下相邻车辆的轨迹：\n"
                f"   - 相邻车辆IDs: {', '.join(neighbor_vehicles)}\n"
                f"8. 考虑车辆间的相对位置、距离、车道关系等因素\n"
                f"9. 确保相邻车辆的预测轨迹与主车辆的预测轨迹在空间和时间上保持合理的关系\n"
                f"10. 对于每个相邻车辆，提供与主车辆相同步数的轨迹预测\n"
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
            logger.info(f"使用MCP Agent预测车辆 {vehicle_id} 的轨迹...")
            result = self.agent.extract_information_direct(
                instruction=instruction,
                context=prediction_text,
                schema=schema
            )
            
            # 验证预测结果
            if not result or not isinstance(result, dict):
                logger.warning(f"MCP Agent返回的结果无效，使用传统算法预测")
                return self._generate_traditional_prediction(vehicle_data, vehicle_id, steps)
            
            return result
        except Exception as e:
            logger.error(f"MCP Agent预测失败: {e}")
            return self._generate_traditional_prediction(vehicle_data, vehicle_id, steps)
            
    def _generate_traditional_prediction(self, vehicle_data: pd.DataFrame, vehicle_id: str, steps: int) -> Dict[str, Any]:
        """
        使用传统算法生成预测结果（当MCP agent失败时的fallback）

        Args:
            vehicle_data: 车辆数据
            vehicle_id: 车辆ID
            steps: 预测步数

        Returns:
            预测结果字典
        """
        logger.info("使用传统算法生成轨迹预测")

        # 获取位置列
        location_cols = []
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['local_x', 'local_y', 'global_x', 'global_y', 'lat', 'lon']):
                location_cols.append(col)

        # 获取速度列
        speed_col = None
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['v_vel', 'speed', 'velocity']):
                speed_col = col
                break

        # 获取车道列
        lane_col = None
        for col in vehicle_data.columns:
            if any(keyword in col.lower() for keyword in ['lane', 'road']):
                lane_col = col
                break

        # 生成预测结果
        result = {
            "vehicle_id": vehicle_id,
            "prediction_steps": steps,
            "predicted_positions": [],
            "predicted_speeds": [],
            "predicted_lane": vehicle_data[lane_col].iloc[-1] if lane_col else "lane_1",
            "trajectory_pattern": "传统算法预测轨迹",
            "prediction_confidence": 0.7,
            "reasoning": "基于历史轨迹数据的统计分析和线性外推算法生成预测",
            "fallback": True,
            "fallback_method": "traditional_algorithm"
        }

        if location_cols:
            x_col, y_col = location_cols[0], location_cols[1]

            # 计算历史轨迹的趋势
            if len(vehicle_data) > 1:
                # 计算平均移动距离和方向
                dx = vehicle_data[x_col].diff().mean()
                dy = vehicle_data[y_col].diff().mean()

                # 获取最后一个真实位置
                last_x = vehicle_data[x_col].iloc[-1]
                last_y = vehicle_data[y_col].iloc[-1]

                # 生成位置预测
                for i in range(steps):
                    predicted_x = last_x + dx * (i + 1)
                    predicted_y = last_y + dy * (i + 1)

                    result["predicted_positions"].append({
                        "step": i + 1,
                        "local_x": float(predicted_x),
                        "local_y": float(predicted_y),
                        "confidence": max(0.1, 0.8 - i * 0.02)  # 置信度递减
                    })
            else:
                # 如果数据不足，使用默认值
                for i in range(steps):
                    result["predicted_positions"].append({
                        "step": i + 1,
                        "local_x": float(100 + i * 2.0),
                        "local_y": float(200 + i * 1.5),
                        "confidence": max(0.1, 0.8 - i * 0.02)
                    })

        if speed_col:
            # 计算历史速度趋势
            if len(vehicle_data) > 1:
                # 计算平均速度变化
                speed_change = vehicle_data[speed_col].diff().mean()
                last_speed = vehicle_data[speed_col].iloc[-1]

                # 生成速度预测
                for i in range(steps):
                    predicted_speed = last_speed + speed_change * (i + 1)
                    # 确保速度不为负数
                    predicted_speed = max(0.0, predicted_speed)

                    result["predicted_speeds"].append({
                        "step": i + 1,
                        "speed": float(predicted_speed),
                        "confidence": max(0.1, 0.8 - i * 0.02)
                    })
            else:
                # 如果数据不足，使用默认值
                for i in range(steps):
                    result["predicted_speeds"].append({
                        "step": i + 1,
                        "speed": float(25.0 + i * 0.3),
                        "confidence": max(0.1, 0.8 - i * 0.02)
                    })

        logger.info(f"传统算法成功生成 {len(result['predicted_positions'])} 个位置预测点和 {len(result['predicted_speeds'])} 个速度预测点")
        return result
    
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

    def _predict_trajectory_with_mcp(self, vehicle_data: pd.DataFrame, vehicle_id: str, steps: int = 10, predict_neighbors: bool = False) -> Dict[str, Any]:
        # 构建预测文本
        prediction_text = self._build_enhanced_prediction_text(vehicle_data, vehicle_id, predict_neighbors)
        
        # 获取相邻车辆ID（如果需要）
        neighbor_vehicles = []
        if predict_neighbors and 'neighbor_info' in self.feature_results:
            neighbor_vehicles = list(self.feature_results['neighbor_info'].get('relationships', {}).keys())
        
        # 定义增强的输出模式 - 确保预测点数量与steps一致
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
            "predicted_speeds": [
                {
                    "step": f"integer (1 to {steps})",
                    "speed": "float (predicted speed)",
                    "confidence": "float (0.0 to 1.0, prediction confidence)"
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
                    "predicted_speeds": [
                        {
                            "step": f"integer (1 to {steps})",
                            "speed": "float (predicted speed)",
                            "confidence": "float (0.0 to 1.0, prediction confidence)"
                        }
                    ],
                    "prediction_confidence": "float (overall prediction confidence 0.0 to 1.0)"
                }

        instruction = (
            f"基于提供的车辆轨迹数据和语义分析结果，使用MCP Agent智能预测车辆 {vehicle_id} 未来 {steps} 步的轨迹。\n\n"
            f"关键要求：\n"
            f"1. 真实轨迹有 {len(vehicle_data)} 个数据点，请生成恰好 {steps} 个预测点\n"
            f"2. 充分利用特征提取结果中的时空特征、统计特征等\n"
            f"3. 结合语义分析结果中的交通模式、异常检测、安全评估等\n"
            f"4. 考虑车辆的历史运动模式、速度变化、加速度特征、车道信息\n"
            f"5. 给出合理的位置、速度和车道预测，并说明预测依据\n"
            f"6. 分析特征和语义信息如何影响预测结果\n"
        )
        
        # 如果需要预测相邻车辆，添加相应的指令
        if predict_neighbors and neighbor_vehicles:
            instruction += (
                f"\n7. 基于车辆 {vehicle_id} 的预测轨迹和与相邻车辆的关系，预测以下相邻车辆的轨迹：\n"
                f"   - 相邻车辆IDs: {', '.join(neighbor_vehicles)}\n"
                f"8. 考虑车辆间的相对位置、距离、车道关系等因素\n"
                f"9. 确保相邻车辆的预测轨迹与主车辆的预测轨迹在空间和时间上保持合理的关系\n"
                f"10. 对于每个相邻车辆，提供与主车辆相同步数的轨迹预测\n"
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
            logger.info(f"使用MCP Agent预测车辆 {vehicle_id} 的轨迹...")
            result = self.agent.extract_information_direct(
                instruction=instruction,
                context=prediction_text,
                schema=schema
            )
            
            # 验证预测结果
            if not result or not isinstance(result, dict):
                logger.warning(f"MCP Agent返回的结果无效，使用传统算法预测")
                return self._generate_traditional_prediction(vehicle_data, vehicle_id, steps)
            
            return result
        except Exception as e:
            logger.error(f"MCP Agent预测失败: {e}")
            return self._generate_traditional_prediction(vehicle_data, vehicle_id, steps)
    
    def predict_trajectory(self, vehicle_data: pd.DataFrame, vehicle_id: str, steps: int = 10, predict_neighbors: bool = False) -> Dict[str, Any]:
        # 获取预测结果
        result = self._predict_trajectory_with_mcp(vehicle_data, vehicle_id, steps, predict_neighbors)
        
        # 验证预测点数量
        if 'predicted_positions' not in result or len(result['predicted_positions']) != steps:
            logger.warning(f"预测位置点数量不匹配，期望 {steps} 个点，实际 {len(result.get('predicted_positions', []))} 个点")
            result = self._extend_predictions(result, steps, None)
            
        if 'predicted_speeds' not in result or len(result['predicted_speeds']) != steps:
            logger.warning(f"预测速度点数量不匹配，期望 {steps} 个点，实际 {len(result.get('predicted_speeds', []))} 个点")
            result = self._extend_speed_predictions(result, steps, None)
                
        # 验证相邻车辆预测结果（如果需要）
        if predict_neighbors and 'neighbor_predictions' in result:
            for neighbor_id in result['neighbor_predictions']:
                if neighbor_id not in result['neighbor_predictions']:
                    logger.warning(f"缺少车辆 {neighbor_id} 的预测结果")
                    continue
                    
                neighbor_result = result['neighbor_predictions'][neighbor_id]
                
                # 验证相邻车辆的预测点数量
                if 'predicted_positions' not in neighbor_result or len(neighbor_result['predicted_positions']) != steps:
                    logger.warning(f"车辆 {neighbor_id} 的预测位置点数量不匹配，期望 {steps} 个点，实际 {len(neighbor_result.get('predicted_positions', []))} 个点")
                    # 使用与主车辆相同的扩展方法
                    neighbor_result = self._extend_predictions(neighbor_result, steps, None)
                    result['neighbor_predictions'][neighbor_id] = neighbor_result
                    
                if 'predicted_speeds' not in neighbor_result or len(neighbor_result['predicted_speeds']) != steps:
                    logger.warning(f"车辆 {neighbor_id} 的预测速度点数量不匹配，期望 {steps} 个点，实际 {len(neighbor_result.get('predicted_speeds', []))} 个点")
                    # 使用与主车辆相同的扩展方法
                    neighbor_result = self._extend_speed_predictions(neighbor_result, steps, None)
                    result['neighbor_predictions'][neighbor_id] = neighbor_result
        
        return result
            
    def _extend_predictions(self, result: Dict[str, Any], target_steps: int, vehicle_data: pd.DataFrame = None):
        """
        扩展预测结果到目标步数
        
        Args:
            result: 预测结果
            target_steps: 目标步数
            vehicle_data: 真实车辆数据
        """
        current_positions = result.get('predicted_positions', [])
        current_count = len(current_positions)
        
        if current_count >= target_steps:
            return result
        
        # 如果没有真实数据，使用预测数据进行外推
        if vehicle_data is None or len(vehicle_data) == 0:
            # 使用已有预测点进行外推
            if current_count >= 2:
                # 获取最后两个预测点
                last_point = current_positions[-1]
                second_last_point = current_positions[-2]
                
                # 计算移动趋势
                dx = last_point['local_x'] - second_last_point['local_x']
                dy = last_point['local_y'] - second_last_point['local_y']
                
                # 生成额外的预测点
                for i in range(current_count, target_steps):
                    # 基于趋势进行预测
                    predicted_x = last_point['local_x'] + dx * (i - current_count + 1)
                    predicted_y = last_point['local_y'] + dy * (i - current_count + 1)
                    
                    new_position = {
                        "step": i + 1,
                        "local_x": float(predicted_x),
                        "local_y": float(predicted_y),
                        "confidence": max(0.1, last_point.get('confidence', 0.5) * 0.8)  # 置信度递减
                    }
                    
                    current_positions.append(new_position)
            else:
                # 如果只有一个预测点，使用默认移动
                last_point = current_positions[-1] if current_positions else {"local_x": 0, "local_y": 0, "confidence": 0.5}
                
                for i in range(current_count, target_steps):
                    new_position = {
                        "step": i + 1,
                        "local_x": float(last_point['local_x'] + 0.1 * (i - current_count + 1)),
                        "local_y": float(last_point['local_y'] + 0.1 * (i - current_count + 1)),
                        "confidence": max(0.1, last_point.get('confidence', 0.5) * 0.7)  # 置信度递减
                    }
                    
                    current_positions.append(new_position)
        else:
            # 获取位置列
            location_cols = []
            for col in vehicle_data.columns:
                if any(keyword in col.lower() for keyword in ['local_x', 'local_y', 'global_x', 'global_y', 'lat', 'lon']):
                    location_cols.append(col)
            
            if len(location_cols) >= 2:
                x_col, y_col = location_cols[0], location_cols[1]
                
                # 基于真实轨迹的最后几个点进行扩展预测
                last_real_x = vehicle_data[x_col].iloc[-1]
                last_real_y = vehicle_data[y_col].iloc[-1]
                
                # 计算平均移动距离
                if len(vehicle_data) > 1:
                    dx = vehicle_data[x_col].diff().mean()
                    dy = vehicle_data[y_col].diff().mean()
                else:
                    dx, dy = 0.1, 0.1  # 默认值
                
                # 生成额外的预测点
                for i in range(current_count, target_steps):
                    # 基于历史趋势进行预测
                    predicted_x = last_real_x + dx * (i - current_count + 1)
                    predicted_y = last_real_y + dy * (i - current_count + 1)
                    
                    new_position = {
                        "step": i + 1,
                        "local_x": float(predicted_x),
                        "local_y": float(predicted_y),
                        "confidence": max(0.1, 1.0 - (i - current_count) * 0.1)  # 置信度递减
                    }
                    
                    current_positions.append(new_position)
            else:
                # 如果找不到位置列，使用默认移动
                last_point = current_positions[-1] if current_positions else {"local_x": 0, "local_y": 0, "confidence": 0.5}
                
                for i in range(current_count, target_steps):
                    new_position = {
                        "step": i + 1,
                        "local_x": float(last_point['local_x'] + 0.1 * (i - current_count + 1)),
                        "local_y": float(last_point['local_y'] + 0.1 * (i - current_count + 1)),
                        "confidence": max(0.1, last_point.get('confidence', 0.5) * 0.7)  # 置信度递减
                    }
                    
                    current_positions.append(new_position)
        
        result['predicted_positions'] = current_positions
        logger.info(f"扩展位置预测到 {len(current_positions)} 个点")
        return result
        
    def _extend_speed_predictions(self, result: Dict[str, Any], target_steps: int, vehicle_data: pd.DataFrame = None):
        """
        扩展速度预测结果到目标步数
        
        Args:
            result: 预测结果
            target_steps: 目标步数
            vehicle_data: 真实车辆数据
        """
        current_speeds = result.get('predicted_speeds', [])
        current_count = len(current_speeds)
        
        if current_count >= target_steps:
            return result
            
        # 如果没有真实数据，使用预测数据进行外推
        if vehicle_data is None or len(vehicle_data) == 0:
            # 使用已有预测点进行外推
            if current_count >= 2:
                # 获取最后两个预测点
                last_speed = current_speeds[-1]
                second_last_speed = current_speeds[-2]
                
                # 计算速度变化趋势
                speed_change = last_speed['speed'] - second_last_speed['speed']
                
                # 生成额外的预测点
                for i in range(current_count, target_steps):
                    # 基于趋势进行预测
                    predicted_speed = last_speed['speed'] + speed_change * (i - current_count + 1)
                    
                    new_speed = {
                        "step": i + 1,
                        "speed": float(predicted_speed),
                        "confidence": max(0.1, last_speed.get('confidence', 0.5) * 0.8)  # 置信度递减
                    }
                    
                    current_speeds.append(new_speed)
            else:
                # 如果只有一个预测点，保持速度不变
                last_speed = current_speeds[-1] if current_speeds else {"speed": 0, "confidence": 0.5}
                
                for i in range(current_count, target_steps):
                    new_speed = {
                        "step": i + 1,
                        "speed": float(last_speed['speed']),
                        "confidence": max(0.1, last_speed.get('confidence', 0.5) * 0.7)  # 置信度递减
                    }
                    
                    current_speeds.append(new_speed)
        else:
            # 获取速度列
            speed_col = None
            for col in vehicle_data.columns:
                if any(keyword in col.lower() for keyword in ['v_vel', 'speed', 'velocity']):
                    speed_col = col
                    break
            
            if speed_col:
                # 基于真实速度的最后几个值进行扩展预测
                last_real_speed = vehicle_data[speed_col].iloc[-1]
                
                # 计算平均速度变化
                if len(vehicle_data) > 1:
                    speed_change = vehicle_data[speed_col].diff().mean()
                else:
                    speed_change = 0.0
                
                # 生成额外的预测点
                for i in range(current_count, target_steps):
                    # 基于历史趋势进行预测
                    predicted_speed = last_real_speed + speed_change * (i - current_count + 1)
                    
                    new_speed = {
                        "step": i + 1,
                        "speed": float(predicted_speed),
                        "confidence": max(0.1, 1.0 - (i - current_count) * 0.1)  # 置信度递减
                    }
                    
                    current_speeds.append(new_speed)
            else:
                # 如果找不到速度列，使用默认速度
                last_speed = current_speeds[-1] if current_speeds else {"speed": 0, "confidence": 0.5}
                
                for i in range(current_count, target_steps):
                    new_speed = {
                        "step": i + 1,
                        "speed": float(last_speed['speed']),
                        "confidence": max(0.1, last_speed.get('confidence', 0.5) * 0.7)  # 置信度递减
                    }
                    
                    current_speeds.append(new_speed)
        
        result['predicted_speeds'] = current_speeds
        logger.info(f"扩展速度预测到 {len(current_speeds)} 个点")
        return result
            
    def save_prediction(self, prediction_result: Dict[str, Any], output_file: str) -> None:
        """保存预测结果到文件"""
        # 转换numpy数据类型为Python原生类型，以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            else:
                return obj
                
        # 转换数据类型
        prediction_result = convert_numpy_types(prediction_result)
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"预测结果已保存到 {output_file}")


def run(csv_path: str, vehicle_id: str, api_key: str, model: str, steps: int = None,
        output_file: str = None, semantic_json: str = None, features_json: str = None,
        predict_neighbors: bool = False) -> Dict[str, Any]:
    """运行轨迹预测"""
    # 创建预测器
    predictor = US101TrajectoryPredictor(csv_path, api_key, model)
    
    # 处理车辆ID
    if vehicle_id.lower() == 'random':
        # 加载数据
        predictor.load_data()
        # 随机选择一个车辆ID
        if predictor.relationship_analyzer is None:
            predictor.relationship_analyzer = VehicleRelationshipAnalyzer(predictor.data)
        vehicle_ids = predictor.relationship_analyzer.vehicle_ids
        if not vehicle_ids:
            raise ValueError("未找到任何车辆ID")
        vehicle_id = random.choice(vehicle_ids)
        logger.info(f"随机选择车辆ID: {vehicle_id}")
    
    # 预测轨迹
    prediction_result = predictor.predict_trajectory(
        vehicle_id=vehicle_id,
        steps=steps,
        semantic_json=semantic_json,
        features_json=features_json,
        predict_neighbors=predict_neighbors
    )
    
    # 保存预测结果
    if output_file:
        predictor.save_prediction(prediction_result, output_file)
    
    return prediction_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用阿里云MCP Agent预测车辆轨迹")
    parser.add_argument("--csv", required=True, help="US101 CSV文件路径")
    parser.add_argument("--vehicle_id", required=True, help="车辆ID，或使用'random'随机选择")
    parser.add_argument("--api_key", required=True, help="阿里云API密钥")
    parser.add_argument("--model", default="qwen-turbo", help="模型名称，默认为qwen-turbo")
    parser.add_argument("--steps", type=int, help="预测步数，默认使用真实轨迹的数据点数量")
    parser.add_argument("--output", help="输出文件路径，默认为当前目录下的prediction_result.json")
    parser.add_argument("--semantic_json", help="语义分析结果文件路径")
    parser.add_argument("--features_json", help="特征提取结果文件路径")
    parser.add_argument("--predict_neighbors", action="store_true", help="是否同时预测相邻车辆的轨迹")
    
    args = parser.parse_args()
    
    # 设置默认输出文件
    if not args.output:
        args.output = "prediction_result.json"
    
    # 运行预测
    run(
        csv_path=args.csv,
        vehicle_id=args.vehicle_id,
        api_key=args.api_key,
        model=args.model,
        steps=args.steps,
        output_file=args.output,
        semantic_json=args.semantic_json,
        features_json=args.features_json,
        predict_neighbors=args.predict_neighbors
    )