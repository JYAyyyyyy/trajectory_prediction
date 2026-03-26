import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import requests
import time
import warnings

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AliyunMCPClient:

    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.session = requests.Session()


    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_retries: int = 2) -> str:
        if not self.api_key:
            return '{"notice": "No API key provided. Returning mock analysis."}'

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
                resp = self.session.post(
                    self.endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    text = (
                        data.get("output", {}).get("text")
                        or data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content")
                        or data.get("output", {}).get("message", {}).get("content")
                        or json.dumps(data)
                    )
                    return text
                else:
                    logger.warning(f"API调用失败，状态码: {resp.status_code}")
                    time.sleep(0.8 * (attempt + 1))
            except Exception as e:
                logger.warning(f"API调用异常: {e}")
                time.sleep(0.8 * (attempt + 1))

        return '{"notice": "DashScope call failed. Returning mock analysis."}'


class MCPInformationExtractor:
    """MCP信息提取器"""

    def __init__(self, mcp: AliyunMCPClient):
        self.mcp = mcp

    def _extract_json_dict(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            # 尝试提取JSON块
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    json_str = text[start:end + 1]
                    # 检查JSON字符串是否被截断，如果是则尝试修复
                    left_braces = json_str.count('{')
                    right_braces = json_str.count('}')
                    if left_braces != right_braces:
                        if left_braces > right_braces:
                            # 添加缺失的右括号
                            json_str += '}' * (left_braces - right_braces)
                        else:
                            # 删除多余的右括号
                            excess_braces = right_braces - left_braces
                            for _ in range(excess_braces):
                                last_brace_pos = json_str.rfind('}')
                                if last_brace_pos != -1:
                                    json_str = json_str[:last_brace_pos] + json_str[last_brace_pos + 1:]
                    return json.loads(json_str)
                except Exception:
                    pass

            # 尝试从markdown代码块中提取
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end != -1:
                    try:
                        json_str = text[start:end].strip()
                        # 检查JSON字符串是否被截断
                        left_braces = json_str.count('{')
                        right_braces = json_str.count('}')
                        if left_braces != right_braces:
                            if left_braces > right_braces:
                                # 添加缺失的右括号
                                json_str += '}' * (left_braces - right_braces)
                            elif left_braces < right_braces:
                                # 删除多余的右括号
                                excess_braces = right_braces - left_braces
                                for _ in range(excess_braces):
                                    last_brace_pos = json_str.rfind('}')
                                    if last_brace_pos != -1:
                                        json_str = json_str[:last_brace_pos] + json_str[last_brace_pos + 1:]
                        return json.loads(json_str)
                    except Exception:
                        pass

            return {"raw": text, "error": "JSON解析失败"}

    def extract_information(self, instruction: str = "", text: str = "", examples: str = "",
                            schema: str = "", additional_info: str = "") -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的交通数据分析MCP代理。请严格按照提供的JSON模式返回纯JSON格式的结果，不要包含任何额外的注释或说明。"
            },
            {
                "role": "user",
                "content": (
                    f"指令:\n{instruction}\n\n"
                    f"示例:\n{examples}\n\n"
                    f"数据:\n{text}\n\n"
                    f"附加信息:\n{additional_info}\n\n"
                    f"JSON模式:\n{schema}\n\n"
                    "请只返回有效的JSON, 不要包含任何额外内容."
                )
            }
        ]

        resp = self.mcp.chat(messages)
        #print("resp:" + str(resp))
        return self._extract_json_dict(resp)


class MCPExtractionAgent:
    """MCP提取代理"""

    def __init__(self, mcp: AliyunMCPClient):
        self.mcp = mcp
        self.module = MCPInformationExtractor(mcp)

    def extract_information_direct(self, instruction: str, text: str, schema_obj: Dict[str, Any],
                                   examples: Optional[str] = None, constraint: Optional[Any] = None) -> Dict[str, Any]:
        schema = json.dumps(schema_obj, ensure_ascii=False)
        #print("schema:" + str(schema))
        #print("instruction:" + str(instruction))
        #print("text:" + str(text))
        #print("examples:" + str(examples))
        return self.module.extract_information(
			instruction=instruction,
			text=text,
			examples=examples or "",
			schema=schema,
			additional_info=json.dumps({"constraint": constraint}, ensure_ascii=False) if constraint else ""
		)


class US101FeatureExtractor:
    """US101数据集特征提取器"""

    def __init__(self, csv_path: str, api_key: str, model: str = "qwen-turbo"):
        self.csv_path = csv_path
        self.api_key = api_key
        self.model = model
        self.data = None
        self.features = {}
        self.mcp_client = AliyunMCPClient(api_key, model)
        self.extraction_agent = MCPExtractionAgent(self.mcp_client)

    def load_data(self) -> pd.DataFrame:
        """加载US101数据集"""
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.csv_path, encoding=encoding)
                    logger.info(f"成功使用 {encoding} 编码加载数据")
                    return self.data
                except UnicodeDecodeError:
                    continue

            self.data = pd.read_csv(self.csv_path, encoding='utf-8', errors='ignore')
            logger.warning("使用错误忽略模式加载数据")
            return self.data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def explore_data(self) -> Dict[str, Any]:
        """数据探索分析"""
        if self.data is None:
            self.load_data()

        exploration = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object']).columns.tolist(),
            'sample_data': self.data.head().to_dict('records')
        }

        logger.info("数据探索完成")
        return exploration

    def extract_basic_features(self) -> Dict[str, Any]:
        """提取基础统计特征"""
        # 尝试加载数据，如果失败则直接使用传统方法
        try:
            if self.data is None:
                self.load_data()
        except Exception as e:
            logger.warning(f"数据加载失败，使用传统方法: {e}")
            return self._extract_basic_features_traditional()

        #print(self.data)

        # 2. 构建数据摘要
        # data_summary = self._build_data_summary()

        schema = {
            "basic_features": {
                "numeric_columns": {
                    "column_name": {
                        "mean": "float",
                        "std": "float",
                        "min": "float",
                        "max": "float",
                        "median": "float",
                        "skewness": "float",
                        "kurtosis": "float"
                    }
                }
            }
        }

        instruction = "分析US101交通数据集的基础统计特征，计算数值列的均值、标准差、最小值、最大值、中位数、偏度和峰度"

        try:
            logger.info("使用MCP agent提取基础特征...")
            result = self.extraction_agent.extract_information_direct(
                instruction=instruction,
                text=self.data,
                schema_obj=schema
            )

            #print(result)
            if result and 'basic_features' in result:
                logger.info("MCP agent基础特征提取成功")
                return result.get('basic_features', {})
            else:
                logger.warning("MCP agent返回结果格式不正确，回退到传统方法")
                return self._extract_basic_features_traditional()

        except Exception as e:
            logger.error(f"MCP agent提取失败: {e}")
            return self._extract_basic_features_traditional()

    def extract_traffic_specific_features(self) -> Dict[str, Any]:
        """提取交通特定特征"""
        try:
            if self.data is None:
                self.load_data()
        except Exception as e:
            logger.warning(f"数据加载失败，使用传统方法: {e}")
            return self._extract_traffic_specific_features_traditional()

        #traffic_summary = self._build_traffic_summary()

        schema = {
            "traffic_features": {
                "vehicle_count": "int",
                "time_range": {
                    "start_time": "string",
                    "end_time": "string",
                    "duration_minutes": "float"
                },
                "speed_analysis": {
                    "avg_speed": "float",
                    "speed_variance": "float",
                    "speed_distribution": "string"
                },
                "spatial_analysis": {
                    "road_length": "float",
                    "spatial_coverage": "string"
                }
            }
        }

        instruction = "分析US101交通数据集的交通特定特征，包括车辆数量、时间范围、速度分析和空间分析"

        try:
            logger.info("使用MCP agent提取交通特定特征...")
            result = self.extraction_agent.extract_information_direct(
                instruction=instruction,
                text=self.data,
                schema_obj=schema
            )

            if result and 'traffic_features' in result:
                logger.info("MCP agent交通特定特征提取成功")
                return result.get('traffic_features', {})
            else:
                logger.warning("MCP agent返回结果格式不正确，回退到传统方法")
                return self._extract_traffic_specific_features_traditional()

        except Exception as e:
            logger.error(f"MCP agent提取失败: {e}")
            return self._extract_traffic_specific_features_traditional()


    '''def extract_spatiotemporal_features(self) -> Dict[str, Any]:
        """提取时空特征：传统方法计算，LLM解读"""
        try:
            if self.data is None:
                self.load_data()

            # 1. 使用传统方法进行精确计算
            logger.info("使用传统方法计算时空特征...")
            traditional_result = self._extract_spatiotemporal_features_traditional()

            # 2. 将传统方法的结果转换成文本摘要，让LLM进行分析解读
            #analysis_summary = self._build_spatiotemporal_summary(traditional_result)

            schema = {
                "spatiotemporal_analysis": {  # 注意：这里重命名为analysis，强调是分析而非计算
                    "key_insights": "array",  # 主要发现
                    "traffic_pattern_description": "string",  # 交通模式描述
                    "anomalies_detected": "array"  # 检测到的异常
                }
            }

            instruction = "你是一名交通数据分析专家。请基于以下计算出的时空特征结果，总结关键洞察、描述交通模式、并指出任何潜在的异常情况。"

            try:
                logger.info("使用MCP agent分析时空特征...")
                analysis_result = self.extraction_agent.extract_information_direct(
                    instruction=instruction,
                    text=self.data,  # 发送的是计算结果，不是原始数据摘要
                    schema_obj=schema
                )
                # 将LLM的分析结论与传统计算结果合并返回
                traditional_result['llm_analysis'] = analysis_result.get('spatiotemporal_analysis', {})

            except Exception as e:
                logger.debug(f"MCP分析跳过: {e}")

            return traditional_result

        except Exception as e:
            logger.error(f"时空特征提取失败: {e}")
            return {}'''

    def extract_spatiotemporal_features(self) -> Dict[str, Any]:
        """提取时空特征"""
        try:
            if self.data is None:
                self.load_data()
        except Exception as e:
            logger.warning(f"数据加载失败，使用传统方法: {e}")
            return self._extract_spatiotemporal_features_traditional()

        # 使用传统方法计算
        traditional_result = self._extract_spatiotemporal_features_traditional()

        # 基于计算结果构建摘要
        spatiotemporal_summary = self._build_spatiotemporal_summary(traditional_result)

        schema = {
            "spatiotemporal_features": {
                "temporal_patterns": {
                    "hourly_distribution": "object",
                    "daily_patterns": "object",
                    "seasonal_trends": "string"
                },
                "spatial_patterns": {
                    "location_distribution": "object",
                    "spatial_clustering": "string",
                    "road_segments": "object"
                },
                "trajectory_features": {
                    "avg_trajectory_length": "float",
                    "trajectory_complexity": "string",
                    "intersection_patterns": "object"
                }
            }
        }

        instruction = "基于以下时空特征分析结果，提供深入的交通模式洞察和解释"

        try:
            logger.info("使用MCP agent分析时空特征...")
            result = self.extraction_agent.extract_information_direct(
                instruction=instruction,
                text=spatiotemporal_summary,  # 发送基于真实计算的摘要
                schema_obj=schema
            )

            if result and 'spatiotemporal_features' in result:
                logger.info("MCP agent时空特征分析成功")
                # 将LLM的分析结果与传统计算结果合并
                traditional_result['llm_analysis'] = result.get('spatiotemporal_features', {})
                return traditional_result
            else:
                logger.warning("MCP agent返回结果格式不正确，返回传统计算结果")
                return traditional_result

        except Exception as e:
            logger.error(f"MCP agent分析失败: {e}")
            return traditional_result

    '''def extract_statistical_features(self) -> Dict[str, Any]:
        """提取统计特征"""
        try:
            if self.data is None:
                self.load_data()
        except Exception as e:
            logger.warning(f"数据加载失败，使用传统方法: {e}")
            return self._extract_statistical_features_traditional()


        # 基于计算结果构建摘要
        statistical_summary = self._build_statistical_summary()

        schema = {
            "statistical_features": {
                "correlation_analysis": "object",
                "outlier_detection": {
                    "outlier_count": "int",
                    "outlier_percentage": "float",
                    "outlier_columns": "list"
                },
                "distribution_analysis": {
                    "normal_distribution": "list",
                    "skewed_distribution": "list",
                    "uniform_distribution": "list"
                }
            }
        }

        instruction = "基于以下统计特征分析结果，提供深入的数据洞察和解释"

        #print("traditional_result:"+str(traditional_result))

        try:
            logger.info("使用MCP agent分析统计特征...")
            result = self.extraction_agent.extract_information_direct(
                instruction=instruction,
                text=self.data,  # 发送基于真实计算的摘要
                schema_obj=schema
            )

        except Exception as e:
            logger.error(f"MCP agent分析失败: {e}")
            return statistical_summary'''

    def _extract_basic_features_traditional(self) -> Dict[str, Any]:
        """传统方法提取基础特征"""
        logger.info("传统方法基础特征提取完成")
        
        # 检查数据是否存在
        if self.data is None:
            logger.warning("数据未加载，返回默认基础特征")
            return {
                "numeric_columns": {
                    "Vehicle_ID": {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "median": 0.0,
                        "skewness": 0.0,
                        "kurtosis": 0.0
                    }
                }
            }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        basic_features = {"numeric_columns": {}}

        for col in numeric_cols:
            try:
                basic_features["numeric_columns"][col] = {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max()),
                    'median': float(self.data[col].median()),
                    'skewness': float(self.data[col].skew()),
                    'kurtosis': float(self.data[col].kurtosis())
                }
            except Exception as e:
                logger.warning(f"列 {col} 处理失败: {e}")

        return basic_features

    def _extract_traffic_specific_features_traditional(self) -> Dict[str, Any]:
        """传统方法提取交通特定特征"""
        logger.info("传统方法交通特定特征提取完成")
        
        # 检查数据是否存在
        if self.data is None:
            logger.warning("数据未加载，返回默认交通特定特征")
            return {
                "vehicle_count": 0,
                "time_range": {
                    "start_time": "unknown",
                    "end_time": "unknown",
                    "duration_minutes": 0
                },
                "speed_analysis": {
                    "avg_speed": 0.0,
                    "speed_variance": 0.0,
                    "speed_distribution": "unknown"
                },
                "spatial_analysis": {
                    "road_length": 0.0,
                    "spatial_coverage": "unknown"
                }
            }

        # 假设有时间和位置列
        time_cols = [col for col in self.data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        speed_cols = [col for col in self.data.columns if 'speed' in col.lower() or 'velocity' in col.lower()]
        pos_cols = [col for col in self.data.columns if
                    'position' in col.lower() or 'location' in col.lower() or 'x' in col.lower() or 'y' in col.lower()]

        traffic_features = {
            "vehicle_count": len(self.data),
            "time_range": {
                "start_time": str(self.data[time_cols[0]].min()) if time_cols else "unknown",
                "end_time": str(self.data[time_cols[0]].max()) if time_cols else "unknown",
                "duration_minutes": 0
            },
            "speed_analysis": {
                "avg_speed": float(self.data[speed_cols[0]].mean()) if speed_cols else 0.0,
                "speed_variance": float(self.data[speed_cols[0]].var()) if speed_cols else 0.0,
                "speed_distribution": "normal"
            },
            "spatial_analysis": {
                "road_length": 0.0,
                "spatial_coverage": "unknown"
            }
        }

        return traffic_features

    '''def _extract_spatiotemporal_features_traditional(self) -> Dict[str, Any]:
        """传统方法提取时空特征"""
        logger.info("传统方法时空特征提取完成")
        
        # 检查数据是否存在
        if self.data is None:
            logger.warning("数据未加载，返回默认时空特征")

        spatiotemporal_features = {
            "temporal_patterns": {
                "hourly_distribution": {},
                "daily_patterns": {},
                "seasonal_trends": "未提供足够的数据以分析季节性变化" if self.data is None else "unknown"
            },
            "spatial_patterns": {
                "location_distribution": {},
                "spatial_clustering": "unknown",
                "road_segments": {}
            },
            "trajectory_features": {
                "avg_trajectory_length": 0.0,
                "trajectory_complexity": "simple",
                "intersection_patterns": {}
            }
        }

        return spatiotemporal_features'''

    def _extract_spatiotemporal_features_traditional(self) -> Dict[str, Any]:
        """使用传统方法计算时空特征"""
        if self.data is None or len(self.data) == 0:
            return {
                "temporal_patterns": {},
                "spatial_patterns": {},
                "trajectory_features": {}
            }

        result = {
            "temporal_patterns": {},
            "spatial_patterns": {},
            "trajectory_features": {}
        }

        # 1. 时间模式分析
        if 'Global_Time' in self.data.columns:
            try:
                # 转换时间戳为datetime
                timestamps = pd.to_datetime(self.data['Global_Time'], unit='ms')

                # 小时分布
                hourly_dist = timestamps.dt.hour.value_counts().to_dict()
                result["temporal_patterns"]["hourly_distribution"] = hourly_dist

                # 日模式（按星期几）
                daily_dist = timestamps.dt.dayofweek.value_counts().to_dict()
                result["temporal_patterns"]["daily_patterns"] = daily_dist

                # 季节性趋势
                month_dist = timestamps.dt.month.value_counts()
                if month_dist.nunique() > 1:
                    result["temporal_patterns"]["seasonal_trends"] = "存在月度变化趋势"
                else:
                    result["temporal_patterns"]["seasonal_trends"] = "数据集中在单月内"

            except Exception as e:
                logger.warning(f"时间分析错误: {e}")

        # 2. 空间模式分析
        if {'Local_X', 'Local_Y'}.issubset(self.data.columns):
            try:
                # 位置分布（热点区域）- 修复：将Interval转换为字符串
                x_bins = pd.cut(self.data['Local_X'], bins=10)
                y_bins = pd.cut(self.data['Local_Y'], bins=10)

                x_bins_dict = {str(interval): int(count) for interval, count in x_bins.value_counts().to_dict().items()}
                y_bins_dict = {str(interval): int(count) for interval, count in y_bins.value_counts().to_dict().items()}

                result["spatial_patterns"]["location_distribution"] = {
                    'x_bins': x_bins_dict,
                    'y_bins': y_bins_dict
                }

                # 空间聚类
                x_std, y_std = self.data['Local_X'].std(), self.data['Local_Y'].std()
                if x_std > 50 or y_std > 50:
                    result["spatial_patterns"]["spatial_clustering"] = "空间分布较分散"
                else:
                    result["spatial_patterns"]["spatial_clustering"] = "空间分布较集中"

                # 路段分析（按X坐标分段）- 修复：将Interval转换为字符串
                if 'Local_X' in self.data.columns:
                    segments = pd.cut(self.data['Local_X'], bins=5)
                    road_segments = {}

                    for segment in segments.unique():
                        if pd.notna(segment):
                            segment_data = self.data[segments == segment]
                            road_segments[str(segment)] = {
                                'vehicle_count': int(len(segment_data)),
                                'avg_speed': float(segment_data['v_Vel'].mean()) if 'v_Vel' in segment_data else 0.0
                            }

                    result["spatial_patterns"]["road_segments"] = road_segments

            except Exception as e:
                logger.warning(f"空间分析错误: {e}")

        # 3. 轨迹特征分析
        if {'Vehicle_ID', 'Local_X', 'Local_Y'}.issubset(self.data.columns):
            try:
                # 按车辆分组计算轨迹
                trajectory_lengths = []
                for vehicle_id, group in self.data.groupby('Vehicle_ID'):
                    if len(group) > 1:
                        coords = group[['Local_X', 'Local_Y']].values
                        path_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))
                        trajectory_lengths.append(path_length)

                if trajectory_lengths:
                    result["trajectory_features"]["avg_trajectory_length"] = float(np.mean(trajectory_lengths))

                    # 轨迹复杂度（基于路径长度与直线距离的比值）
                    complexity_scores = []
                    for vehicle_id, group in self.data.groupby('Vehicle_ID'):
                        if len(group) > 1:
                            start_point = group[['Local_X', 'Local_Y']].iloc[0].values
                            end_point = group[['Local_X', 'Local_Y']].iloc[-1].values
                            straight_distance = np.sqrt(np.sum((end_point - start_point) ** 2))
                            actual_length = np.sum(
                                np.sqrt(np.sum(np.diff(group[['Local_X', 'Local_Y']].values, axis=0) ** 2, axis=1)))

                            if straight_distance > 0:
                                complexity = actual_length / straight_distance
                                complexity_scores.append(complexity)

                    if complexity_scores:
                        avg_complexity = np.mean(complexity_scores)
                        if avg_complexity < 1.1:
                            result["trajectory_features"]["trajectory_complexity"] = "轨迹较直"
                        elif avg_complexity < 1.5:
                            result["trajectory_features"]["trajectory_complexity"] = "轨迹中等弯曲"
                        else:
                            result["trajectory_features"]["trajectory_complexity"] = "轨迹复杂弯曲"

            except Exception as e:
                logger.warning(f"轨迹分析错误: {e}")

        return result

    '''def _extract_statistical_features_traditional(self) -> Dict[str, Any]:
        """使用传统方法计算统计特征"""
        if self.data is None or len(self.data) == 0:
            return {
                "correlation_analysis": {},
                "outlier_detection": {
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "outlier_columns": []
                },
                "distribution_analysis": {
                    "normal_distribution": [],
                    "skewed_distribution": [],
                    "uniform_distribution": []
                }
            }

        result = {
            "correlation_analysis": {},
            "outlier_detection": {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_columns": []
            },
            "distribution_analysis": {
                "normal_distribution": [],
                "skewed_distribution": [],
                "uniform_distribution": []
            }
        }

        # 选择数值列，排除ID等非特征列
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]

        # 1. 相关性分析
        try:
            if len(feature_columns) > 1:
                correlation_matrix = self.data[feature_columns].corr()
                # 转换为可序列化的字典格式
                for col in correlation_matrix.columns:
                    # 将Series转换为字典，并确保所有值都是基本类型
                    result["correlation_analysis"][col] = {
                        other_col: float(value) for other_col, value in correlation_matrix[col].items()
                    }
        except Exception as e:
            logger.warning(f"相关性分析错误: {e}")

        # 2. 异常值检测
        try:
            outlier_columns = []
            total_outliers = 0
            total_points = 0

            for column in feature_columns:
                col_data = self.data[column].dropna()
                if len(col_data) > 0:
                    total_points += len(col_data)
                    # 使用IQR方法检测异常值
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    if len(outliers) > 0:
                        outlier_columns.append(column)
                        total_outliers += len(outliers)

            result["outlier_detection"]["outlier_count"] = int(total_outliers)
            result["outlier_detection"]["outlier_percentage"] = float(
                total_outliers / total_points * 100) if total_points > 0 else 0.0
            result["outlier_detection"]["outlier_columns"] = outlier_columns

        except Exception as e:
            logger.warning(f"异常值检测错误: {e}")

        # 3. 分布分析
        try:
            normal_distributions = []
            skewed_distributions = []
            uniform_distributions = []

            for column in feature_columns:
                col_data = self.data[column].dropna()
                if len(col_data) > 0:
                    # 计算偏度和峰度
                    skewness = col_data.skew()
                    kurtosis = col_data.kurtosis()

                    # 判断分布类型
                    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
                        normal_distributions.append(column)
                    elif abs(skewness) > 1:
                        skewed_distributions.append(column)
                    elif abs(skewness) < 0.2 and abs(kurtosis) < 0.5:
                        uniform_distributions.append(column)

            result["distribution_analysis"]["normal_distribution"] = normal_distributions
            result["distribution_analysis"]["skewed_distribution"] = skewed_distributions
            result["distribution_analysis"]["uniform_distribution"] = uniform_distributions

        except Exception as e:
            logger.warning(f"分布分析错误: {e}")

        return result'''

    def extract_all_features(self) -> Dict[str, Any]:
        """提取所有特征"""
        logger.info("开始提取所有特征...")

        # 基础特征
        self.features['basic_features'] = self.extract_basic_features()

        # 交通特定特征
        self.features['traffic_features'] = self.extract_traffic_specific_features()

        # 时空特征
        self.features['spatiotemporal_features'] = self.extract_spatiotemporal_features()

        # 统计特征
        # self.features['statistical_features'] = self.extract_statistical_features()

        logger.info("所有特征提取完成")
        return self.features

    def save_features(self, output_dir: str = "output") -> str:
        """保存特征到文件"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"features_mcp.json")
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #output_path = os.path.join(output_dir, f"features_{timestamp}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.features, f, ensure_ascii=False, indent=2)

        logger.info(f"特征已保存至 {output_path}")
        return output_path

    def _build_data_summary(self) -> str:
        """构建数据摘要"""
        if self.data is None:
            return ""

        summary = [
            f"数据集形状: {self.data.shape}",
            f"数值列: {self.data.select_dtypes(include=[np.number]).columns.tolist()}",
            f"分类列: {self.data.select_dtypes(include=['object']).columns.tolist()}",
            f"缺失值统计: {self.data.isnull().sum().to_dict()}",
            f"数据样本:\n{self.data.head().to_string()}"
        ]
        return "\n".join(summary)

    '''def _build_traffic_summary(self) -> str:
        """构建交通数据摘要"""
        if self.data is None:
            return ""

        summary = [
            f"数据集大小: {len(self.data)} 条记录",
            f"列名: {list(self.data.columns)}",
            f"数据类型: {self.data.dtypes.to_dict()}",
            f"数值列统计:\n{self.data.describe().to_string()}"
        ]
        return "\n".join(summary)'''

    def _build_traffic_summary(self) -> str:
        """构建交通数据摘要"""
        if self.data is None or len(self.data) == 0:
            return "无数据可用"

        summary_lines = []

        # 1. 基础信息
        summary_lines.append("=== 交通数据集基础信息 ===")
        summary_lines.append(f"数据集大小: {len(self.data)} 条记录, {len(self.data.columns)} 个特征")
        summary_lines.append(f"时间范围: {self._get_time_range_summary()}")
        summary_lines.append(f"车辆数量: {self.data['Vehicle_ID'].nunique()} 辆")

        # 2. 交通特定统计
        summary_lines.append("\n=== 交通特征统计 ===")

        if 'v_Vel' in self.data.columns:
            speeds = self.data['v_Vel'].dropna()
            if len(speeds) > 0:
                summary_lines.append(f"速度统计: 均值={speeds.mean():.2f}m/s, 标准差={speeds.std():.2f}, "
                                     f"范围=[{speeds.min():.2f}-{speeds.max():.2f}]")

        if 'v_Acc' in self.data.columns:
            accels = self.data['v_Acc'].dropna()
            if len(accels) > 0:
                summary_lines.append(f"加速度统计: 均值={accels.mean():.2f}m/s², "
                                     f"范围=[{accels.min():.2f}-{accels.max():.2f}]")

        if 'Lane_ID' in self.data.columns:
            lane_counts = self.data['Lane_ID'].value_counts()
            summary_lines.append(f"车道分布: {dict(lane_counts.head())}")

        # 3. 空间信息
        summary_lines.append("\n=== 空间特征 ===")
        if {'Local_X', 'Local_Y'}.issubset(self.data.columns):
            summary_lines.append(f"X坐标范围: [{self.data['Local_X'].min():.1f}-{self.data['Local_X'].max():.1f}]")
            summary_lines.append(f"Y坐标范围: [{self.data['Local_Y'].min():.1f}-{self.data['Local_Y'].max():.1f}]")

        # 4. 数据质量信息
        summary_lines.append("\n=== 数据质量 ===")
        missing_stats = self.data.isnull().sum()
        missing_percent = (missing_stats / len(self.data) * 100).round(2)
        summary_lines.append(f"缺失值统计: {dict(missing_percent[missing_percent > 0])}")

        # 5. 关键洞察（基于简单计算）
        summary_lines.append("\n=== 初步洞察 ===")
        if 'v_Vel' in self.data.columns and len(speeds) > 0:
            if speeds.std() < 2.0:
                summary_lines.append("• 车辆速度变化较小，交通流较为稳定")
            else:
                summary_lines.append("• 车辆速度变化较大，可能存在拥堵或自由流混合")

        if 'v_Acc' in self.data.columns and len(accels) > 0:
            if (accels < 0).sum() > (accels > 0).sum():
                summary_lines.append("• 减速行为多于加速行为，可能处于拥堵路段")

        return "\n".join(summary_lines)

    def _get_time_range_summary(self) -> str:
        """获取时间范围摘要"""
        if 'Global_Time' not in self.data.columns:
            return "时间信息不可用"

        try:
            # 假设Global_Time是毫秒时间戳
            timestamps = pd.to_datetime(self.data['Global_Time'], unit='ms')
            time_min = timestamps.min()
            time_max = timestamps.max()
            duration = time_max - time_min

            return (f"{time_min.strftime('%Y-%m-%d %H:%M:%S')} 到 "
                    f"{time_max.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(持续时间: {duration.total_seconds() / 60:.1f}分钟)")
        except:
            return "时间格式解析失败"

    '''def _build_spatiotemporal_summary(self) -> str:
        """构建时空数据摘要"""
        if self.data is None:
            return ""

        summary = [
            f"数据集大小: {len(self.data)} 条记录",
            f"时间相关列: {[col for col in self.data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]}",
            f"位置相关列: {[col for col in self.data.columns if 'position' in col.lower() or 'location' in col.lower() or 'x' in col.lower() or 'y' in col.lower()]}",
            f"速度相关列: {[col for col in self.data.columns if 'speed' in col.lower() or 'velocity' in col.lower()]}"
        ]
        return "\n".join(summary)'''

    '''def _build_spatiotemporal_summary(self, traditional_result: Dict[str, Any] = None) -> str:
        """基于交通数据集构建时空特征摘要"""
        if self.data is None or len(self.data) == 0:
            return "无数据可用"

        summary_lines = []

        # 1. 数据集概览
        summary_lines.append("=== 交通时空数据集概览 ===")
        summary_lines.append(f"• 总记录数: {len(self.data):,} 条")
        summary_lines.append(f"• 独立车辆数: {self.data['Vehicle_ID'].nunique():,} 辆")
        summary_lines.append(f"• 数据维度: {len(self.data.columns)} 个特征")

        # 2. 时间特征分析
        summary_lines.append("\n=== 时间特征分析 ===")
        if 'Global_Time' in self.data.columns:
            try:
                # 转换时间戳为datetime
                timestamps = pd.to_datetime(self.data['Global_Time'], unit='ms')
                time_min = timestamps.min()
                time_max = timestamps.max()
                duration = time_max - time_min
                summary_lines.append(
                    f"• 时间范围: {time_min.strftime('%Y-%m-%d %H:%M:%S')} 至 {time_max.strftime('%H:%M:%S')}")
                summary_lines.append(f"• 持续时间: {duration.total_seconds() / 60:.1f} 分钟")

                # 分析时间分布
                hour_dist = timestamps.dt.hour.value_counts()
                if not hour_dist.empty:
                    peak_hour = hour_dist.idxmax()
                    summary_lines.append(
                        f"• 高峰时段: {peak_hour:02d}:00 ({(hour_dist.max() / len(timestamps) * 100):.1f}%数据量)")
            except Exception as e:
                summary_lines.append(f"• 时间分析错误: {str(e)}")

        # 3. 空间位置分析
        summary_lines.append("\n=== 空间位置分析 ===")
        if {'Local_X', 'Local_Y'}.issubset(self.data.columns):
            x_range = self.data['Local_X'].max() - self.data['Local_X'].min()
            y_range = self.data['Local_Y'].max() - self.data['Local_Y'].min()
            summary_lines.append(f"• 局部坐标范围: X方向 {x_range:.1f}m, Y方向 {y_range:.1f}m")

            # 全局坐标（如果可用）
            if {'Global_X', 'Global_Y'}.issubset(self.data.columns):
                global_x_range = self.data['Global_X'].max() - self.data['Global_X'].min()
                global_y_range = self.data['Global_Y'].max() - self.data['Global_Y'].min()
                summary_lines.append(f"• 全局坐标范围: X方向 {global_x_range:.1f}m, Y方向 {global_y_range:.1f}m")

        # 4. 车辆运动分析
        summary_lines.append("\n=== 车辆运动特征 ===")
        if 'v_Vel' in self.data.columns:
            speeds = self.data['v_Vel'].dropna()
            if len(speeds) > 0:
                summary_lines.append(f"• 速度统计: {speeds.mean():.2f} ± {speeds.std():.2f} m/s")
                summary_lines.append(f"• 速度范围: {speeds.min():.2f} - {speeds.max():.2f} m/s")
                summary_lines.append(f"• 85%位速度: {np.percentile(speeds, 85):.2f} m/s")

        if 'v_Acc' in self.data.columns:
            accels = self.data['v_Acc'].dropna()
            if len(accels) > 0:
                summary_lines.append(f"• 加速度统计: {accels.mean():.3f} ± {accels.std():.3f} m/s²")
                summary_lines.append(f"• 急加速次数(v_Acc > 2.0): {(accels > 2.0).sum()}")
                summary_lines.append(f"• 急减速次数(v_Acc < -2.0): {(accels < -2.0).sum()}")

        # 5. 车辆属性分析
        summary_lines.append("\n=== 车辆属性分析 ===")
        if 'v_Length' in self.data.columns:
            lengths = self.data['v_Length'].dropna()
            if len(lengths) > 0:
                summary_lines.append(f"• 车辆长度: {lengths.mean():.1f} ± {lengths.std():.1f} m")

        if 'v_Width' in self.data.columns:
            widths = self.data['v_Width'].dropna()
            if len(widths) > 0:
                summary_lines.append(f"• 车辆宽度: {widths.mean():.1f} ± {widths.std():.1f} m")

        if 'v_Class' in self.data.columns:
            class_dist = self.data['v_Class'].value_counts()
            summary_lines.append(f"• 车辆类型分布: {dict(class_dist)}")

        # 6. 车道和跟车行为分析
        summary_lines.append("\n=== 车道与跟车行为 ===")
        if 'Lane_ID' in self.data.columns:
            lane_dist = self.data['Lane_ID'].value_counts()
            summary_lines.append(f"• 车道使用分布: {dict(lane_dist)}")

        if {'Preceding', 'Following'}.issubset(self.data.columns):
            has_preceding = (self.data['Preceding'] > 0).sum()
            has_following = (self.data['Following'] > 0).sum()
            summary_lines.append(f"• 有前车记录: {has_preceding} 条 ({(has_preceding / len(self.data) * 100):.1f}%)")
            summary_lines.append(f"• 有后车记录: {has_following} 条 ({(has_following / len(self.data) * 100):.1f}%)")

        # 7. 车距分析
        if 'Space_Hdwy' in self.data.columns:
            space_hdwy = self.data['Space_Hdwy'].dropna()
            if len(space_hdwy) > 0:
                summary_lines.append(f"• 空间车距: {space_hdwy.mean():.1f} ± {space_hdwy.std():.1f} m")
                summary_lines.append(f"• 最小安全距离: {space_hdwy.min():.1f} m")

        if 'Time_Hdwy' in self.data.columns:
            time_hdwy = self.data['Time_Hdwy'].dropna()
            if len(time_hdwy) > 0:
                summary_lines.append(f"• 时间车距: {time_hdwy.mean():.2f} ± {time_hdwy.std():.2f} s")

        # 8. 数据质量检查
        summary_lines.append("\n=== 数据质量检查 ===")
        missing_stats = self.data.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]
        if len(missing_cols) > 0:
            summary_lines.append(f"• 存在缺失值的列: {dict(missing_cols)}")
        else:
            summary_lines.append("• 数据完整: 无缺失值")

        # 9. 集成传统计算结果（如果提供）
        if traditional_result and 'numeric_columns' in traditional_result:
            summary_lines.append("\n=== 传统计算结果集成 ===")
            num_features = len(traditional_result['numeric_columns'])
            summary_lines.append(f"• 已计算 {num_features} 个数值特征的统计量")

            # 显示一些关键统计结果
            key_columns = ['v_Vel', 'v_Acc', 'Local_X', 'Local_Y']
            for col in key_columns:
                if col in traditional_result['numeric_columns']:
                    stats = traditional_result['numeric_columns'][col]
                    summary_lines.append(f"• {col}: μ={stats.get('mean', 0):.2f}, σ={stats.get('std', 0):.2f}")

        return "\n".join(summary_lines)'''

    def _build_spatiotemporal_summary(self, traditional_result: Dict[str, Any]) -> str:
        """基于传统计算结果构建时空特征摘要"""
        if not traditional_result:
            return "无时空特征计算结果可用"

        summary_lines = []

        # 1. 时间模式摘要
        if 'temporal_patterns' in traditional_result:
            temp_patterns = traditional_result['temporal_patterns']
            summary_lines.append("=== 时间模式分析 ===")

            if 'hourly_distribution' in temp_patterns:
                hourly = temp_patterns['hourly_distribution']
                if hourly:
                    peak_hour = max(hourly.items(), key=lambda x: x[1])[0]
                    summary_lines.append(f"• 交通高峰时段: {peak_hour}时")

            if 'seasonal_trends' in temp_patterns:
                summary_lines.append(f"• 季节性趋势: {temp_patterns['seasonal_trends']}")

        # 2. 空间模式摘要
        if 'spatial_patterns' in traditional_result:
            spatial_patterns = traditional_result['spatial_patterns']
            summary_lines.append("\n=== 空间模式分析 ===")

            if 'spatial_clustering' in spatial_patterns:
                summary_lines.append(f"• 空间分布: {spatial_patterns['spatial_clustering']}")

            if 'road_segments' in spatial_patterns:
                segments = spatial_patterns['road_segments']
                summary_lines.append(f"• 路段分析: 共{len(segments)}个路段段")

        # 3. 轨迹特征摘要
        if 'trajectory_features' in traditional_result:
            trajectory = traditional_result['trajectory_features']
            summary_lines.append("\n=== 轨迹特征分析 ===")

            if 'avg_trajectory_length' in trajectory:
                summary_lines.append(f"• 平均轨迹长度: {trajectory['avg_trajectory_length']:.1f}米")

            if 'trajectory_complexity' in trajectory:
                summary_lines.append(f"• 轨迹复杂度: {trajectory['trajectory_complexity']}")

        return "\n".join(summary_lines)

    def _build_statistical_summary(self) -> str:
        """构建统计摘要"""
        if self.data is None:
            return ""

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        summary = [
            f"数值列数量: {len(numeric_cols)}",
            f"数值列: {list(numeric_cols)}",
            f"相关性矩阵:\n{self.data[numeric_cols].corr().to_string()}",
            f"描述性统计:\n{self.data[numeric_cols].describe().to_string()}"
        ]
        return "\n".join(summary)


def run(csv_path: str, api_key: str, model: str = "qwen-turbo", output_dir: str = "output") -> str:
    """
    运行特征提取流程

    Args:
        csv_path: 数据文件路径
        api_key: API密钥
        model: 模型名称
        output_dir: 输出目录
    """
    extractor = US101FeatureExtractor(csv_path, api_key, model)
    extractor.load_data()
    extractor.explore_data()
    extractor.extract_all_features()
    return extractor.save_features(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="US101交通数据特征提取")
    parser.add_argument("csv_path", help="CSV文件路径")
    parser.add_argument("--api_key", required=True, help="DashScope API密钥")
    parser.add_argument("--model", default="qwen-turbo", help="模型名称")
    parser.add_argument("--output", default="output", help="输出目录")

    args = parser.parse_args()

    try:
        result_path = run(
            csv_path=args.csv_path,
            api_key=args.api_key,
            model=args.model,
            output_dir=args.output
        )
        print(f"特征提取完成，结果保存至: {result_path}")
    except Exception as e:
        logger.error(f"运行失败: {e}")
        raise