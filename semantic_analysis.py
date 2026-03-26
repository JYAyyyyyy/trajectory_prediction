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
                "result_format": "message"  # 确保返回消息格式
            }
        }
        
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.post(
                    self.endpoint, 
                    headers=headers,
                    json=payload,
                    data=json.dumps(payload), 
                    timeout=60
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"API响应数据: {json.dumps(data, ensure_ascii=False)[:200]}...")  # 调试信息
                    # 尝试从不同路径提取生成文本
                    text = (
                        data.get("output", {}).get("text")
                        or data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content")
                        or json.dumps(data)
                    )
                    return text
                else:
                    time.sleep(0.8 * (attempt + 1))
            except Exception:
                    time.sleep(0.8 * (attempt + 1))

        return '{"notice": "DashScope call failed. Returning mock analysis."}'


class MCPInformationExtractor:
    """MCP信息提取器，参考OneKE的extraction_agent.py风格"""
    
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
        try:
            return json.loads(text)
        except Exception:
            # 启发式提取JSON块
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    return {"raw": text}
            return {"raw": text}

    def extract_information(self, instruction: str = "", text: str = "", examples: str = "", 
                           schema: str = "", additional_info: str = "") -> Dict[str, Any]:
        """
        提取结构化信息
        
        Args:
            instruction: 指令
            text: 输入文本
            examples: 示例
            schema: 输出模式
            additional_info: 附加信息
            
        Returns:
            提取的结构化信息
        """
        messages = [
        {
            "role": "system",
            "content": "You are an expert traffic-data MCP agent. Always return pure JSON that matches the provided schema."
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                f"Examples:\n{examples}\n\n"
                f"Text:\n{text}\n\n"
                f"Additional Info:\n{additional_info}\n\n"
                f"JSON Schema (keys & types hint):\n{schema}\n\n"
                "Return only valid JSON with no extra commentary."
            )
        }
        ]

        resp = self.mcp.chat(messages)
        return self._extract_json_dict(resp)


class MCPExtractionAgent:
    """MCP提取代理"""
    
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


class TrafficSemanticAnalyzer:
    """交通语义分析器"""
    
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        """
        初始化语义分析器
        
        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.mcp = AliyunMCPClient(api_key, model)
        self.agent = MCPExtractionAgent(self.mcp)
    
    def analyze_traffic_patterns(self, features_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析交通模式
        
        Args:
            features_data: 特征数据
            
        Returns:
            分析结果
        """
        # 构建分析文本
        analysis_text = self._build_analysis_text(features_data)
        
        # 定义输出模式
        schema = {
            "traffic_patterns": ["list of identified traffic patterns"],
            "anomaly_detection": ["list of detected anomalies"], #检测到的交通异常
            "congestion_analysis": { #拥堵分析
                "congestion_level": "string (low/medium/high/severe)",
                "peak_hours": ["list of peak hours"],
                "congestion_causes": ["list of potential causes"]
            },
            "safety_assessment": { #安全评估
                "risk_level": "string (low/medium/high)",
                "risk_factors": ["list of risk factors"],
                "safety_recommendations": ["list of safety recommendations"] #安全改进建议
            },
            "efficiency_insights": {
                "flow_efficiency": "string (low/medium/high)",
                "bottlenecks": ["list of identified bottlenecks"], #瓶颈路段/点
                "optimization_suggestions": ["list of optimization suggestions"]
            },
            "spatiotemporal_analysis": {
                "temporal_patterns": {
                    "daily_rhythms": "string (description of daily traffic patterns)",
                    "weekly_patterns": "string (description of weekly variations)",
                    "seasonal_trends": "string (description of seasonal changes)",
                    "peak_periods": ["list of peak traffic periods"]
                },
                "spatial_patterns": {
                    "spatial_distribution": "string (description of spatial traffic distribution)",
                    "hotspots": ["list of traffic hotspots"],
                    "spatial_clustering": "string (description of spatial clustering patterns)", # 空间聚类特征
                    "road_segment_analysis": "string (analysis of different road segments)"
                },
                "trajectory_insights": {
                    "trajectory_patterns": "string (description of vehicle trajectory patterns)",
                    "intersection_behavior": "string (analysis of intersection behavior)",
                    "lane_changing_patterns": "string (analysis of lane changing behavior)",
                    "speed_spatial_correlation": "string (correlation between speed and spatial location)"
                }
            },
            "overall_assessment": "string (comprehensive assessment of traffic conditions)"
        }
        
        instruction = (
            "基于提供的交通特征数据，进行深度语义分析。特别关注时空特征的分析，包括：\n"
            "1. 识别交通模式、检测异常、评估拥堵情况\n"
            "2. 分析安全风险、提供效率洞察\n"
            "3. 深入分析时空特征：时间模式（日、周、季节性变化）、空间模式（分布、热点、聚类）、轨迹洞察（轨迹模式、交叉口行为、变道模式、速度空间相关性）\n"
            "4. 给出整体评估和优化建议"
        )
        
        result = self.agent.extract_information_direct(
            instruction=instruction,
            text=analysis_text,
            schema_obj=schema
        )
        
        return result
    
    def analyze_spatiotemporal_features(self, features_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        专门分析时空特征
        
        Args:
            features_data: 特征数据
            
        Returns:
            时空特征分析结果
        """
        if 'spatiotemporal_features' not in features_data:
            return {"error": "未找到时空特征数据"}
        
        spatiotemporal_data = features_data['spatiotemporal_features']
        analysis_text = self._build_spatiotemporal_analysis_text(spatiotemporal_data)
        
        schema = {
                    "temporal_analysis": {
                        "daily_patterns": "string (详细描述日交通模式)",
                        "weekly_patterns": "string (详细描述周交通模式)",
                        "peak_hours": ["list of peak traffic hours"],
                        "off_peak_characteristics": "string (描述非高峰时段特征)",
                        "temporal_anomalies": ["list of temporal anomalies detected"]
                    },
                    "spatial_analysis": {
                        "spatial_distribution": "string (详细描述空间分布特征)",
                        "traffic_hotspots": ["list of identified traffic hotspots with coordinates"],
                        "spatial_clustering": "string (描述空间聚类模式)",
                        "road_segment_characteristics": "object (不同道路段的特征分析)",
                        "spatial_anomalies": ["list of spatial anomalies detected"]
                    },
                    "trajectory_analysis": {
                        "trajectory_patterns": "string (详细描述轨迹模式)",
                        "intersection_behavior": "string (分析交叉口行为特征)",
                        "lane_changing_analysis": "string (变道行为分析)",
                        "speed_spatial_correlation": "string (速度与空间位置的相关性分析)",
                        "trajectory_anomalies": ["list of trajectory anomalies detected"]
                    },
                    "spatiotemporal_correlations": {
                        "time_space_interaction": "string (时间与空间的交互模式)",
                        "traffic_flow_dynamics": "string (交通流动态特征)",
                        "predictive_insights": "string (基于时空特征的预测洞察)"
                    }
                }

        instruction = (
                    "深入分析交通数据的时空特征，重点关注：\n"
                    "1. 时间模式：日变化、周变化、高峰时段、非高峰特征、时间异常\n"
                    "2. 空间模式：空间分布、交通热点、空间聚类、道路段特征、空间异常\n"
                    "3. 轨迹分析：轨迹模式、交叉口行为、变道分析、速度空间相关性、轨迹异常\n"
                    "4. 时空相关性：时空交互、交通流动态、预测洞察\n"
                    "请提供详细的分析结果和洞察。"
                )
        
        result = self.agent.extract_information_direct(
            instruction=instruction,
            text=analysis_text,
            schema_obj=schema
        )
        
        return result
    
    def _build_spatiotemporal_analysis_text(self, spatiotemporal_data: Dict[str, Any]) -> str:
        """
        构建时空特征分析文本
        
        Args:
            spatiotemporal_data: 时空特征数据
            
        Returns:
            分析文本
        """
        text_parts = ["时空特征详细分析:"]
        
        # 时间模式
        if 'temporal_patterns' in spatiotemporal_data:
            temporal = spatiotemporal_data['temporal_patterns']
            text_parts.append("\n时间模式分析:")
            for key, value in temporal.items():
                text_parts.append(f"  {key}: {value}")
        
        # 空间模式
        if 'spatial_patterns' in spatiotemporal_data:
            spatial = spatiotemporal_data['spatial_patterns']
            text_parts.append("\n空间模式分析:")
            for key, value in spatial.items():
                text_parts.append(f"  {key}: {value}")
        
        # 轨迹特征
        if 'trajectory_features' in spatiotemporal_data:
            trajectory = spatiotemporal_data['trajectory_features']
            text_parts.append("\n轨迹特征分析:")
            for key, value in trajectory.items():
                text_parts.append(f"  {key}: {value}")
        
        return "\n".join(text_parts)
    
    def _build_analysis_text(self, features_data: Dict[str, Any]) -> str:
        """
        构建分析文本
        
        Args:
            features_data: 特征数据
            
        Returns:
            分析文本
        """
        text_parts = []
        
        # 检查数据有效性
        data_valid = self._check_data_validity(features_data)
        if not data_valid:
            text_parts.append("警告：检测到数据质量问题，可能是由于数据文件缺失或加载失败导致的。")
        
        # 基础特征摘要
        if 'basic_features' in features_data:
            text_parts.append("基础特征摘要:")
            for feature_name, stats in features_data['basic_features'].items():
                if isinstance(stats, dict):
                    mean_val = stats.get('mean', 'N/A')
                    std_val = stats.get('std', 'N/A')
                    min_val = stats.get('min', 'N/A')
                    max_val = stats.get('max', 'N/A')
                    
                    # 安全地格式化数值
                    mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else str(mean_val)
                    std_str = f"{std_val:.2f}" if isinstance(std_val, (int, float)) else str(std_val)
                    min_str = f"{min_val:.2f}" if isinstance(min_val, (int, float)) else str(min_val)
                    max_str = f"{max_val:.2f}" if isinstance(max_val, (int, float)) else str(max_val)
                    
                    text_parts.append(f"- {feature_name}: 均值={mean_str}, "
                                   f"标准差={std_str}, "
                                   f"范围=[{min_str}, {max_str}]")
                else:
                    text_parts.append(f"- {feature_name}: {stats}")
        
        # 交通特定特征
        if 'traffic_features' in features_data:
            text_parts.append("\n交通特定特征:")
            for feature_name, feature_values in features_data['traffic_features'].items():
                if isinstance(feature_values, pd.Series):
                    # 统计分类特征
                    value_counts = feature_values.value_counts()
                    text_parts.append(f"- {feature_name}: {dict(value_counts)}")
                else:
                    text_parts.append(f"- {feature_name}: {feature_values}")
        
        # 统计特征
        if 'statistical_features' in features_data:
            text_parts.append("\n统计特征:")
            for feature_name, feature_values in features_data['statistical_features'].items():
                if isinstance(feature_values, pd.Series):
                    # 计算统计摘要
                    if feature_values.dtype in ['float64', 'int64']:
                        mean_val = feature_values.mean()
                        std_val = feature_values.std()
                        text_parts.append(f"- {feature_name}: 均值={mean_val:.2f}, "
                                       f"标准差={std_val:.2f}")
                elif isinstance(feature_values, dict):
                    # 处理字典类型的统计特征
                    text_parts.append(f"- {feature_name}: {feature_values}")
                else:
                    text_parts.append(f"- {feature_name}: {feature_values}")
        
        # 时空特征 - 重点分析
        if 'spatiotemporal_features' in features_data:
            text_parts.append("\n时空特征（重点分析）:")
            spatiotemporal = features_data['spatiotemporal_features']
            
            # 时间模式分析
            if 'temporal_patterns' in spatiotemporal:
                temporal = spatiotemporal['temporal_patterns']
                text_parts.append("时间模式:")
                if 'hourly_distribution' in temporal:
                    text_parts.append(f"  - 小时分布: {temporal['hourly_distribution']}")
                if 'daily_patterns' in temporal:
                    text_parts.append(f"  - 日模式: {temporal['daily_patterns']}")
                if 'seasonal_trends' in temporal:
                    text_parts.append(f"  - 季节性趋势: {temporal['seasonal_trends']}")
            
            # 空间模式分析
            if 'spatial_patterns' in spatiotemporal:
                spatial = spatiotemporal['spatial_patterns']
                text_parts.append("空间模式:")
                if 'location_distribution' in spatial:
                    text_parts.append(f"  - 位置分布: {spatial['location_distribution']}")
                if 'spatial_clustering' in spatial:
                    text_parts.append(f"  - 空间聚类: {spatial['spatial_clustering']}")
                if 'road_segments' in spatial:
                    text_parts.append(f"  - 道路段: {spatial['road_segments']}")
            
            # 轨迹特征分析
            if 'trajectory_features' in spatiotemporal:
                trajectory = spatiotemporal['trajectory_features']
                text_parts.append("轨迹特征:")
                if 'avg_trajectory_length' in trajectory:
                    text_parts.append(f"  - 平均轨迹长度: {trajectory['avg_trajectory_length']}")
                if 'trajectory_complexity' in trajectory:
                    text_parts.append(f"  - 轨迹复杂度: {trajectory['trajectory_complexity']}")
                if 'intersection_patterns' in trajectory:
                    text_parts.append(f"  - 交叉口模式: {trajectory['intersection_patterns']}")
        
        return "\n".join(text_parts)
    
    def _check_data_validity(self, features_data: Dict[str, Any]) -> bool:
        """
        检查特征数据的有效性
        
        Args:
            features_data: 特征数据
            
        Returns:
            数据是否有效
        """
        # 检查基础特征是否全为0（表示数据加载失败）
        if 'basic_features' in features_data and 'numeric_columns' in features_data['basic_features']:
            for col_name, stats in features_data['basic_features']['numeric_columns'].items():
                if isinstance(stats, dict):
                    # 如果所有统计值都是0，可能表示数据有问题
                    if (stats.get('mean', 0) == 0 and 
                        stats.get('std', 0) == 0 and 
                        stats.get('min', 0) == 0 and 
                        stats.get('max', 0) == 0):
                        return False
        
        # 检查交通特征是否有意义
        if 'traffic_features' in features_data:
            traffic_features = features_data['traffic_features']
            if (traffic_features.get('vehicle_count', 0) == 0 and 
                traffic_features.get('speed_analysis', {}).get('avg_speed', 0) == 0):
                return False
        
        return True
    
    def save_analysis(self, analysis_result: Dict[str, Any], output_dir: str = "output") -> str:
        """
        保存分析结果
        
        Args:
            analysis_result: 分析结果
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加时间戳
        analysis_result['analysis_timestamp'] = datetime.now().isoformat()
        
        # 保存到文件
        output_path = os.path.join(output_dir, "semantic_mcp.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"语义分析结果已保存到 {output_path}")
        return output_path


def run(feature_json: str, csv_path: str = None, api_key: str = None,
        model: str = "qwen-turbo", output_dir: str = "output") -> str:
    """
    运行语义分析的主函数

    Args:
        feature_json: 特征JSON文件路径
        csv_path: CSV文件路径（可选）
        api_key: API密钥
        model: 模型名称
        output_dir: 输出目录

    Returns:
        分析结果文件路径
    """
    print(f"  api_key: {'已提供' if api_key else '未提供'}")  # 显示是否提供了API密钥
    try:
        # 加载特征数据
        with open(feature_json, 'r', encoding='utf-8') as f:
            features_data = json.load(f)

        # 创建分析器
        analyzer = TrafficSemanticAnalyzer(api_key, model)

        # 进行语义分析
        analysis_result = analyzer.analyze_traffic_patterns(features_data)

        # 如果有时空特征，进行专门的时空特征分析
        if 'spatiotemporal_features' in features_data:
            print("🔍 检测到时空特征，进行专门的时空特征语义分析...")
            spatiotemporal_analysis = analyzer.analyze_spatiotemporal_features(features_data)
            analysis_result['detailed_spatiotemporal_analysis'] = spatiotemporal_analysis

        # 保存结果
        output_path = analyzer.save_analysis(analysis_result, output_dir)

        return output_path

    except Exception as e:
        logger.error(f"语义分析失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="交通语义分析")
    parser.add_argument("--features", required=True, help="特征JSON文件路径")
    parser.add_argument("--csv", help="CSV文件路径（可选）")
    parser.add_argument("--model", default="qwen-turbo", help="模型名称")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--api_key", required=True, help="阿里云DashScope API密钥")

    args = parser.parse_args()
    
    run(args.features, args.csv, args.api_key, args.model, args.output)
