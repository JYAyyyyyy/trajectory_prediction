import json
import torch
import numpy as np
from typing import Dict, Any, List


class StructuredJSONToKANInputConverter:
    """将结构化交通特征JSON转换为KAN模型输入的转换器"""

    def __init__(self):
        self.target_dim = 384

    def extract_basic_numeric_features(self, numeric_data: Dict[str, Any]) -> List[float]:
        """提取基础数值统计特征"""
        features = []

        # 定义重要的数值列及其统计量
        important_columns = ['Local_X', 'Local_Y', 'Global_X', 'v_Vel', 'v_Acc', 'Frame_ID']
        important_stats = ['mean', 'std', 'min', 'max']

        for col_name in important_columns:
            if col_name in numeric_data:
                col_stats = numeric_data[col_name]
                for stat in important_stats:
                    if stat in col_stats:
                        features.append(float(col_stats[stat]))

        return features

    def extract_all_numeric_features(self, numeric_data: Dict[str, Any]) -> List[float]:
        """提取所有数值列的统计特征"""
        features = []

        # 按顺序处理所有数值列
        for col_name, col_stats in numeric_data.items():
            # 提取主要统计量
            for stat in ['mean', 'std', 'min', 'max', 'median']:
                if stat in col_stats:
                    features.append(float(col_stats[stat]))

        return features

    def extract_traffic_features(self, traffic_data: Dict[str, Any]) -> List[float]:
        """提取交通特征"""
        features = []

        # 车辆数量
        features.append(float(traffic_data.get('vehicle_count', 0)))

        # 速度分析
        speed_analysis = traffic_data.get('speed_analysis', {})
        features.append(speed_analysis.get('avg_speed', 0.0))
        features.append(speed_analysis.get('speed_variance', 0.0))

        # 时间范围
        time_range = traffic_data.get('time_range', {})
        features.append(float(time_range.get('duration_minutes', 0.0)))

        return features

    def extract_spatial_features(self, spatio_data: Dict[str, Any]) -> List[float]:
        """提取空间特征"""
        features = []
        spatial_patterns = spatio_data.get('spatial_patterns', {})

        # 位置分布（x_bins和y_bins）
        location_dist = spatial_patterns.get('location_distribution', {})

        # x_bins特征
        x_bins = location_dist.get('x_bins', {})
        x_values = list(x_bins.values())
        if x_values:
            features.extend([
                len(x_values),  # x分桶数量
                sum(x_values),  # x方向总车辆数
                max(x_values),  # x方向最大车辆数
                min(x_values),  # x方向最小车辆数
                sum(x_values) / len(x_values)  # x方向平均车辆数
            ])
            # 添加前5个分桶的车辆数
            features.extend(list(x_values)[:5])
        else:
            features.extend([0, 0, 0, 0, 0] + [0] * 5)

        # y_bins特征
        y_bins = location_dist.get('y_bins', {})
        y_values = list(y_bins.values())
        if y_values:
            features.extend([
                len(y_values),  # y分桶数量
                sum(y_values),  # y方向总车辆数
                max(y_values),  # y方向最大车辆数
                min(y_values),  # y方向最小车辆数
                sum(y_values) / len(y_values)  # y方向平均车辆数
            ])
            # 添加前5个分桶的车辆数
            features.extend(list(y_values)[:5])
        else:
            features.extend([0, 0, 0, 0, 0] + [0] * 5)

        return features

    def extract_road_segment_features(self, spatio_data: Dict[str, Any]) -> List[float]:
        """提取道路段特征（最重要的数值特征）"""
        features = []
        spatial_patterns = spatio_data.get('spatial_patterns', {})
        road_segments = spatial_patterns.get('road_segments', {})

        vehicle_counts = []
        speeds = []

        # 提取每个道路段的详细数据
        for seg_name, seg_data in road_segments.items():
            vehicle_count = seg_data.get('vehicle_count', 0)
            avg_speed = seg_data.get('avg_speed', 0.0)

            vehicle_counts.append(vehicle_count)
            speeds.append(avg_speed)

            # 直接添加每个道路段的特征
            features.append(vehicle_count)
            features.append(avg_speed)

        # 添加道路段统计特征
        features.append(len(road_segments))  # 道路段数量

        if vehicle_counts:
            features.extend([
                sum(vehicle_counts),  # 总车辆数
                max(vehicle_counts),  # 最大车辆数
                min(vehicle_counts),  # 最小车辆数
                sum(vehicle_counts) / len(vehicle_counts)  # 平均车辆数
            ])
        else:
            features.extend([0, 0, 0, 0])

        if speeds:
            features.extend([
                sum(speeds),  # 总速度
                max(speeds),  # 最大速度
                min(speeds),  # 最小速度
                sum(speeds) / len(speeds)  # 平均速度
            ])
        else:
            features.extend([0, 0, 0, 0])

        return features

    def extract_trajectory_features(self, spatio_data: Dict[str, Any]) -> List[float]:
        """提取轨迹特征"""
        features = []
        trajectory_features = spatio_data.get('trajectory_features', {})

        # 轨迹长度
        features.append(trajectory_features.get('avg_trajectory_length', 0.0))

        # 轨迹复杂度（文本编码）
        complexity = trajectory_features.get('trajectory_complexity', '')
        features.append(1.0 if "直" in complexity else 0.5 if "复杂" in complexity else 0.0)

        return features

    def extract_all_features(self, data: Dict[str, Any]) -> List[float]:
        """提取所有特征"""
        features = []

        # 1. 基础数值特征（所有列的统计量）
        basic_features = data.get('basic_features', {})
        numeric_data = basic_features.get('numeric_columns', {})
        features.extend(self.extract_all_numeric_features(numeric_data))

        # 2. 交通特征
        traffic_data = data.get('traffic_features', {})
        features.extend(self.extract_traffic_features(traffic_data))

        # 3. 时空特征
        spatio_data = data.get('spatiotemporal_features', {})
        features.extend(self.extract_spatial_features(spatio_data))
        features.extend(self.extract_road_segment_features(spatio_data))
        features.extend(self.extract_trajectory_features(spatio_data))

        return features

    def smart_padding(self, features: List[float]) -> List[float]:
        """智能填充到目标维度"""
        current_len = len(features)

        if current_len >= self.target_dim:
            return features[:self.target_dim]

        padding_needed = self.target_dim - current_len

        if padding_needed > 0:
            # 策略1：使用重要特征进行重复填充
            important_features = features[:min(50, len(features))]  # 前50个重要特征

            if important_features:
                repeat_count = padding_needed // len(important_features)
                remainder = padding_needed % len(important_features)

                padded = features + important_features * repeat_count
                padded += important_features[:remainder]
                return padded

        # 策略2：零填充
        return features + [0.0] * padding_needed

    def normalize_features(self, features: List[float]) -> List[float]:
        """简单特征归一化"""
        if not features:
            return features

        # 转换为numpy数组便于计算
        arr = np.array(features)

        # 避免除零错误
        std = np.std(arr)
        if std < 1e-10:
            return features

        # Z-score归一化
        normalized = (arr - np.mean(arr)) / std

        # 缩放到[-1, 1]范围
        normalized = np.clip(normalized, -3.0, 3.0) / 3.0

        return normalized.tolist()

    def convert(self, json_data: Dict[str, Any], normalize: bool = True) -> torch.Tensor:
        """将JSON数据转换为KAN模型输入"""
        # 提取所有特征
        features = self.extract_all_features(json_data)

        # 可选：归一化特征
        if normalize:
            features = self.normalize_features(features)

        # 填充到目标维度
        features = self.smart_padding(features)

        print(f"提取特征数量: {len(features)}")

        # 转换为张量并添加批次维度
        return torch.FloatTensor(features).unsqueeze(0)  # [1, 384]


# 使用示例
def load_and_convert_structured_json(json_path: str) -> torch.Tensor:
    """从结构化JSON文件加载数据并转换为KAN输入"""
    converter = StructuredJSONToKANInputConverter()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return converter.convert(data)


if __name__ == "__main__":

    input_tensor = load_and_convert_structured_json("E:\LearningData\PHD\Experiment\llm agent\\agent\mcp\output\semantic_mcp.json")
    print(f"输入张量形状: {input_tensor.shape}")