import json
import torch
import numpy as np
from typing import Dict, List, Union
import math
import os
from semantic_communication import SemCom_linear, SemCom_KAN, semantic


class JSONToSemCom:
    """将JSON交通数据转换为384维特征向量并进行语义通信传输"""

    def __init__(self, use_kan=True, device="cpu"):
        """
        初始化JSON到语义通信转换器

        Args:
            use_kan (bool): 是否使用KAN模型，默认True
            device (str): 计算设备
        """
        self.device = device
        self.use_kan = use_kan

        # 初始化语义通信模型
        if use_kan:
            self.model = SemCom_KAN().to(device)
        else:
            self.model = SemCom_linear().to(device)

    def extract_numeric_features(self, data: Dict) -> List[float]:
        """从basic_features中提取数值特征"""
        features = []

        if 'basic_features' in data and 'numeric_columns' in data['basic_features']:
            numeric_cols = data['basic_features']['numeric_columns']

            # 为每个数值列提取统计特征
            for col_name, stats in numeric_cols.items():
                features.extend([
                    stats.get('mean', 0.0),
                    stats.get('std', 0.0),
                    stats.get('min', 0.0),
                    stats.get('max', 0.0),
                    stats.get('median', 0.0),
                    stats.get('skewness', 0.0),
                    stats.get('kurtosis', 0.0)
                ])

        return features

    def extract_traffic_features(self, data: Dict) -> List[float]:
        """从traffic_features中提取交通特征"""
        features = []

        if 'traffic_features' in data:
            tf = data['traffic_features']

            # 车辆数量特征
            features.append(tf.get('vehicle_count', 0))

            # 时间范围特征
            if 'time_range' in tf:
                tr = tf['time_range']
                features.extend([
                    float(tr.get('start_time', 0)),
                    float(tr.get('end_time', 0)),
                    tr.get('duration_minutes', 0.0)
                ])

            # 速度分析特征
            if 'speed_analysis' in tf:
                sa = tf['speed_analysis']
                features.extend([
                    sa.get('avg_speed', 0.0),
                    sa.get('speed_variance', 0.0)
                ])

                # 将速度分布转换为数值
                speed_dist = sa.get('speed_distribution', 'unknown')
                if speed_dist == 'normal':
                    features.append(1.0)
                elif speed_dist == 'skewed':
                    features.append(2.0)
                else:
                    features.append(0.0)

            # 空间分析特征
            if 'spatial_analysis' in tf:
                spa = tf['spatial_analysis']
                features.extend([
                    spa.get('road_length', 0.0)
                ])

                # 将空间覆盖转换为数值
                spatial_coverage = spa.get('spatial_coverage', 'unknown')
                if spatial_coverage == 'comprehensive':
                    features.append(1.0)
                elif spatial_coverage == 'limited':
                    features.append(0.5)
                else:
                    features.append(0.0)

        return features

    def extract_spatiotemporal_features(self, data: Dict) -> List[float]:
        """从spatiotemporal_features中提取时空特征"""
        features = []

        if 'spatiotemporal_features' in data:
            stf = data['spatiotemporal_features']

            # 时间模式特征
            if 'temporal_patterns' in stf:
                tp = stf['temporal_patterns']

                # 小时分布
                hourly_dist = tp.get('hourly_distribution', {})
                hourly_features = [0.0] * 24  # 24小时特征向量
                for hour, count in hourly_dist.items():
                    if hour.isdigit() and 0 <= int(hour) <= 23:
                        hourly_features[int(hour)] = float(count)
                features.extend(hourly_features)

                # 日模式
                daily_patterns = tp.get('daily_patterns', {})
                daily_features = [0.0] * 7  # 7天特征向量
                for day, count in daily_patterns.items():
                    if day.isdigit() and 0 <= int(day) <= 6:
                        daily_features[int(day)] = float(count)
                features.extend(daily_features)

            # 空间模式特征
            if 'spatial_patterns' in stf:
                sp = stf['spatial_patterns']

                # 位置分布 - X轴bins
                if 'location_distribution' in sp and 'x_bins' in sp['location_distribution']:
                    x_bins = sp['location_distribution']['x_bins']
                    x_features = [0.0] * 10  # 10个X轴bins
                    for i, (bin_range, count) in enumerate(x_bins.items()):
                        if i < 10:
                            x_features[i] = float(count)
                    features.extend(x_features)

                # 位置分布 - Y轴bins
                if 'location_distribution' in sp and 'y_bins' in sp['location_distribution']:
                    y_bins = sp['location_distribution']['y_bins']
                    y_features = [0.0] * 10  # 10个Y轴bins
                    for i, (bin_range, count) in enumerate(y_bins.items()):
                        if i < 10:
                            y_features[i] = float(count)
                    features.extend(y_features)

                # 路段特征
                if 'road_segments' in sp:
                    road_segments = sp['road_segments']
                    road_features = [0.0] * 15  # 最多15个路段特征 (5个路段 * 3个特征)
                    for i, (segment, info) in enumerate(road_segments.items()):
                        if i < 5:  # 最多处理5个路段
                            road_features[i * 3] = float(info.get('vehicle_count', 0))
                            road_features[i * 3 + 1] = float(info.get('avg_speed', 0))
                            road_features[i * 3 + 2] = 1.0 if info else 0.0
                    features.extend(road_features)

            # 轨迹特征
            if 'trajectory_features' in stf:
                traj = stf['trajectory_features']
                features.extend([
                    traj.get('avg_trajectory_length', 0.0)
                ])

                # 轨迹复杂度转换为数值
                complexity = traj.get('trajectory_complexity', 'unknown')
                if complexity == '轨迹较直':
                    features.append(0.5)
                elif complexity == '轨迹复杂':
                    features.append(1.0)
                else:
                    features.append(0.0)

        return features

    def extract_analysis_features(self, data: Dict) -> List[float]:
        """从spatiotemporal_analysis中提取分析特征"""
        features = []

        if 'spatiotemporal_analysis' in data:
            sta = data['spatiotemporal_analysis']

            # 时间分析特征
            if 'temporal_analysis' in sta:
                ta = sta['temporal_analysis']

                # 高峰时段
                peak_hours = ta.get('peak_hours', [])
                peak_features = [0.0] * 24
                for hour in peak_hours:
                    if hour.isdigit() and 0 <= int(hour) <= 23:
                        peak_features[int(hour)] = 1.0
                features.extend(peak_features)

            # 空间分析特征
            if 'spatial_analysis' in sta:
                spa = sta['spatial_analysis']

                # 交通热点
                hotspots = spa.get('traffic_hotspots', [])
                hotspot_features = [0.0] * 20  # 最多20个热点
                for i, hotspot in enumerate(hotspots):
                    if i < 20:
                        hotspot_features[i] = 1.0
                features.extend(hotspot_features)

                # 路段特征
                if 'road_segment_characteristics' in spa:
                    road_chars = spa['road_segment_characteristics']
                    road_char_features = [0.0] * 15  # 最多15个路段特征
                    for i, (segment, info) in enumerate(road_chars.items()):
                        if i < 5:  # 最多处理5个路段
                            road_char_features[i * 3] = float(info.get('vehicle_count', 0))
                            road_char_features[i * 3 + 1] = float(info.get('avg_speed', 0))
                            road_char_features[i * 3 + 2] = 1.0 if info else 0.0
                    features.extend(road_char_features)

            # 轨迹分析特征
            if 'trajectory_analysis' in sta:
                traj_analysis = sta['trajectory_analysis']

                # 轨迹异常
                anomalies = traj_analysis.get('trajectory_anomalies', [])
                anomaly_features = [0.0] * 10  # 最多10个异常特征
                for i, anomaly in enumerate(anomalies):
                    if i < 10:
                        anomaly_features[i] = 1.0 if "异常" in anomaly else 0.0
                features.extend(anomaly_features)

        return features

    def load_json_from_file(self, file_path: str) -> Dict:
        """
        从JSON文件加载数据

        Args:
            file_path (str): JSON文件路径

        Returns:
            Dict: 加载的JSON数据

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON文件格式错误: {e}")
        except Exception as e:
            raise Exception(f"加载JSON文件失败: {e}")

    def save_json_to_file(self, data: Dict, file_path: str, indent: int = 2):
        """
        将数据保存为JSON文件

        Args:
            data (Dict): 要保存的数据
            file_path (str): 保存路径
            indent (int): JSON缩进，默认2
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
        except Exception as e:
            raise Exception(f"保存JSON文件失败: {e}")

    def json_to_features(self, json_data: Union[str, Dict]) -> torch.Tensor:
        """
        将JSON数据转换为384维特征向量

        Args:
            json_data: JSON字符串或字典

        Returns:
            torch.Tensor: 384维特征向量
        """
        # 解析JSON数据
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        # 提取各种特征
        features = []
        features.extend(self.extract_numeric_features(data))
        features.extend(self.extract_traffic_features(data))
        features.extend(self.extract_spatiotemporal_features(data))
        features.extend(self.extract_analysis_features(data))

        # 确保特征向量长度为384
        if len(features) < 384:
            # 如果特征不足，用0填充
            features.extend([0.0] * (384 - len(features)))
        elif len(features) > 384:
            # 如果特征过多，截断
            features = features[:384]

        # 转换为tensor并归一化
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # 添加batch维度

        # 简单的归一化
        features_tensor = torch.tanh(features_tensor)  # 使用tanh归一化到[-1,1]

        return features_tensor.to(self.device)

    def features_to_json_approximation(self, features: torch.Tensor) -> Dict:
        """
        将解码后的特征向量转换回JSON格式的近似表示
        注意：这是一个简化的反向映射，实际应用中可能需要更复杂的重构方法

        Args:
            features: 解码后的特征向量

        Returns:
            Dict: 近似的JSON结构
        """
        # 安全转换
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = np.array(features)

        features_np = features_np.flatten()

        # 创建基本的JSON结构
        result = {
            "reconstructed_features": {
                "numeric_features": features_np[:98].tolist(),  # 前98个特征对应数值特征
                "traffic_features": features_np[98:106].tolist(),  # 接下来8个特征对应交通特征
                "spatiotemporal_features": features_np[106:206].tolist(),  # 接下来100个特征对应时空特征
                "analysis_features": features_np[206:384].tolist()  # 剩余特征对应分析特征
            },
            "feature_statistics": {
                "mean": float(np.mean(features_np)),
                "std": float(np.std(features_np)),
                "min": float(np.min(features_np)),
                "max": float(np.max(features_np))
            }
        }

        return result

    def transmit_json_from_file(self, file_path: str, snr: float = 20.0) -> Dict:
        """
        从JSON文件加载数据并进行传输

        Args:
            file_path (str): JSON文件路径
            snr (float): 信噪比

        Returns:
            Dict: 传输结果，包含原始数据、传输数据、接收数据和重构数据
        """
        # 1. 从文件加载JSON数据
        json_data = self.load_json_from_file(file_path)

        # 2. 进行传输
        result = self.transmit_json(json_data, snr)

        # 3. 添加文件信息
        result["source_file"] = file_path
        result["file_size"] = os.path.getsize(file_path)

        return result

    def transmit_json(self, json_data: Union[str, Dict], snr: float = 20.0) -> Dict:
        try:
            # 1. 加载JSON数据
            if isinstance(json_data, str):
                if os.path.exists(json_data):
                    with open(json_data, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = json.loads(json_data)
            else:
                data = json_data

            # 2. 转换为特征张量
            features = self.json_to_features(data)

            # 3. 语义传输
            self.model.eval()
            with torch.no_grad():
                transmitted = self.model(features, snr)

            # 4. 安全转换为JSON
            return {
                "original_features": self._safe_tensor_to_json(features),
                "transmitted_features": self._safe_tensor_to_json(transmitted),
                "reconstructed_json": self.features_to_json_approximation(transmitted),
                "transmission_info": {
                    "snr": snr,
                    "model_type": "KAN" if self.use_kan else "Linear",
                    "device": str(self.device),
                    "status": "success"
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "transmission_info": {
                    "snr": snr,
                    "model_type": "KAN" if self.use_kan else "Linear",
                    "device": str(self.device),
                    "status": "failed"
                }
            }

    def batch_transmit_from_files(self, file_paths: List[str], snr: float = 20.0) -> List[Dict]:
        """
        批量从文件传输JSON数据

        Args:
            file_paths (List[str]): JSON文件路径列表
            snr (float): 信噪比

        Returns:
            List[Dict]: 批量传输结果列表
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.transmit_json_from_file(file_path, snr)
                results.append(result)
            except Exception as e:
                # 如果某个文件传输失败，记录错误但继续处理其他文件
                error_result = {
                    "error": str(e),
                    "source_file": file_path,
                    "transmission_info": {"snr": snr, "model_type": "KAN" if self.use_kan else "Linear"}
                }
                results.append(error_result)
        return results

    def batch_transmit(self, json_list: List[Union[str, Dict]], snr: float = 20.0) -> List[Dict]:
        """
        批量传输JSON数据

        Args:
            json_list: JSON数据列表
            snr: 信噪比

        Returns:
            List[Dict]: 批量传输结果列表
        """
        results = []
        for json_data in json_list:
            result = self.transmit_json(json_data, snr)
            results.append(result)
        return results


# 使用示例
if __name__ == "__main__":
    # 示例JSON数据
    sample_json = {
        "basic_features": {
            "numeric_columns": {
                "Vehicle_ID": {"mean": 2.0, "std": 0.0, "min": 2.0, "max": 2.0, "median": 2.0, "skewness": 0.0,
                               "kurtosis": 0.0},
                "Frame_ID": {"mean": 225.5, "std": 127.4368986824586, "min": 13.0, "max": 442.0, "median": 225.5,
                             "skewness": 0.0, "kurtosis": -1.2}
            }
        },
        "traffic_features": {
            "vehicle_count": 430,
            "time_range": {"start_time": "1118846980200", "end_time": "1118847023100", "duration_minutes": 0.0},
            "speed_analysis": {"avg_speed": 54.92345, "speed_variance": 300.0, "speed_distribution": "normal"}
        }
    }

    model = JSONToSemCom()

    json_data = model.load_json_from_file("output/features_mcp.json")

    # 初始化转换器
    converter = JSONToSemCom(use_kan=True, device="cpu")

    # 传输数据
    result = converter.transmit_json(json_data, snr=20.0)

    print(result)

    print("传输完成！")
    #print(f"原始特征维度: {len(result['original_features'][0])}")
    #print(f"传输特征维度: {len(result['transmitted_features'][0])}")
    print(f"SNR: {result['transmission_info']['snr']}")
    print(f"模型类型: {result['transmission_info']['model_type']}")
