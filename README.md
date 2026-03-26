# MCP Agent 交通数据分析系统

这是一个基于阿里云DashScope MCP agent的交通数据分析系统，包含三个核心模块：

## 系统架构

```
特征提取 → 语义分析 → 轨迹预测
    ↓           ↓         ↓
MCP Agent   MCP Agent  MCP Agent
```

## 模块说明

### 1. 特征提取模块 (`feature_extraction.py`)

**功能**: 使用MCP agent从US101交通数据中提取多种特征
- **基础特征**: 数值列的统计特征（均值、标准差、偏度、峰度等）
- **交通特定特征**: 车辆数量、时间范围、速度分析、空间分析
- **时空特征**: 时间模式、空间模式、轨迹特征
- **统计特征**: 相关性分析、异常值检测、分布分析

**Agent作用**: 智能分析数据并生成结构化的特征描述

### 2. 语义分析模块 (`semantic_analysis.py`)

**功能**: 基于提取的特征进行语义分析
- 交通模式识别
- 异常检测
- 交通流分析
- 安全评估
- 效率分析

**Agent作用**: 将数值特征转换为可理解的语义描述

### 3. 轨迹预测模块 (`trajectory_prediction.py`)

**功能**: 预测车辆未来轨迹
- 位置预测
- 速度预测
- 车道预测
- 轨迹模式识别

**Agent作用**: 结合特征和语义信息进行智能轨迹预测

## 使用方法

### 前置要求

1. **安装依赖**
```bash
pip install pandas numpy requests
```

2. **设置API密钥**
在项目根目录创建 `api_key.txt` 文件，内容为你的DashScope API密钥：
```
your_actual_dashscope_api_key_here
```

### 运行方式

#### 方式1: 使用主程序（推荐）
```bash
python main.py --csv ../datas/0750_0805_us101.csv
```

#### 方式2: 单独运行模块
```bash
# 特征提取
python feature_extraction.py ../datas/0750_0805_us101.csv --api_key your_key

# 语义分析
python semantic_analysis.py --feature_json output/features_xxx.json --api_key your_key

# 轨迹预测
python trajectory_prediction.py --csv ../datas/0750_0805_us101.csv --vehicle_id 2 --api_key your_key
```


## 输出文件

系统会在 `output/` 目录下生成以下文件：

- `features_YYYYMMDD_HHMMSS.json` - 特征提取结果
- `semantic_YYYYMMDD_HHMMSS.json` - 语义分析结果  
- `trajectory_predictions.json` - 轨迹预测结果

## 故障排除

### 常见问题

1. **"MCP agent返回结果格式不正确，回退到传统方法"**
   - 原因: MCP agent返回的JSON格式不符合预期
   - 解决: 系统会自动回退到传统算法，不影响功能

2. **"API密钥无效或未提供"**
   - 检查 `api_key.txt` 文件是否存在且包含有效密钥
   - 确认API密钥有足够的余额和权限

3. **"连接超时"**
   - 检查网络连接
   - 增加超时时间参数

### 调试建议

1. 使用 `test_mcp_connection.py` 进行详细诊断
2. 检查日志输出中的错误信息
3. 确认CSV数据文件格式正确

## 技术特点

- **智能降级**: 当MCP agent失败时自动回退到传统算法
- **多编码支持**: 自动处理不同编码的CSV文件
- **错误恢复**: 内置重试机制和异常处理
- **模块化设计**: 三个模块可独立运行或组合使用

## 扩展性

系统设计为可扩展架构，可以：
- 添加新的特征提取方法
- 集成其他LLM服务
- 支持更多数据格式
- 添加可视化组件

## 注意事项

1. 确保API密钥安全，不要提交到版本控制系统
2. 大数据集处理时注意内存使用
3. 建议在测试环境中先验证功能
4. 定期检查API使用量和费用
