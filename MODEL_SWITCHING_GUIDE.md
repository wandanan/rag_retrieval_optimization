# 模型切换逻辑使用指南

## 概述

本系统支持两种嵌入模型后端：
- **BGE (本地模型)**: 使用本地下载的模型文件
- **HuggingFace (在线模型)**: 自动从HuggingFace下载模型

## 前端界面

### 编码后端选择
- 选择 "BGE (本地模型)" 时，显示BGE模型路径输入框
- 选择 "HuggingFace (在线模型)" 时，显示HF模型名输入框
- 切换时会自动清空另一个输入框

### 模型配置要求

#### BGE本地模型
- **编码后端**: 选择 "BGE (本地模型)"
- **BGE模型路径**: 必须填写本地模型路径
  - 例如: `models--BAAI--bge-small-zh-v1.5/snapshots/xxx`
  - 需要先下载模型到本地

#### HuggingFace在线模型
- **编码后端**: 选择 "HuggingFace (在线模型)"
- **HF模型名**: 必须填写模型名称
  - 例如: `BAAI/bge-small-zh-v1.5`
  - 系统会自动下载模型

#### 重排序模型
- 当启用重排序时，必须填写重排序模型名称
- 支持BGE和HF格式的模型名

## 后端配置

### 预设配置
所有预设配置都已移除硬编码的模型路径和名称：
- `balanced`: 平衡模式
- `precision`: 精确模式  
- `speed`: 快速模式
- `conversational`: 对话模式
- `hf_optimized`: HF优化模式

### 配置验证
系统会自动验证：
1. 编码后端选择对应的模型配置是否完整
2. 重排序模型名称是否填写（当启用时）
3. 数值参数是否在有效范围内

## 使用流程

1. **选择编码后端**: 根据需求选择BGE或HF
2. **填写模型信息**: 输入对应的模型路径或名称
3. **配置其他参数**: 设置权重、数量等参数
4. **验证配置**: 使用"验证配置"按钮检查配置完整性
5. **保存配置**: 保存当前配置到本地存储

## 注意事项

- 使用BGE本地模型时，确保模型文件已下载到指定路径
- 使用HF在线模型时，首次使用需要网络连接下载模型
- 重排序模型名称必须与编码后端兼容
- 所有模型配置都支持动态切换，无需重启系统

## 错误处理

配置验证失败时会显示具体错误信息：
- ❌ BGE模型路径不能为空
- ❌ HF模型名不能为空  
- ❌ 重排序模型名不能为空
- 其他参数范围错误

## 示例配置

### BGE本地模型配置
```json
{
  "encoder_backend": "bge",
  "bge_model_path": "models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
  "hf_model_name": "",
  "use_reranker": true,
  "reranker_model_name": "BAAI/bge-reranker-large"
}
```

### HF在线模型配置
```json
{
  "encoder_backend": "hf", 
  "bge_model_path": "",
  "hf_model_name": "BAAI/bge-small-zh-v1.5",
  "use_reranker": true,
  "reranker_model_name": "BAAI/bge-reranker-large"
}
``` 