# 重排序功能使用指南

## 概述

V3 RAG引擎集成了强大的交叉编码器重排序功能，支持多种后端实现，能够显著提升检索结果的准确性。

## 支持的重排序后端

### 1. FlagEmbedding (推荐)
- **优点**: 性能最佳，内存占用低，支持FP16加速
- **安装**: `pip install FlagEmbedding`
- **适用场景**: 生产环境，需要最佳性能

### 2. HuggingFace Transformers
- **优点**: 兼容性好，支持模型微调
- **安装**: `pip install transformers torch`
- **适用场景**: 开发测试，需要自定义模型

### 3. ONNX Runtime
- **优点**: 推理速度快，跨平台支持
- **安装**: `pip install optimum[onnxruntime] onnxruntime`
- **适用场景**: 生产环境，需要快速推理

## 配置说明

### 基础配置
```python
# 启用重排序
use_reranker: True

# 重排序模型名称
reranker_model_name: "BAAI/bge-reranker-large"

# 重排序候选数量
reranker_top_n: 50

# 重排序权重
reranker_weight: 1.5

# 重排序后端选择
reranker_backend: "auto"  # auto, flagembedding, transformers, onnx
```

### 预设配置
- **平衡模式**: 重排序权重1.5，候选数量50
- **精确模式**: 重排序权重2.0，候选数量100
- **快速模式**: 禁用重排序
- **对话模式**: 重排序权重1.8，候选数量80

## 使用方法

### 1. 前端配置
在Web界面中：
1. 勾选"启用交叉编码器重排序"
2. 选择重排序模型（推荐：BAAI/bge-reranker-large）
3. 设置候选数量（建议：50-100）
4. 调整重排序权重（建议：1.0-2.0）
5. 选择重排序后端（推荐：自动选择）

### 2. 后端配置
```python
from advanced_zipper_engine_v3 import ZipperV3Config, AdvancedZipperQueryEngineV3

config = ZipperV3Config(
    use_reranker=True,
    reranker_model_name="BAAI/bge-reranker-large",
    reranker_top_n=50,
    reranker_weight=1.5,
    reranker_backend="auto"
)

engine = AdvancedZipperQueryEngineV3(config)
```

## 工作流程

1. **初始检索**: BM25 + ColBERT混合检索
2. **候选选择**: 选择前N个候选文档
3. **重排序**: 使用交叉编码器重新评分
4. **分数融合**: 融合原始分数和重排序分数
5. **最终排序**: 按融合分数排序返回结果

## 性能优化建议

### 1. 候选数量优化
- **小数据集**: reranker_top_n = 30-50
- **中等数据集**: reranker_top_n = 50-100
- **大数据集**: reranker_top_n = 100-200

### 2. 权重调优
- **高精度需求**: reranker_weight = 2.0-3.0
- **平衡性能**: reranker_weight = 1.5-2.0
- **快速检索**: reranker_weight = 1.0-1.5

### 3. 后端选择
- **生产环境**: 使用FlagEmbedding或ONNX
- **开发测试**: 使用Transformers
- **自动选择**: 让系统自动选择最佳后端

## 故障排除

### 1. 重排序器加载失败
- 检查模型名称是否正确
- 确认网络连接正常
- 验证依赖包是否安装完整

### 2. 内存不足
- 减少reranker_top_n值
- 使用较小的重排序模型
- 启用FP16加速

### 3. 推理速度慢
- 使用ONNX后端
- 启用FP16加速
- 减少候选文档数量

## 模型选择建议

### 中文场景
- **BAAI/bge-reranker-large**: 最佳性能，适合生产环境
- **BAAI/bge-reranker-base**: 平衡性能和速度

### 英文场景
- **BAAI/bge-reranker-large**: 最佳性能
- **BAAI/bge-reranker-base**: 平衡选择

### 多语言场景
- **BAAI/bge-reranker-large**: 支持中英文混合

## 更新日志

- **v3.0**: 初始版本，支持FlagEmbedding后端
- **v3.1**: 新增Transformers和ONNX后端支持
- **v3.2**: 优化自动后端选择逻辑
- **v3.3**: 完善配置管理和性能监控 