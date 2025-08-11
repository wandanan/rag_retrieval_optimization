# 重排序动态配置功能指南

## 概述

V3 RAG引擎现在支持动态重排序配置，允许用户在不重启系统的情况下：
- 启用/禁用重排序功能
- 动态切换重排序模型
- 调整重排序参数
- 实时监控重排序器状态

## 功能特性

### 1. 重排序功能开关
- **主开关**: 可以随时启用或禁用整个重排序功能
- **智能禁用**: 当模型加载失败时，系统会自动禁用重排序功能
- **状态同步**: 前端界面实时反映重排序功能的启用状态

### 2. 动态模型切换
- **在线模型**: 支持HuggingFace上的各种重排序模型
- **自动下载**: 新模型会自动下载并缓存到本地
- **热切换**: 无需重启系统即可切换模型
- **多后端支持**: 支持FlagEmbedding、Transformers、ONNX等后端

### 3. 参数动态调整
- **候选数量**: 调整参与重排序的文档数量（10-200）
- **权重系数**: 调整重排序分数在最终融合中的权重（0-5）
- **后端选择**: 选择重排序计算的后端类型

### 4. 实时状态监控
- **模型状态**: 显示当前重排序模型的加载状态
- **设备信息**: 显示模型运行的设备（CPU/GPU）
- **优化状态**: 显示FP16等优化功能的启用状态
- **配置信息**: 实时显示当前的重排序配置

### 5. 实时配置同步 ⭐ 新功能
- **保存即同步**: 点击保存配置后立即同步到后台
- **加载即同步**: 加载配置后自动同步到后台
- **清除即同步**: 清除配置后自动同步默认值到后台
- **预设即同步**: 应用预设配置后自动同步到后台
- **无需重启**: 所有配置更改立即生效，无需重启系统

## 使用方法

### 1. 基本配置

#### 启用重排序
```javascript
// 勾选"启用交叉编码器重排序"复选框
document.getElementById('use_reranker').checked = true;
```

#### 设置模型名称
```javascript
// 输入HuggingFace模型名称
document.getElementById('reranker_model_name').value = 'BAAI/bge-reranker-large';
```

#### 调整参数
```javascript
// 设置候选数量
document.getElementById('reranker_top_n').value = 50;

// 设置权重
document.getElementById('reranker_weight').value = 1.5;

// 选择后端
document.getElementById('reranker_backend').value = 'auto';
```

### 2. 动态更新配置

#### 通过前端界面
1. 在重排序配置区域调整参数
2. 点击"🔄 更新重排序配置"按钮
3. 系统会自动应用新配置并重新加载模型

#### 通过保存配置 ⭐ 新功能
1. 调整任何配置参数
2. 点击"💾 保存配置"按钮
3. 系统自动保存到本地并同步到后台
4. 配置立即生效，无需重启

#### 通过预设配置 ⭐ 新功能
1. 选择预设配置（平衡模式、精确模式等）
2. 系统自动应用配置并保存
3. 配置立即同步到后台并生效

#### 通过API接口
```bash
curl -X POST http://localhost:8000/api/v3/update_reranker \
  -H "Content-Type: application/json" \
  -d '{
    "use_reranker": true,
    "reranker_model_name": "BAAI/bge-reranker-base",
    "reranker_top_n": 100,
    "reranker_weight": 2.0,
    "reranker_backend": "transformers"
  }'
```

### 3. 状态监控

#### 获取重排序状态
```bash
curl http://localhost:8000/api/v3/reranker_status
```

#### 响应示例
```json
{
  "success": true,
  "reranker_status": {
    "enabled": true,
    "model_name": "BAAI/bge-reranker-large",
    "top_n": 50,
    "weight": 1.5,
    "backend": "auto",
    "available": true,
    "model_info": {
      "model_name": "BAAI/bge-reranker-large",
      "backend": "transformers",
      "is_available": true,
      "device": "cuda:0",
      "use_fp16": true
    }
  }
}
```

## 支持的模型

### 推荐模型
- **BAAI/bge-reranker-large**: 大型重排序模型，精度最高
- **BAAI/bge-reranker-base**: 中型重排序模型，平衡性能和精度
- **BAAI/bge-reranker-v2-m3**: 多语言重排序模型

### 其他兼容模型
- **moka-ai/m3e-reranker**: 多语言重排序
- **intfloat/multilingual-e5-reranker**: 多语言E5重排序
- **ms-marco-MiniLM-L-12-v2**: MS MARCO重排序

## 性能优化建议

### 1. 候选数量选择
- **小规模文档**: 选择20-50个候选
- **中等规模文档**: 选择50-100个候选
- **大规模文档**: 选择100-200个候选

### 2. 权重调整
- **高精度需求**: 权重设置为2.0-3.0
- **平衡模式**: 权重设置为1.0-2.0
- **快速模式**: 权重设置为0.5-1.0

### 3. 后端选择
- **自动模式**: 系统自动选择最佳后端
- **FlagEmbedding**: 适合中文场景，性能较好
- **Transformers**: 兼容性最好，支持所有模型
- **ONNX**: 推理速度最快，但需要模型支持

## 故障排除

### 1. 模型加载失败
**症状**: 重排序器状态显示"不可用"
**解决方案**:
- 检查网络连接
- 验证模型名称是否正确
- 尝试切换后端类型
- 检查GPU内存是否充足

### 2. 性能下降
**症状**: 重排序速度变慢
**解决方案**:
- 减少候选数量
- 启用FP16优化
- 使用ONNX后端
- 检查设备资源使用情况

### 3. 内存不足
**症状**: 系统报错内存不足
**解决方案**:
- 减少候选数量
- 使用较小的模型
- 禁用FP16优化
- 清理模型缓存

## 配置示例

### 高精度配置
```json
{
  "use_reranker": true,
  "reranker_model_name": "BAAI/bge-reranker-large",
  "reranker_top_n": 100,
  "reranker_weight": 2.5,
  "reranker_backend": "auto"
}
```

### 快速配置
```json
{
  "use_reranker": true,
  "reranker_model_name": "BAAI/bge-reranker-base",
  "reranker_top_n": 30,
  "reranker_weight": 1.0,
  "reranker_backend": "onnx"
}
```

### 平衡配置
```json
{
  "use_reranker": true,
  "reranker_model_name": "BAAI/bge-reranker-base",
  "reranker_top_n": 50,
  "reranker_weight": 1.5,
  "reranker_backend": "auto"
}
```

## 技术架构

### 1. 前端组件
- **配置界面**: 重排序参数设置
- **状态显示**: 实时状态监控
- **操作按钮**: 配置更新和测试
- **自动同步**: 配置变化时自动同步到后台

### 2. 后端API
- **配置更新**: `/api/v3/update_reranker`
- **状态查询**: `/api/v3/reranker_status`

### 3. 核心引擎
- **动态配置**: 运行时配置更新
- **模型管理**: 模型加载和切换
- **状态监控**: 实时状态跟踪

### 4. 配置同步机制 ⭐ 新功能
- **保存同步**: `saveConfig()` → `updateBackendRerankerConfig()`
- **加载同步**: `loadConfig()` → 延迟同步到后台
- **清除同步**: `clearConfig()` → 同步默认配置到后台
- **预设同步**: `applyPreset()` → 同步预设配置到后台
- **上传同步**: `uploadFile()` → 保存配置到本地存储

## 更新日志

### v3.1.0 (当前版本)
- ✅ 新增重排序功能开关
- ✅ 支持动态模型切换
- ✅ 实时状态监控
- ✅ 多后端支持
- ✅ 参数动态调整
- ✅ 实时配置同步 ⭐ 新功能
  - 保存配置后立即同步到后台
  - 加载配置后自动同步到后台
  - 清除配置后自动同步默认值到后台
  - 应用预设配置后自动同步到后台
  - 所有配置更改立即生效，无需重启系统

### 计划功能
- 🔄 模型性能基准测试
- 🔄 自动模型选择
- 🔄 批量配置管理
- 🔄 配置模板系统

## 联系支持

如果您在使用过程中遇到问题，请：
1. 查看系统日志
2. 检查配置参数
3. 参考故障排除指南
4. 提交问题报告

---

*最后更新: 2024年12月* 