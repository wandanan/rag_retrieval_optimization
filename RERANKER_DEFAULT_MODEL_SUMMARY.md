# 重排序模型默认值配置总结

## 修改概述

本次修改为RAG系统的重排序功能添加了默认的在线模型配置，用户无需手动输入重排序模型名称，系统会自动使用 `BAAI/bge-reranker-large` 作为默认值。

## 修改的文件

### 1. `web/index.html`
- 为重排序模型输入框添加了默认值：`value="BAAI/bge-reranker-large"`
- 添加了提示信息：`<small class="form-hint">默认使用在线模型 BAAI/bge-reranker-large</small>`

### 2. `web/app.js`
- 在页面加载时自动设置重排序模型的默认值
- 为所有启用重排序的预设配置添加了默认的重排序模型名：
  - `balanced`: `"BAAI/bge-reranker-large"`
  - `precision`: `"BAAI/bge-reranker-large"`
  - `conversational`: `"BAAI/bge-reranker-large"`
  - `hf_optimized`: `"BAAI/bge-reranker-large"`
  - `speed`: 不启用重排序，所以为空字符串

### 3. `config/v3_engine_config.py`
- 为所有启用重排序的预设配置添加了默认的重排序模型名
- 在 `DEFAULT_V3_CONFIG` 中添加了重排序相关配置：
  ```python
  "use_reranker": True,
  "reranker_model_name": "BAAI/bge-reranker-large",
  "reranker_top_n": 50,
  "reranker_weight": 1.5,
  "reranker_backend": "auto"
  ```

## 功能特点

1. **自动填充**: 页面加载时自动填入默认的重排序模型名
2. **预设支持**: 所有预设配置都包含默认的重排序模型
3. **用户友好**: 用户可以直接使用默认值，也可以根据需要修改
4. **在线模型**: 默认使用HuggingFace上的在线模型，无需本地下载

## 使用说明

1. **首次使用**: 打开页面后，重排序模型输入框会自动填入 `BAAI/bge-reranker-large`
2. **应用预设**: 选择任何预设配置时，重排序模型名会自动设置为默认值
3. **自定义配置**: 用户可以随时修改重排序模型名以满足特定需求
4. **验证机制**: 系统会验证重排序模型名不能为空（当启用重排序时）

## 技术实现

- **前端**: JavaScript在页面加载时自动设置默认值
- **后端**: Python配置文件定义预设和默认配置
- **验证**: 前后端都有相应的验证逻辑确保配置完整性

## 注意事项

1. 默认模型 `BAAI/bge-reranker-large` 需要网络连接才能访问
2. 如果用户需要离线使用，可以手动输入本地模型路径
3. 所有修改都保持了向后兼容性，不影响现有功能 