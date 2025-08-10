# 编码后端同步问题修复总结

## 问题描述

用户发现前端选择了"HuggingFace (在线模型)"，但后台仍然在加载本地的BGE模型：
```
2025-08-10 17:47:17,771 - advanced_zipper_engine_v3 - INFO - 正在加载BGE模型用于Token级编码: models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620
```

## 问题分析

经过代码分析，发现以下问题：

1. **前端配置传递问题**: 前端在发送请求时，无论选择什么编码后端，都会发送 `bge_model_path` 的值
2. **预设配置缺失**: JavaScript中的预设配置没有包含 `encoder_backend` 字段
3. **UI状态不同步**: 应用预设后，UI状态没有正确更新

## 修复方案

### 1. 修复前端配置传递逻辑 (`web/app.js`)

在 `askQuestion()` 函数中添加了智能的模型配置处理：

```javascript
// 根据编码后端选择性地设置模型配置
const modelConfig = {};
if (encoder_backend === 'bge') {
  modelConfig.bge_model_path = bge_model_path;
  modelConfig.hf_model_name = '';
} else if (encoder_backend === 'hf') {
  modelConfig.bge_model_path = '';
  modelConfig.hf_model_name = hf_model_name;
}

const payload = {
  // ... 其他配置
  v3_config: {
    encoder_backend,
    ...modelConfig,  // 使用展开运算符应用模型配置
    // ... 其他配置
  }
};
```

### 2. 完善预设配置 (`web/app.js`)

为所有预设添加了 `encoder_backend` 字段：

```javascript
const presets = {
  balanced: {
    encoder_backend: 'bge',  // 明确指定编码后端
    // ... 其他配置
  },
  precision: {
    encoder_backend: 'bge',
    // ... 其他配置
  },
  speed: {
    encoder_backend: 'bge',
    // ... 其他配置
  },
  conversational: {
    encoder_backend: 'bge',
    // ... 其他配置
  },
  hf_optimized: {
    encoder_backend: 'hf',  // HF优化模式使用HF后端
    // ... 其他配置
  }
};
```

### 3. 改进预设应用逻辑 (`web/app.js`)

在应用预设后自动更新UI状态：

```javascript
// 应用预设配置
Object.keys(preset).forEach(key => {
  const element = document.getElementById(key);
  if (element) {
    if (element.type === 'checkbox') {
      element.checked = preset[key];
    } else {
      element.value = preset[key];
    }
  }
});

// 更新UI状态
toggleModelInputs();
```

## 修复效果

1. **配置同步**: 前端选择的编码后端现在能正确传递给后端
2. **模型选择**: 当选择HF后端时，`bge_model_path` 会被设置为空字符串
3. **预设支持**: 所有预设都包含完整的编码后端配置
4. **UI一致性**: 应用预设后UI状态会自动更新

## 技术细节

- **智能配置**: 根据编码后端动态构建模型配置对象
- **展开运算符**: 使用 `...modelConfig` 优雅地应用模型配置
- **状态管理**: 确保前端UI状态与配置数据保持一致
- **向后兼容**: 所有修改都保持了向后兼容性

## 测试建议

1. 选择"HF优化模式"预设，确认编码后端自动切换到"hf"
2. 手动选择"HuggingFace (在线模型)"，确认BGE模型路径被清空
3. 发送查询请求，确认后端日志显示正确的模型加载信息
4. 测试其他预设，确认编码后端设置正确

## 注意事项

1. 确保后端正确处理空的 `bge_model_path` 和 `hf_model_name`
2. 当使用HF后端时，需要网络连接来下载模型
3. 本地BGE模型路径仍然可以手动输入，用于离线场景 