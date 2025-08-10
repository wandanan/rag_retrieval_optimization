# v3_engine_config.py
# V3引擎配置文件

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class V3EnginePreset:
    """V3引擎预设配置"""
    name: str
    description: str
    config: Dict[str, Any]

# 预设配置
V3_ENGINE_PRESETS = {
    "balanced": V3EnginePreset(
        name="平衡模式",
        description="BM25和ColBERT权重平衡，适合一般用途",
        config={
            "encoder_backend": "hf",
            "hf_model_name": "BAAI/bge-small-zh-v1.5",
            "bm25_weight": 1.0,
            "colbert_weight": 1.0,
            "num_heads": 8,
            "context_influence": 0.3,
            "final_top_k": 10,
            "use_hybrid_search": True,
            "use_multi_head": True,
            "use_length_penalty": True,
            "use_stateful_reranking": True,
            # 新增重排序配置
            "use_reranker": True,
            "reranker_model_name": "BAAI/bge-reranker-large",
            "reranker_top_n": 50,
            "reranker_weight": 1.5,
            "reranker_backend": "auto"  # 新增：重排序后端选择
        }
    ),
    
    "precision": V3EnginePreset(
        name="精确模式",
        description="高ColBERT权重，适合需要高精度的场景",
        config={
            "encoder_backend": "hf",
            "hf_model_name": "BAAI/bge-small-zh-v1.5",
            "bm25_weight": 0.5,
            "colbert_weight": 2.9,
            "num_heads": 8,
            "context_influence": 0.4,
            "final_top_k": 15,
            "use_hybrid_search": True,
            "use_multi_head": True,
            "use_length_penalty": True,
            "use_stateful_reranking": True,
            # 新增重排序配置
            "use_reranker": True,
            "reranker_model_name": "BAAI/bge-reranker-large",
            "reranker_top_n": 100,
            "reranker_weight": 2.0,
            "reranker_backend": "auto"  # 新增：重排序后端选择
        }
    ),
    
    "speed": V3EnginePreset(
        name="快速模式",
        description="高BM25权重，快速检索，适合大量文档",
        config={
            "encoder_backend": "hf",
            "hf_model_name": "BAAI/bge-small-zh-v1.5",
            "bm25_weight": 2.0,
            "colbert_weight": 0.5,
            "num_heads": 4,
            "context_influence": 0.2,
            "final_top_k": 8,
            "use_hybrid_search": True,
            "use_multi_head": False,
            "use_length_penalty": False,
            "use_stateful_reranking": False,
            # 新增重排序配置
            "use_reranker": False,
            "reranker_model_name": "",
            "reranker_top_n": 30,
            "reranker_weight": 1.0,
            "reranker_backend": "auto"  # 新增：重排序后端选择
        }
    ),
    
    "conversational": V3EnginePreset(
        name="对话模式",
        description="高上下文影响，适合多轮对话",
        config={
            "encoder_backend": "hf",
            "hf_model_name": "BAAI/bge-small-zh-v1.5",
            "bm25_weight": 1.0,
            "colbert_weight": 1.5,
            "num_heads": 8,
            "context_influence": 0.6,
            "final_top_k": 12,
            "use_hybrid_search": True,
            "use_multi_head": True,
            "use_length_penalty": True,
            "use_stateful_reranking": True,
            # 新增重排序配置
            "use_reranker": True,
            "reranker_model_name": "BAAI/bge-reranker-large",
            "reranker_top_n": 80,
            "reranker_weight": 1.8,
            "reranker_backend": "auto"  # 新增：重排序后端选择
        }
    ),
    
    "hf_optimized": V3EnginePreset(
        name="HF优化模式",
        description="专为HuggingFace模型优化的配置",
        config={
            "encoder_backend": "hf",
            "hf_model_name": "BAAI/bge-small-zh-v1.5",
            "bm25_weight": 0.8,
            "colbert_weight": 1.8,
            "num_heads": 6,
            "context_influence": 0.35,
            "final_top_k": 10,
            "use_hybrid_search": True,
            "use_multi_head": True,
            "use_length_penalty": True,
            "use_stateful_reranking": True,
            # 新增重排序配置
            "use_reranker": True,
            "reranker_model_name": "BAAI/bge-reranker-large",
            "reranker_top_n": 60,
            "reranker_weight": 1.6,
            "reranker_backend": "auto"  # 新增：重排序后端选择
        }
    )
}

# 默认配置
DEFAULT_V3_CONFIG = {
    "encoder_backend": "hf",
    "hf_model_name": "BAAI/bge-small-zh-v1.5",
    "bm25_weight": 1.0,
    "colbert_weight": 1.5,
    "num_heads": 8,
    "context_influence": 0.3,
    "final_top_k": 10,
    "use_hybrid_search": True,
    "use_multi_head": True,
    "use_length_penalty": True,
    "use_stateful_reranking": True,
    "precompute_doc_tokens": False,
    "enable_amp_if_beneficial": True,
    # 重排序默认配置
    "use_reranker": True,
    "reranker_model_name": "BAAI/bge-reranker-large",
    "reranker_top_n": 50,
    "reranker_weight": 1.5,
    "reranker_backend": "auto"  # 新增：重排序后端选择
}

# 配置验证规则
CONFIG_VALIDATION_RULES = {
    "encoder_backend": {
        "type": str,
        "allowed_values": ["hf"],
        "description": "编码后端，支持HF"
    },
    "hf_model_name": {
        "type": str,
        "required_if": {"encoder_backend": "hf"},
        "description": "HuggingFace模型名称，当使用HF后端时必需"
    },
    "reranker_model_name": {
        "type": str,
        "required_if": {"use_reranker": True},
        "description": "重排序模型名称，当启用重排序时必需"
    },
    "bm25_weight": {
        "type": float,
        "min": 0.0,
        "max": 5.0,
        "description": "BM25权重，范围0.0-5.0"
    },
    "colbert_weight": {
        "type": float,
        "min": 0.0,
        "max": 5.0,
        "description": "ColBERT权重，范围0.0-5.0"
    },
    "num_heads": {
        "type": int,
        "min": 1,
        "max": 16,
        "description": "多头数量，范围1-16"
    },
    "context_influence": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "上下文影响因子，范围0.0-1.0"
    },
    "final_top_k": {
        "type": int,
        "min": 1,
        "max": 50,
        "description": "最终返回结果数量，范围1-50"
    }
}

def validate_v3_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """验证V3引擎配置"""
    errors = []
    
    for key, rule in CONFIG_VALIDATION_RULES.items():
        if key not in config:
            if "required_if" in rule:
                required_condition = rule["required_if"]
                for req_key, req_value in required_condition.items():
                    if config.get(req_key) == req_value:
                        errors.append(f"缺少必需配置: {key}")
            continue
        
        value = config[key]
        
        # 类型检查
        if "type" in rule and not isinstance(value, rule["type"]):
            errors.append(f"{key} 类型错误，期望 {rule['type'].__name__}，实际 {type(value).__name__}")
            continue
        
        # 字符串类型特殊处理
        if isinstance(value, str) and "type" in rule and rule["type"] == str:
            # 检查字符串是否为空（对于必需字段）
            if "required_if" in rule and not value.strip():
                errors.append(f"{key} 不能为空")
                continue
        
        # 值范围检查
        if "min" in rule and value < rule["min"]:
            errors.append(f"{key} 值过小，最小值: {rule['min']}")
        if "max" in rule and value > rule["max"]:
            errors.append(f"{key} 值过大，最大值: {rule['max']}")
        
        # 允许值检查
        if "allowed_values" in rule and value not in rule["allowed_values"]:
            errors.append(f"{key} 值无效，允许值: {rule['allowed_values']}")
    
    return len(errors) == 0, errors

def get_preset_config(preset_name: str) -> Optional[Dict[str, Any]]:
    """获取预设配置"""
    if preset_name in V3_ENGINE_PRESETS:
        return V3_ENGINE_PRESETS[preset_name].config.copy()
    return None

def get_all_presets() -> Dict[str, V3EnginePreset]:
    """获取所有预设配置"""
    return V3_ENGINE_PRESETS.copy()

def merge_config_with_preset(base_config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """将基础配置与预设配置合并"""
    preset_config = get_preset_config(preset_name)
    if preset_config:
        merged = base_config.copy()
        merged.update(preset_config)
        return merged
    return base_config 