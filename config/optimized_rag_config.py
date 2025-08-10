"""
优化的RAG系统配置
针对小说文本和中文查询进行了优化
"""

# 文档切块配置（针对小说文本优化）
DOCUMENT_CHUNKING_CONFIG = {
    # 小说文本建议使用较小的块大小，保持上下文完整性
    "novel": {
        "parent_chunk_size": 800,      # 父块大小（字符）
        "parent_overlap": 150,         # 父块重叠（字符）
        "sub_chunk_size": 150,         # 子块大小（字符）
        "sub_overlap": 30,             # 子块重叠（字符）
    },
    # 技术文档可以使用较大的块大小
    "technical": {
        "parent_chunk_size": 1200,     # 父块大小（字符）
        "parent_overlap": 200,         # 父块重叠（字符）
        "sub_chunk_size": 250,         # 子块大小（字符）
        "sub_overlap": 50,             # 子块重叠（字符）
    },
    # 默认配置
    "default": {
        "parent_chunk_size": 1000,     # 父块大小（字符）
        "parent_overlap": 200,         # 父块重叠（字符）
        "sub_chunk_size": 200,         # 子块大小（字符）
        "sub_overlap": 50,             # 子块重叠（字符）
    }
}

# 向量检索配置
VECTOR_RETRIEVAL_CONFIG = {
    # 中文模型优先级
    "chinese_models": [
        "BAAI/bge-large-zh-v1.5",      # 中文BGE大模型（需要token）
        "BAAI/bge-base-zh-v1.5",       # 中文BGE基础模型（需要token）
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言模型
    ],
    # 英文模型作为备选
    "english_models": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
    # 检索参数
    "search_params": {
        "top_k_sub": 100,              # 子文档检索数量
        "top_k_parents": 6,            # 父文档检索数量
        "similarity_threshold": 0.3,   # 相似度阈值
    }
}

# 注意力检索配置
ATTENTION_RETRIEVAL_CONFIG = {
    # 查询类型权重配置
    "query_type_weights": {
        "factual": {           # 事实性查询
            "vector_weight": 0.4,
            "attention_weight": 0.6,
        },
        "analytical": {        # 分析性查询
            "vector_weight": 0.3,
            "attention_weight": 0.7,
        },
        "comparative": {       # 比较性查询
            "vector_weight": 0.2,
            "attention_weight": 0.8,
        },
        "creative": {          # 创造性查询
            "vector_weight": 0.3,
            "attention_weight": 0.7,
        }
    },
    # 注意力模式配置
    "attention_patterns": {
        "focused": {           # 集中注意力
            "boost": 0.1,      # 权重提升
        },
        "distributed": {       # 分散注意力
            "boost": 0.05,     # 权重提升
        },
        "hierarchical": {      # 层次注意力
            "boost": 0.0,      # 无提升
        },
        "sequential": {        # 顺序注意力
            "boost": -0.05,    # 权重降低
        },
        "random": {            # 随机注意力
            "boost": -0.1,     # 权重降低
        }
    },
    # 语义匹配配置
    "semantic_matching": {
        "max_doc_tokens": 32,      # 文档最大token数
        "max_query_tokens": 8,     # 查询最大token数
        "temperature": 0.2,        # 注意力温度
        "threshold": 0.55,         # 语义匹配阈值
    }
}

# LLM配置
LLM_CONFIG = {
    # 默认模型配置
    "default_model": "gpt-3.5-turbo",
    "default_temperature": 0.2,
    "max_tokens": 1000,
    "timeout": 120,
    
    # RAG提示词模板
    "rag_prompt_template": """你是一个严谨的中文助手。请严格基于给定的参考上下文回答用户问题：

要求：
1. 如果上下文无法支持答案，必须直接回答："无法根据参考上下文回答。"
2. 禁止编造或引入上下文以外的信息。
3. 回答要准确、精炼、有逻辑。
4. 如果上下文中有多个相关信息，请综合回答。
5. 对于小说类文本，注意人物、地点、事件的准确性。

参考上下文：
{context}

用户问题：{question}

请仅依据参考上下文作答。若无法回答，请直接回复：无法根据参考上下文回答。""",

    # 对比提示词模板
    "comparison_prompt_template": """请对比分析以下两种RAG系统的回答：

注意力RAG回答：{attention_answer}
向量RAG回答：{vector_answer}

请从以下角度分析：
1. 答案的准确性
2. 信息的完整性
3. 逻辑的合理性
4. 与问题的相关性

分析结果："""
}

# 系统性能配置
PERFORMANCE_CONFIG = {
    # 缓存配置
    "cache": {
        "enable_document_cache": True,     # 启用文档缓存
        "enable_embedding_cache": True,    # 启用嵌入缓存
        "cache_ttl": 3600,                # 缓存过期时间（秒）
    },
    # 批处理配置
    "batch": {
        "embedding_batch_size": 32,        # 嵌入批处理大小
        "attention_batch_size": 16,        # 注意力批处理大小
    },
    # 并发配置
    "concurrency": {
        "max_workers": 4,                  # 最大工作线程数
        "timeout": 30,                     # 超时时间（秒）
    }
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "rag_system.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# 错误处理配置
ERROR_HANDLING_CONFIG = {
    # 重试配置
    "retry": {
        "max_retries": 3,
        "retry_delay": 1,
        "backoff_factor": 2,
    },
    # 降级配置
    "fallback": {
        "enable_vector_fallback": True,    # 启用向量检索降级
        "enable_attention_fallback": True, # 启用注意力检索降级
        "fallback_score": 0.1,            # 降级分数
    }
}

# 中文特定配置
CHINESE_CONFIG = {
    # 分词配置
    "tokenization": {
        "use_jieba": True,                # 使用jieba分词
        "use_char_ngram": True,           # 使用字符n-gram
        "ngram_sizes": [2, 3],            # n-gram大小
    },
    # 停用词配置
    "stopwords": [
        '的', '了', '和', '与', '及', '是', '在', '为', '对', '于', '有', '中', '上', '下', '等', 
        '也', '都', '并', '或', '把', '被', '就', '而', '其', '及其', '一个', '一种', '我们', 
        '你', '我', '他', '她', '它', '吗', '呢', '啊', '吧', '着', '地', '得', '之'
    ],
    # 语义关系配置
    "semantic_relations": {
        "人物": ["易书元", "黄宏川", "楚航", "老城隍"],
        "地点": ["阔南山", "元江县", "岭东大地", "山神庙"],
        "事件": ["说书", "显灵", "治灾", "对饮"],
        "身份": ["说书人", "山神", "文吏", "仙道真人"],
    }
}

# 测试配置
TEST_CONFIG = {
    # 测试查询
    "test_queries": [
        "易书元是谁",
        "阔南山的山神可能是谁",
        "易书元在阔南山遇见了谁",
        "易书元第一次遇见超凡之事是什么",
        "主角是谁",
        "易书元出场是什么场景",
        "易书元第一次出现时坐的什么交通工具",
        "易书元第一次出现时坐的什么船",
        "谁是易书元"
    ],
    # 预期答案关键词
    "expected_keywords": {
        "易书元是谁": ["说书人", "文吏", "仙道真人"],
        "阔南山的山神可能是谁": ["黄宏川", "山神"],
        "易书元在阔南山遇见了谁": ["樵夫", "黄宏川", "山神"],
        "易书元第一次遇见超凡之事是什么": ["遇见黄宏川", "山神显灵"],
        "主角是谁": ["易书元"],
        "易书元出场是什么场景": ["客船", "船舱"],
        "易书元第一次出现时坐的什么交通工具": ["客船"],
        "易书元第一次出现时坐的什么船": ["客船"],
        "谁是易书元": ["说书人", "文吏", "仙道真人"]
    }
}

def get_config(config_type: str = "default"):
    """获取配置"""
    configs = {
        "chunking": DOCUMENT_CHUNKING_CONFIG,
        "vector": VECTOR_RETRIEVAL_CONFIG,
        "attention": ATTENTION_RETRIEVAL_CONFIG,
        "llm": LLM_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "error": ERROR_HANDLING_CONFIG,
        "chinese": CHINESE_CONFIG,
        "test": TEST_CONFIG,
    }
    return configs.get(config_type, {})

def get_optimized_chunking_config(text_type: str = "novel"):
    """获取优化的文档切块配置"""
    return DOCUMENT_CHUNKING_CONFIG.get(text_type, DOCUMENT_CHUNKING_CONFIG["default"])

def get_chinese_models():
    """获取中文模型列表"""
    return VECTOR_RETRIEVAL_CONFIG["chinese_models"]

def get_rag_prompt(context: str, question: str) -> str:
    """生成RAG提示词"""
    return LLM_CONFIG["rag_prompt_template"].format(
        context=context,
        question=question
    ) 