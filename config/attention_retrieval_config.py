"""
注意力检索配置文件
"""

class AttentionRetrievalConfig:
    """注意力检索配置类"""
    
    # 数据配置
    DATA_CONFIG = {
        'max_documents': 1000000,  # 最大文档数量
        'max_query_length': 512,   # 查询最大长度
        'max_doc_length': 1024,    # 文档最大长度
        'chunk_size': 512,         # 文档分块大小
        'overlap': 50,             # 分块重叠大小
    }
    
    # 向量检索配置
    VECTOR_CONFIG = {
        'embedding_model': 'BAAI/bge-small-zh-v1.5',  # 向量编码模型
        'top_k_candidates': 1000,  # 向量检索返回的候选数量
        'similarity_threshold': 0.3,  # 相似度阈值
        'index_type': 'faiss',     # 索引类型
    }
    
    # 注意力机制配置
    ATTENTION_CONFIG = {
        'model_name': 'bert-base-uncased',  # 注意力模型
        'hidden_size': 768,        # 隐藏层大小
        'num_attention_heads': 12, # 注意力头数
        'num_layers': 6,           # Transformer层数
        'dropout': 0.1,            # Dropout率
        'attention_dropout': 0.1,  # 注意力Dropout率
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'batch_size': 16,          # 批次大小
        'learning_rate': 2e-5,     # 学习率
        'num_epochs': 10,          # 训练轮数
        'warmup_steps': 1000,      # 预热步数
        'max_grad_norm': 1.0,      # 梯度裁剪
        'weight_decay': 0.01,      # 权重衰减
    }
    
    # 融合配置
    FUSION_CONFIG = {
        'vector_weight': 0.3,      # 向量得分权重
        'attention_weight': 0.7,   # 注意力得分权重
        'fusion_method': 'weighted_sum',  # 融合方法
    }
    
    # 评估配置
    EVALUATION_CONFIG = {
        'metrics': ['precision@k', 'recall@k', 'ndcg@k', 'mrr'],  # 评估指标
        'k_values': [1, 5, 10, 20],  # K值
        'test_size': 0.2,          # 测试集比例
        'random_state': 42,        # 随机种子
    }
    
    # 系统配置
    SYSTEM_CONFIG = {
        'device': 'cuda',          # 设备类型
        'num_workers': 4,          # 工作进程数
        'cache_dir': './cache',    # 缓存目录
        'log_level': 'INFO',       # 日志级别
        'save_checkpoints': True,  # 是否保存检查点
    } 