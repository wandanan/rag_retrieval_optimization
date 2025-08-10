#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重排序功能演示脚本
展示如何在AdvancedZipperQueryEngineV3中使用重排序功能
"""

from advanced_zipper_engine_v3 import AdvancedZipperQueryEngineV3, ZipperV3Config
import time

def demo_reranker():
    """演示重排序功能"""
    
    print("🚀 重排序功能演示")
    print("=" * 50)
    
    # 1. 创建配置
    config = ZipperV3Config(
        # 基础配置
        bge_model_path="models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
        bm25_top_n=100,  # 扩大初步召回
        final_top_k=10,
        
        # 重排序配置
        use_reranker=True,
        reranker_model_name="BAAI/bge-reranker-large",
        reranker_top_n=50,  # 对前50个文档进行重排序
        reranker_weight=1.5,  # 重排序分数权重
        
        # 其他优化配置
        use_hybrid_search=True,
        bm25_weight=1.0,
        colbert_weight=1.0,
        use_multi_head=True,
        num_heads=8
    )
    
    print(f"✅ 配置创建完成")
    print(f"   - 重排序启用: {config.use_reranker}")
    print(f"   - 重排序模型: {config.reranker_model_name}")
    print(f"   - 重排序候选数: {config.reranker_top_n}")
    print(f"   - 重排序权重: {config.reranker_weight}")
    print(f"   - BM25候选数: {config.bm25_top_n}")
    print()
    
    # 2. 创建引擎实例
    try:
        engine = AdvancedZipperQueryEngineV3(config)
        print("✅ 引擎初始化完成")
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        return
    
    # 3. 准备测试文档
    test_documents = {
        1: "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        2: "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        3: "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。",
        4: "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
        5: "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中获取信息。",
        6: "机器人技术结合了人工智能、机械工程和电子学，创建能够执行物理任务的机器。",
        7: "专家系统是早期的人工智能应用，使用规则和知识库来模拟人类专家的决策过程。",
        8: "神经网络是受生物神经元启发的计算模型，是现代人工智能的基础。",
        9: "强化学习是一种机器学习方法，通过与环境交互来学习最优策略。",
        10: "知识图谱是表示实体及其关系的结构化方式，广泛应用于搜索引擎和推荐系统。"
    }
    
    print(f"📚 准备测试文档: {len(test_documents)} 个")
    
    # 4. 构建索引
    try:
        engine.build_document_index(test_documents)
        print("✅ 索引构建完成")
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")
        return
    
    # 5. 测试查询
    test_queries = [
        "什么是机器学习？",
        "神经网络如何工作？",
        "人工智能的主要应用领域有哪些？"
    ]
    
    print(f"\n🔍 开始测试查询...")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            # 执行检索
            results = engine.retrieve(query)
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            print(f"⏱️  检索耗时: {retrieval_time:.3f}秒")
            print(f"📊 返回结果数: {len(results)}")
            
            # 显示前3个结果
            for j, (doc_id, score, content) in enumerate(results[:3], 1):
                print(f"  {j}. 文档ID: {doc_id}, 分数: {score:.4f}")
                print(f"     内容: {content[:80]}...")
                print()
                
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    print("=" * 50)
    print("🎉 重排序功能演示完成！")
    print("\n💡 使用提示:")
    print("   - 重排序会显著提升检索精度，但会增加计算时间")
    print("   - 可以通过调整 reranker_top_n 来平衡精度和速度")
    print("   - 重排序权重越高，重排序分数的影响越大")
    print("   - 建议在生产环境中启用重排序以获得最佳效果")

if __name__ == "__main__":
    demo_reranker() 