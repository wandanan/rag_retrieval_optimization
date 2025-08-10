#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的V3引擎使用示例
展示如何避免重复构建索引和优化token编码策略
"""

import time
from advanced_zipper_engine_v3 import AdvancedZipperQueryEngineV3, ZipperV3Config, ZipperV3State
import torch

def create_sample_documents():
    """创建示例文档"""
    documents = {
        1: "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        2: "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        3: "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。",
        4: "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
        5: "计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取高级理解。",
        6: "强化学习是一种机器学习方法，其中代理通过与环境交互来学习最优行为策略。",
        7: "神经网络是受生物神经网络启发的计算模型，用于模式识别和机器学习任务。",
        8: "大数据分析涉及处理和分析大型、复杂的数据集，以发现隐藏的模式和见解。",
        9: "云计算提供了通过互联网按需访问计算资源的能力，无需直接管理基础设施。",
        10: "区块链是一种分布式账本技术，提供安全、透明和防篡改的交易记录。"
    }
    return documents

def demonstrate_index_optimization():
    """演示索引优化效果"""
    print("=" * 60)
    print("V3引擎索引优化演示")
    print("=" * 60)
    
    # 配置优化选项
    config = ZipperV3Config(
        hf_model_name="BAAI/bge-small-zh-v1.5",
        precompute_doc_tokens=True,  # 预计算所有文档的token向量
        enable_index_cache=True,      # 启用索引缓存
        cache_dir="index_cache",
        bm25_top_n=50,
        final_top_k=5,
        use_reranker=False  # 暂时关闭重排序以简化演示
    )
    
    # 创建引擎实例
    engine = AdvancedZipperQueryEngineV3(config)
    
    # 创建示例文档
    documents = create_sample_documents()
    
    print(f"\n1. 首次构建索引（包含 {len(documents)} 个文档）...")
    start_time = time.time()
    engine.build_document_index(documents)
    first_build_time = time.time() - start_time
    print(f"   首次构建耗时: {first_build_time:.3f}秒")
    
    # 检查索引状态
    stats = engine.get_index_stats()
    print(f"   索引状态: {stats}")
    
    print(f"\n2. 重复调用build_document_index（应该跳过）...")
    start_time = time.time()
    engine.build_document_index(documents)  # 应该跳过
    repeat_time = time.time() - start_time
    print(f"   重复调用耗时: {repeat_time:.3f}秒")
    
    print(f"\n3. 执行多次查询（应该很快）...")
    queries = [
        "什么是机器学习？",
        "深度学习的特点是什么？",
        "人工智能有哪些应用？",
        "神经网络的工作原理",
        "大数据分析的重要性"
    ]
    
    total_query_time = 0
    for i, query in enumerate(queries, 1):
        print(f"   查询 {i}: '{query}'")
        start_time = time.time()
        results = engine.retrieve(query)
        query_time = time.time() - start_time
        total_query_time += query_time
        
        print(f"     结果数量: {len(results)}")
        print(f"     查询耗时: {query_time:.3f}秒")
        if results:
            top_result = results[0]
            print(f"     最佳匹配: {top_result[2][:50]}...")
        print()
    
    print(f"   总查询耗时: {total_query_time:.3f}秒")
    print(f"   平均查询耗时: {total_query_time/len(queries):.3f}秒")
    
    print(f"\n4. 强制重新构建索引...")
    start_time = time.time()
    engine.build_document_index(documents, force_rebuild=True)
    force_rebuild_time = time.time() - start_time
    print(f"   强制重建耗时: {force_rebuild_time:.3f}秒")
    
    print(f"\n5. 性能对比总结:")
    print(f"   首次构建: {first_build_time:.3f}秒")
    print(f"   重复调用: {repeat_time:.3f}秒 (节省: {first_build_time-repeat_time:.3f}秒)")
    print(f"   强制重建: {force_rebuild_time:.3f}秒")
    print(f"   平均查询: {total_query_time/len(queries):.3f}秒")
    
    return engine

def demonstrate_cache_benefits():
    """演示缓存带来的好处"""
    print("\n" + "=" * 60)
    print("索引缓存演示")
    print("=" * 60)
    
    # 创建新的配置和引擎实例
    config = ZipperV3Config(
        hf_model_name="BAAI/bge-small-zh-v1.5",
        precompute_doc_tokens=True,
        enable_index_cache=True,
        cache_dir="index_cache"
    )
    
    engine = AdvancedZipperQueryEngineV3(config)
    documents = create_sample_documents()
    
    print("1. 从缓存加载索引...")
    start_time = time.time()
    engine.build_document_index(documents)
    cache_load_time = time.time() - start_time
    print(f"   从缓存加载耗时: {cache_load_time:.3f}秒")
    
    print("2. 检查缓存文件...")
    import os
    cache_files = [f for f in os.listdir("index_cache") if f.endswith('.pkl')]
    print(f"   缓存文件数量: {len(cache_files)}")
    for file in cache_files:
        file_path = os.path.join("index_cache", file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   缓存文件: {file} ({file_size:.1f} KB)")
    
    print("3. 清理缓存...")
    if engine.cache_manager:
        engine.cache_manager.clear_cache()
        print("   缓存已清理")

def demonstrate_adaptive_strategies():
    """演示自适应策略"""
    print("\n" + "=" * 60)
    print("自适应策略演示")
    print("=" * 60)
    
    # 创建不同配置的引擎
    configs = [
        ZipperV3Config(
            name="预计算模式",
            precompute_doc_tokens=True,
            enable_index_cache=True
        ),
        ZipperV3Config(
            name="按需编码模式", 
            precompute_doc_tokens=False,
            enable_index_cache=False
        )
    ]
    
    documents = create_sample_documents()
    
    for config in configs:
        print(f"\n{config.name}:")
        engine = AdvancedZipperQueryEngineV3(config)
        
        # 构建索引
        start_time = time.time()
        engine.build_document_index(documents)
        build_time = time.time() - start_time
        print(f"  索引构建耗时: {build_time:.3f}秒")
        
        # 执行查询
        start_time = time.time()
        results = engine.retrieve("什么是人工智能？")
        query_time = time.time() - start_time
        print(f"  查询耗时: {query_time:.3f}秒")
        print(f"  结果数量: {len(results)}")
        
        # 清理
        engine.clear_index()

if __name__ == "__main__":
    try:
        # 演示索引优化
        engine = demonstrate_index_optimization()
        
        # 演示缓存好处
        demonstrate_cache_benefits()
        
        # 演示自适应策略
        demonstrate_adaptive_strategies()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 