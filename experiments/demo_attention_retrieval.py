"""
设想1演示脚本 - 快速验证注意力检索效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import logging
from typing import List, Dict
import json
import time

from models.hybrid_retrieval_system import HybridRetrievalSystem
from utils.data_utils import DataProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoAttentionRetrieval:
    """注意力检索演示类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Initializing demo on device: {device}")
        
        # 初始化混合检索系统
        self.retrieval_system = HybridRetrievalSystem(device=device)
        
        # 示例文档
        self.sample_documents = self._create_sample_documents()
        
        # 示例查询
        self.sample_queries = self._create_sample_queries()
    
    def _create_sample_documents(self) -> List[str]:
        """创建示例文档"""
        documents = [
            "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。",
            
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。机器学习算法通过分析数据来识别模式并做出预测。",
            
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            
            "自然语言处理（NLP）是人工智能的一个领域，专注于使计算机能够理解、解释和生成人类语言。NLP技术被广泛应用于聊天机器人、翻译系统和文本分析工具中。",
            
            "计算机视觉是人工智能的一个分支，致力于使计算机能够从图像和视频中获取高级理解。它被用于面部识别、自动驾驶汽车和医疗图像分析等应用。",
            
            "强化学习是一种机器学习方法，其中智能体通过与环境交互来学习最优行为策略。它被用于游戏AI、机器人控制和推荐系统等领域。",
            
            "神经网络是受生物神经网络启发的计算模型。它们由相互连接的节点（神经元）组成，能够学习复杂的非线性关系。",
            
            "大数据是指无法使用传统数据处理软件有效处理的庞大、复杂的数据集。大数据分析在商业智能、科学研究和政府决策中发挥着重要作用。",
            
            "云计算是一种通过互联网提供计算服务的模型。它允许用户按需访问共享的计算资源，而无需拥有和管理物理基础设施。",
            
            "区块链是一种分布式账本技术，它允许多方在没有中央权威的情况下进行安全、透明的交易。它最著名的应用是加密货币比特币。"
        ]
        
        return documents
    
    def _create_sample_queries(self) -> List[str]:
        """创建示例查询"""
        queries = [
            "什么是人工智能？",
            "机器学习和深度学习有什么区别？",
            "自然语言处理有哪些应用？",
            "神经网络的工作原理是什么？",
            "大数据分析有什么价值？",
            "云计算的优势是什么？",
            "区块链技术如何工作？",
            "计算机视觉在哪些领域有应用？",
            "强化学习的基本原理是什么？",
            "AI在医疗领域的应用有哪些？"
        ]
        
        return queries
    
    def run_basic_demo(self):
        """运行基础演示"""
        logger.info("=== 开始基础演示 ===")
        
        # 1. 构建索引
        logger.info("1. 构建文档索引...")
        self.retrieval_system.build_index(self.sample_documents)
        
        # 2. 测试单个查询
        logger.info("2. 测试单个查询...")
        query = "什么是人工智能？"
        results = self.retrieval_system.search(query, top_k=5)
        
        # 3. 显示结果
        logger.info("3. 检索结果:")
        for i, result in enumerate(results['top_results']):
            logger.info(f"  排名 {i+1}: 得分 {result['final_score']:.4f}")
            logger.info(f"    向量得分: {result['vector_score']:.4f}")
            logger.info(f"    注意力得分: {result['attention_score']:.4f}")
            logger.info(f"    文档: {result['document'][:100]}...")
            logger.info("")
        
        # 4. 性能分析
        logger.info("4. 性能分析:")
        logger.info(f"   向量检索时间: {results['performance']['vector_search_time']:.3f}s")
        logger.info(f"   注意力检索时间: {results['performance']['attention_search_time']:.3f}s")
        logger.info(f"   总检索时间: {results['performance']['total_search_time']:.3f}s")
        logger.info(f"   候选文档数量: {results['performance']['candidate_count']}")
        
        return results
    
    def run_comparison_demo(self):
        """运行对比演示"""
        logger.info("=== 开始对比演示 ===")
        
        query = "机器学习和深度学习有什么区别？"
        
        # 对比传统RAG和混合检索
        comparison = self.retrieval_system.compare_with_traditional_rag(query, top_k=5)
        
        logger.info("传统RAG结果:")
        for i, (doc_id, score) in enumerate(comparison['traditional_rag']['results']):
            logger.info(f"  排名 {i+1}: 得分 {score:.4f}")
        
        logger.info("混合检索结果:")
        for i, result in enumerate(comparison['hybrid_retrieval']['results']):
            logger.info(f"  排名 {i+1}: 最终得分 {result['final_score']:.4f}")
            logger.info(f"    向量得分: {result['vector_score']:.4f}")
            logger.info(f"    注意力得分: {result['attention_score']:.4f}")
        
        logger.info(f"性能对比:")
        logger.info(f"  传统RAG时间: {comparison['traditional_rag']['time']:.3f}s")
        logger.info(f"  混合检索时间: {comparison['hybrid_retrieval']['time']:.3f}s")
        logger.info(f"  时间开销: {comparison['improvement']['time_overhead']:.3f}s")
        logger.info(f"  时间比率: {comparison['improvement']['time_ratio']:.2f}x")
        
        return comparison
    
    def run_attention_analysis_demo(self):
        """运行注意力分析演示"""
        logger.info("=== 开始注意力分析演示 ===")
        
        query = "自然语言处理有哪些应用？"
        
        # 获取注意力洞察
        insights = self.retrieval_system.get_attention_insights(query, top_k=5)
        
        logger.info("注意力模式分析:")
        for pattern in insights['attention_patterns']:
            logger.info(f"  文档 {pattern['doc_id']}: {pattern['pattern_name']} 模式")
        
        logger.info("融合权重分析:")
        for weight in insights['fusion_weights']:
            logger.info(f"  文档 {weight['doc_id']}: 向量权重 {weight['vector_weight']:.3f}, "
                       f"注意力权重 {weight['attention_weight']:.3f}")
        
        logger.info("查询类型分析:")
        for query_type in insights['query_type_analysis']:
            logger.info(f"  文档 {query_type['doc_id']}: {query_type['query_type_name']} 类型")
        
        return insights
    
    def run_batch_demo(self):
        """运行批量演示"""
        logger.info("=== 开始批量演示 ===")
        
        # 批量检索
        results = self.retrieval_system.batch_search(self.sample_queries[:5], top_k=3)
        
        logger.info("批量检索结果摘要:")
        for i, result in enumerate(results):
            logger.info(f"查询 {i+1}: {result['query']}")
            logger.info(f"  最佳结果得分: {result['top_results'][0]['final_score']:.4f}")
            logger.info(f"  检索时间: {result['performance']['total_search_time']:.3f}s")
            logger.info("")
        
        # 性能分析
        performance = self.retrieval_system.analyze_performance()
        logger.info("整体性能分析:")
        logger.info(f"  总检索次数: {performance['total_searches']}")
        logger.info(f"  平均向量检索时间: {performance['average_times']['vector_search']:.3f}s")
        logger.info(f"  平均注意力检索时间: {performance['average_times']['attention_search']:.3f}s")
        logger.info(f"  平均总检索时间: {performance['average_times']['total_search']:.3f}s")
        logger.info(f"  平均候选文档数: {performance['candidate_analysis']['average_candidates']:.1f}")
        
        return results, performance
    
    def run_full_demo(self):
        """运行完整演示"""
        logger.info("=== 开始完整演示 ===")
        
        # 1. 基础演示
        basic_results = self.run_basic_demo()
        
        # 2. 对比演示
        comparison_results = self.run_comparison_demo()
        
        # 3. 注意力分析演示
        attention_results = self.run_attention_analysis_demo()
        
        # 4. 批量演示
        batch_results, performance = self.run_batch_demo()
        
        # 5. 保存结果
        demo_results = {
            'basic_results': basic_results,
            'comparison_results': comparison_results,
            'attention_results': attention_results,
            'batch_results': batch_results,
            'performance': performance
        }
        
        # 保存到文件
        with open('demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("演示完成！结果已保存到 demo_results.json")
        
        return demo_results

def main():
    """主函数"""
    logger.info("启动注意力检索演示...")
    
    # 创建演示实例
    demo = DemoAttentionRetrieval()
    
    # 运行完整演示
    results = demo.run_full_demo()
    
    logger.info("演示完成！")

if __name__ == "__main__":
    main() 