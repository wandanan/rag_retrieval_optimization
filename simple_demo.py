"""
简化演示脚本 - 快速验证设想1的效果
"""

import torch
import numpy as np
import logging
from typing import List, Dict
import json
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟向量检索器
class MockVectorRetriever:
    """模拟向量检索器"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
    
    def build_index(self, documents):
        """构建索引"""
        self.documents = documents
        # 模拟文档嵌入
        self.embeddings = np.random.rand(len(documents), 384)
        logger.info(f"Built index for {len(documents)} documents")
    
    def search(self, query, top_k=1000, similarity_threshold=0.3):
        """模拟搜索"""
        # 模拟查询嵌入
        query_embedding = np.random.rand(384)
        
        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 排序并过滤
        indices = np.argsort(similarities)[::-1]
        filtered_indices = [idx for idx in indices if similarities[idx] >= similarity_threshold]
        
        doc_ids = [f"doc_{idx}" for idx in filtered_indices[:top_k]]
        scores = [float(similarities[idx]) for idx in filtered_indices[:top_k]]
        
        return doc_ids, scores

# 模拟注意力检索器
class MockAttentionRetriever:
    """模拟注意力检索器"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def retrieve(self, query, documents, vector_scores):
        """模拟注意力检索"""
        results = []
        
        for i, (doc, vector_score) in enumerate(zip(documents, vector_scores)):
            # 模拟注意力得分（基于文档长度和关键词匹配）
            attention_score = self._compute_attention_score(query, doc)
            
            # 模拟动态融合
            final_score = self._dynamic_fusion(vector_score, attention_score, query)
            
            results.append({
                'doc_id': i,
                'document': doc,
                'vector_score': vector_score,
                'attention_score': attention_score,
                'final_score': final_score
            })
        
        # 排序
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def _compute_attention_score(self, query, document):
        """计算注意力得分"""
        # 简单的关键词匹配
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # 计算重叠度
        overlap = len(query_words.intersection(doc_words))
        total = len(query_words.union(doc_words))
        
        if total == 0:
            return 0.0
        
        # 添加一些随机性模拟注意力机制
        base_score = overlap / total
        attention_score = base_score + np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, attention_score))
    
    def _dynamic_fusion(self, vector_score, attention_score, query):
        """动态融合"""
        # 根据查询类型调整权重
        if '什么' in query or '如何' in query:
            # 事实性查询，更重视向量得分
            vector_weight = 0.7
            attention_weight = 0.3
        elif '区别' in query or '比较' in query:
            # 比较性查询，更重视注意力得分
            vector_weight = 0.3
            attention_weight = 0.7
        else:
            # 默认权重
            vector_weight = 0.5
            attention_weight = 0.5
        
        final_score = vector_weight * vector_score + attention_weight * attention_score
        return final_score

# 混合检索系统
class SimpleHybridRetrievalSystem:
    """简化的混合检索系统"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.vector_retriever = MockVectorRetriever()
        self.attention_retriever = MockAttentionRetriever(device)
        self.documents = []
        self.stats = {
            'vector_search_time': [],
            'attention_search_time': [],
            'total_search_time': []
        }
    
    def build_index(self, documents):
        """构建索引"""
        self.documents = documents
        self.vector_retriever.build_index(documents)
    
    def search(self, query, top_k=10, vector_candidates=1000):
        """混合检索"""
        start_time = time.time()
        
        # 1. 向量检索
        vector_start = time.time()
        candidate_doc_ids, vector_scores = self.vector_retriever.search(
            query, top_k=vector_candidates
        )
        vector_time = time.time() - vector_start
        
        # 获取候选文档
        candidate_documents = [self.documents[int(doc_id.split('_')[1])] 
                             for doc_id in candidate_doc_ids]
        
        # 2. 注意力检索
        attention_start = time.time()
        attention_results = self.attention_retriever.retrieve(
            query, candidate_documents, vector_scores
        )
        attention_time = time.time() - attention_start
        
        total_time = time.time() - start_time
        
        # 更新统计
        self.stats['vector_search_time'].append(vector_time)
        self.stats['attention_search_time'].append(attention_time)
        self.stats['total_search_time'].append(total_time)
        
        return {
            'query': query,
            'top_results': attention_results[:top_k],
            'performance': {
                'vector_search_time': vector_time,
                'attention_search_time': attention_time,
                'total_search_time': total_time,
                'candidate_count': len(candidate_documents)
            }
        }
    
    def compare_with_traditional_rag(self, query, top_k=10):
        """与传统RAG对比"""
        # 传统RAG（只使用向量检索）
        vector_start = time.time()
        vector_doc_ids, vector_scores = self.vector_retriever.search(query, top_k=top_k)
        vector_time = time.time() - vector_start
        
        # 混合检索
        hybrid_start = time.time()
        hybrid_results = self.search(query, top_k=top_k)
        hybrid_time = time.time() - hybrid_start
        
        return {
            'query': query,
            'traditional_rag': {
                'results': list(zip(vector_doc_ids, vector_scores)),
                'time': vector_time
            },
            'hybrid_retrieval': {
                'results': hybrid_results['top_results'],
                'time': hybrid_time
            },
            'improvement': {
                'time_overhead': hybrid_time - vector_time,
                'time_ratio': hybrid_time / vector_time if vector_time > 0 else float('inf')
            }
        }

def main():
    """主函数"""
    logger.info("启动简化注意力检索演示...")
    
    # 创建示例数据
    documents = [
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理（NLP）是人工智能的一个领域，专注于使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个分支，致力于使计算机能够从图像和视频中获取高级理解。",
        "强化学习是一种机器学习方法，其中智能体通过与环境交互来学习最优行为策略。",
        "神经网络是受生物神经网络启发的计算模型。它们由相互连接的节点组成。",
        "大数据是指无法使用传统数据处理软件有效处理的庞大、复杂的数据集。",
        "云计算是一种通过互联网提供计算服务的模型。",
        "区块链是一种分布式账本技术，它允许多方在没有中央权威的情况下进行安全交易。"
    ]
    
    queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理有哪些应用？"
    ]
    
    # 初始化系统
    system = SimpleHybridRetrievalSystem()
    system.build_index(documents)
    
    # 测试查询
    for query in queries:
        logger.info(f"\n=== 测试查询: {query} ===")
        
        # 混合检索
        results = system.search(query, top_k=3)
        
        logger.info("检索结果:")
        for i, result in enumerate(results['top_results']):
            logger.info(f"  排名 {i+1}: 最终得分 {result['final_score']:.4f}")
            logger.info(f"    向量得分: {result['vector_score']:.4f}")
            logger.info(f"    注意力得分: {result['attention_score']:.4f}")
            logger.info(f"    文档: {result['document'][:50]}...")
        
        # 性能分析
        logger.info(f"性能: 向量检索 {results['performance']['vector_search_time']:.3f}s, "
                   f"注意力检索 {results['performance']['attention_search_time']:.3f}s, "
                   f"总计 {results['performance']['total_search_time']:.3f}s")
        
        # 对比传统RAG
        comparison = system.compare_with_traditional_rag(query, top_k=3)
        logger.info(f"与传统RAG对比: 时间开销 {comparison['improvement']['time_overhead']:.3f}s, "
                   f"时间比率 {comparison['improvement']['time_ratio']:.2f}x")
    
    logger.info("\n演示完成！")

if __name__ == "__main__":
    main() 