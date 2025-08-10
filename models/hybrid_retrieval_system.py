"""
混合检索系统 - 整合向量检索和注意力检索
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time
from tqdm import tqdm

from .vector_retriever import VectorRetriever
from .attention_retriever import InnovativeAttentionRetriever
from utils.data_utils import DocumentIndex, QueryProcessor

logger = logging.getLogger(__name__)

class HybridRetrievalSystem:
    """混合检索系统 - 设想1的实现"""
    
    def __init__(self, 
                 vector_model_name: str = 'BAAI/bge-small-zh-v1.5',
                 attention_model_name: str = 'bert-base-uncased',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        logger.info(f"Initializing Hybrid Retrieval System on device: {device}")
        
        # 初始化组件
        self.vector_retriever = VectorRetriever(vector_model_name, device)
        self.attention_retriever = InnovativeAttentionRetriever(attention_model_name, device)
        
        # 文档索引
        self.doc_index = DocumentIndex()
        
        # 查询处理器
        self.query_processor = QueryProcessor(self.attention_retriever.data_processor)
        
        # 性能统计
        self.stats = {
            'vector_search_time': [],
            'attention_search_time': [],
            'total_search_time': [],
            'candidate_counts': []
        }
    
    def build_index(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """构建索引"""
        logger.info(f"Building index for {len(documents)} documents...")
        
        # 添加到文档索引
        self.doc_index.add_documents(documents, doc_ids)
        
        # 构建向量索引
        start_time = time.time()
        self.vector_retriever.build_index(documents, doc_ids)
        vector_time = time.time() - start_time
        
        logger.info(f"Vector index built in {vector_time:.2f} seconds")
        logger.info(f"Total documents indexed: {len(self.doc_index)}")
    
    def search(self, query: str, top_k: int = 10, 
               vector_candidates: int = 1000,
               similarity_threshold: float = 0.3) -> Dict:
        """混合检索"""
        start_time = time.time()
        
        # 1. 向量检索（快速筛选）
        vector_start = time.time()
        candidate_doc_ids, vector_scores = self.vector_retriever.search(
            query, top_k=vector_candidates, similarity_threshold=similarity_threshold
        )
        vector_time = time.time() - vector_start
        
        # 获取候选文档
        candidate_documents = self.doc_index.get_documents(candidate_doc_ids)
        
        logger.info(f"Vector search found {len(candidate_documents)} candidates in {vector_time:.3f}s")
        
        # 2. 注意力检索（精确匹配）
        attention_start = time.time()
        
        # 使用注意力机制重新排序
        attention_results = self.attention_retriever.retrieve(
            query, candidate_documents, vector_scores
        )
        
        attention_time = time.time() - attention_start
        
        # 3. 分析检索过程
        analysis = self.attention_retriever.analyze_retrieval(
            query, candidate_documents, vector_scores
        )
        
        total_time = time.time() - start_time
        
        # 更新统计信息
        self.stats['vector_search_time'].append(vector_time)
        self.stats['attention_search_time'].append(attention_time)
        self.stats['total_search_time'].append(total_time)
        self.stats['candidate_counts'].append(len(candidate_documents))
        
        # 构建结果
        results = {
            'query': query,
            'top_results': attention_results[:top_k],
            'performance': {
                'vector_search_time': vector_time,
                'attention_search_time': attention_time,
                'total_search_time': total_time,
                'candidate_count': len(candidate_documents)
            },
            'analysis': analysis,
            'candidate_doc_ids': candidate_doc_ids,
            'vector_scores': vector_scores
        }
        
        logger.info(f"Hybrid search completed in {total_time:.3f}s")
        logger.info(f"Top result score: {attention_results[0]['final_score']:.4f}")
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10,
                    vector_candidates: int = 1000) -> List[Dict]:
        """批量检索"""
        results = []
        
        for query in tqdm(queries, desc="Processing queries"):
            result = self.search(query, top_k, vector_candidates)
            results.append(result)
        
        return results
    
    def compare_with_traditional_rag(self, query: str, top_k: int = 10) -> Dict:
        """与传统RAG对比"""
        # 传统RAG（只使用向量检索）
        vector_start = time.time()
        vector_doc_ids, vector_scores = self.vector_retriever.search(query, top_k=top_k)
        vector_time = time.time() - vector_start
        
        # 混合检索
        hybrid_start = time.time()
        hybrid_results = self.search(query, top_k=top_k)
        hybrid_time = time.time() - hybrid_start
        
        # 对比结果
        comparison = {
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
        
        return comparison
    
    def analyze_performance(self) -> Dict:
        """分析性能统计"""
        if not self.stats['total_search_time']:
            return {"error": "No search statistics available"}
        
        analysis = {
            'total_searches': len(self.stats['total_search_time']),
            'average_times': {
                'vector_search': np.mean(self.stats['vector_search_time']),
                'attention_search': np.mean(self.stats['attention_search_time']),
                'total_search': np.mean(self.stats['total_search_time'])
            },
            'time_distribution': {
                'vector_search': {
                    'min': np.min(self.stats['vector_search_time']),
                    'max': np.max(self.stats['vector_search_time']),
                    'std': np.std(self.stats['vector_search_time'])
                },
                'attention_search': {
                    'min': np.min(self.stats['attention_search_time']),
                    'max': np.max(self.stats['attention_search_time']),
                    'std': np.std(self.stats['attention_search_time'])
                }
            },
            'candidate_analysis': {
                'average_candidates': np.mean(self.stats['candidate_counts']),
                'min_candidates': np.min(self.stats['candidate_counts']),
                'max_candidates': np.max(self.stats['candidate_counts'])
            }
        }
        
        return analysis
    
    def get_attention_insights(self, query: str, top_k: int = 10) -> Dict:
        """获取注意力机制的洞察"""
        results = self.search(query, top_k=top_k)
        
        insights = {
            'query': query,
            'attention_patterns': [],
            'fusion_weights': [],
            'query_type_analysis': []
        }
        
        for result in results['top_results'][:5]:  # 分析前5个结果
            # 注意力模式
            pattern_type = result['pattern_type'].argmax().item()
            insights['attention_patterns'].append({
                'doc_id': result['doc_id'],
                'pattern_type': pattern_type,
                'pattern_name': self._get_pattern_name(pattern_type)
            })
            
            # 融合权重
            fusion_weights = result['fusion_info']['dynamic_weights'][0]
            insights['fusion_weights'].append({
                'doc_id': result['doc_id'],
                'vector_weight': fusion_weights[0].item(),
                'attention_weight': fusion_weights[1].item()
            })
            
            # 查询类型
            query_type = result['fusion_info']['query_type'][0].argmax().item()
            insights['query_type_analysis'].append({
                'doc_id': result['doc_id'],
                'query_type': query_type,
                'query_type_name': self._get_query_type_name(query_type)
            })
        
        return insights
    
    def _get_pattern_name(self, pattern_id: int) -> str:
        """获取注意力模式名称"""
        patterns = {
            0: "Focused",      # 集中注意力
            1: "Distributed",  # 分散注意力
            2: "Hierarchical", # 层次注意力
            3: "Sequential",   # 顺序注意力
            4: "Random"        # 随机注意力
        }
        return patterns.get(pattern_id, "Unknown")
    
    def _get_query_type_name(self, type_id: int) -> str:
        """获取查询类型名称"""
        types = {
            0: "Factual",      # 事实查询
            1: "Analytical",   # 分析查询
            2: "Comparative",  # 比较查询
            3: "Creative"      # 创造性查询
        }
        return types.get(type_id, "Unknown")
    
    def save_system(self, filepath: str):
        """保存系统状态"""
        import pickle
        
        system_state = {
            'vector_retriever': self.vector_retriever,
            'attention_retriever': self.attention_retriever,
            'doc_index': self.doc_index,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_state, f)
        
        logger.info(f"System saved to {filepath}")
    
    def load_system(self, filepath: str):
        """加载系统状态"""
        import pickle
        
        with open(filepath, 'rb') as f:
            system_state = pickle.load(f)
        
        self.vector_retriever = system_state['vector_retriever']
        self.attention_retriever = system_state['attention_retriever']
        self.doc_index = system_state['doc_index']
        self.stats = system_state['stats']
        
        logger.info(f"System loaded from {filepath}") 