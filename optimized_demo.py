"""
优化演示脚本 - 改进注意力得分计算，去除停用词处理
"""

import torch
import numpy as np
import logging
from typing import List, Dict
import json
import time
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedAttentionRetriever:
    """优化的注意力检索器"""
    
    def __init__(self):
        self.attention_patterns = {
            0: "Focused",      # 集中注意力
            1: "Distributed",  # 分散注意力
            2: "Hierarchical", # 层次注意力
            3: "Sequential",   # 顺序注意力
            4: "Random"        # 随机注意力
        }
        
        self.query_types = {
            0: "Factual",      # 事实查询
            1: "Analytical",   # 分析查询
            2: "Comparative",  # 比较查询
            3: "Creative"      # 创造性查询
        }
    
    def analyze_query_type(self, query: str) -> Dict:
        """分析查询类型"""
        query_lower = query.lower()
        
        # 事实性查询
        if any(word in query_lower for word in ['什么', '如何', '哪里', '何时', '谁']):
            query_type = 0
            confidence = 0.8
        # 分析性查询
        elif any(word in query_lower for word in ['分析', '解释', '原因', '影响']):
            query_type = 1
            confidence = 0.7
        # 比较性查询
        elif any(word in query_lower for word in ['区别', '比较', '差异', 'vs', '对比']):
            query_type = 2
            confidence = 0.9
        # 创造性查询
        elif any(word in query_lower for word in ['创新', '设计', '开发', '创建']):
            query_type = 3
            confidence = 0.6
        else:
            query_type = 0  # 默认为事实性
            confidence = 0.5
        
        return {
            'type': query_type,
            'type_name': self.query_types[query_type],
            'confidence': confidence
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词 - 不去除停用词"""
        # 简单的分词，保留所有词汇
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def compute_semantic_similarity(self, query: str, document: str) -> float:
        """计算语义相似度 - 改进版本"""
        query_words = self.extract_keywords(query)
        doc_words = self.extract_keywords(document)
        
        if not query_words:
            return 0.0
        
        # 1. 词汇重叠度
        overlap = set(query_words).intersection(set(doc_words))
        overlap_ratio = len(overlap) / len(query_words)
        
        # 2. 语义相关性（基于词汇共现）
        semantic_score = 0.0
        for q_word in query_words:
            # 检查文档中是否包含相关词汇
            if q_word in doc_words:
                semantic_score += 1.0
            # 检查同义词或相关词汇
            elif self._has_semantic_relation(q_word, doc_words):
                semantic_score += 0.8
        
        semantic_score = semantic_score / len(query_words)
        
        # 3. 位置权重（关键词在文档中的位置）
        position_score = self._compute_position_score(query_words, document)
        
        # 4. 综合得分
        final_score = (overlap_ratio * 0.4 + semantic_score * 0.4 + position_score * 0.2)
        
        return min(1.0, max(0.0, final_score))
    
    def _has_semantic_relation(self, word: str, doc_words: List[str]) -> bool:
        """检查语义关系"""
        # 简单的语义关系检查
        semantic_groups = {
            'ai': ['人工智能', '机器学习', '深度学习', '神经网络'],
            'machine': ['机器学习', '深度学习', '神经网络', '算法'],
            'learning': ['学习', '训练', '模型', '算法'],
            'neural': ['神经网络', '深度学习', '机器学习'],
            'nlp': ['自然语言处理', '语言', '文本', '翻译'],
            'vision': ['计算机视觉', '图像', '视频', '识别'],
            'data': ['数据', '大数据', '分析', '处理'],
            'cloud': ['云计算', '云服务', '分布式', '网络'],
            'blockchain': ['区块链', '分布式', '加密', '交易']
        }
        
        for group_word, related_words in semantic_groups.items():
            if word in group_word or group_word in word:
                for related_word in related_words:
                    if related_word in doc_words:
                        return True
        
        return False
    
    def _compute_position_score(self, query_words: List[str], document: str) -> float:
        """计算位置得分"""
        doc_lower = document.lower()
        total_positions = []
        
        for word in query_words:
            pos = doc_lower.find(word)
            if pos != -1:
                # 位置越靠前，得分越高
                position_score = 1.0 - (pos / len(document))
                total_positions.append(position_score)
        
        if not total_positions:
            return 0.0
        
        return np.mean(total_positions)
    
    def compute_attention_pattern(self, query: str, document: str) -> Dict:
        """计算注意力模式"""
        query_words = self.extract_keywords(query)
        doc_words = self.extract_keywords(document)
        
        # 计算重叠
        overlap = set(query_words).intersection(set(doc_words))
        overlap_ratio = len(overlap) / len(query_words) if query_words else 0
        
        # 根据重叠模式判断注意力类型
        if overlap_ratio > 0.7:
            pattern_type = 0  # Focused
        elif overlap_ratio > 0.3:
            pattern_type = 1  # Distributed
        elif overlap_ratio > 0.1:
            pattern_type = 2  # Hierarchical
        elif overlap_ratio > 0:
            pattern_type = 3  # Sequential
        else:
            pattern_type = 4  # Random
        
        return {
            'pattern_type': pattern_type,
            'pattern_name': self.attention_patterns[pattern_type],
            'overlap_ratio': overlap_ratio,
            'overlap_words': list(overlap)
        }
    
    def dynamic_fusion(self, vector_score: float, attention_score: float, 
                      query_analysis: Dict, attention_analysis: Dict) -> Dict:
        """动态融合 - 优化版本"""
        
        # 根据查询类型调整权重
        query_type = query_analysis['type']
        if query_type == 0:  # 事实性查询
            vector_weight = 0.5
            attention_weight = 0.5
        elif query_type == 1:  # 分析性查询
            vector_weight = 0.4
            attention_weight = 0.6
        elif query_type == 2:  # 比较性查询
            vector_weight = 0.2
            attention_weight = 0.8
        else:  # 创造性查询
            vector_weight = 0.3
            attention_weight = 0.7
        
        # 根据注意力模式微调权重
        pattern_type = attention_analysis['pattern_type']
        if pattern_type == 0:  # Focused - 更重视注意力
            attention_weight += 0.1
            vector_weight -= 0.1
        elif pattern_type == 4:  # Random - 更重视向量
            vector_weight += 0.1
            attention_weight -= 0.1
        
        # 根据注意力得分强度调整权重
        if attention_score > 0.7:
            # 注意力得分很高，增加注意力权重
            attention_weight += 0.1
            vector_weight -= 0.1
        elif attention_score < 0.2:
            # 注意力得分很低，增加向量权重
            vector_weight += 0.1
            attention_weight -= 0.1
        
        # 确保权重和为1
        total_weight = vector_weight + attention_weight
        vector_weight /= total_weight
        attention_weight /= total_weight
        
        # 计算最终得分
        final_score = vector_weight * vector_score + attention_weight * attention_score
        
        return {
            'final_score': final_score,
            'vector_weight': vector_weight,
            'attention_weight': attention_weight,
            'vector_score': vector_score,
            'attention_score': attention_score
        }

class OptimizedHybridSystem:
    """优化的混合检索系统"""
    
    def __init__(self):
        self.attention_retriever = OptimizedAttentionRetriever()
        self.documents = []
        self.stats = {
            'query_types': [],
            'attention_patterns': [],
            'fusion_weights': [],
            'attention_scores': [],
            'vector_scores': []
        }
    
    def build_index(self, documents: List[str]):
        """构建索引"""
        self.documents = documents
        logger.info(f"Built index for {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """优化检索"""
        start_time = time.time()
        
        # 1. 分析查询类型
        query_analysis = self.attention_retriever.analyze_query_type(query)
        logger.info(f"查询类型: {query_analysis['type_name']} (置信度: {query_analysis['confidence']:.2f})")
        
        # 2. 模拟向量检索（保持原有逻辑）
        vector_scores = np.random.rand(len(self.documents))
        
        # 3. 计算注意力得分和模式
        results = []
        for i, doc in enumerate(self.documents):
            # 计算注意力模式
            attention_analysis = self.attention_retriever.compute_attention_pattern(query, doc)
            
            # 计算语义相似度（改进的注意力得分）
            attention_score = self.attention_retriever.compute_semantic_similarity(query, doc)
            
            # 动态融合
            fusion_result = self.attention_retriever.dynamic_fusion(
                vector_scores[i], attention_score, query_analysis, attention_analysis
            )
            
            results.append({
                'doc_id': i,
                'document': doc,
                'query_analysis': query_analysis,
                'attention_analysis': attention_analysis,
                'fusion_result': fusion_result
            })
        
        # 4. 排序
        results.sort(key=lambda x: x['fusion_result']['final_score'], reverse=True)
        
        # 5. 统计
        self._update_stats(results)
        
        total_time = time.time() - start_time
        
        return {
            'query': query,
            'query_analysis': query_analysis,
            'top_results': results[:top_k],
            'all_results': results,
            'performance': {'total_time': total_time}
        }
    
    def _update_stats(self, results: List[Dict]):
        """更新统计信息"""
        for result in results:
            self.stats['query_types'].append(result['query_analysis']['type'])
            self.stats['attention_patterns'].append(result['attention_analysis']['pattern_type'])
            self.stats['fusion_weights'].append({
                'vector': result['fusion_result']['vector_weight'],
                'attention': result['fusion_result']['attention_weight']
            })
            self.stats['attention_scores'].append(result['fusion_result']['attention_score'])
            self.stats['vector_scores'].append(result['fusion_result']['vector_score'])
    
    def analyze_performance(self) -> Dict:
        """分析性能"""
        if not self.stats['query_types']:
            return {"error": "No data available"}
        
        # 查询类型分布
        query_type_counts = {}
        for query_type in self.stats['query_types']:
            query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1
        
        # 注意力模式分布
        pattern_counts = {}
        for pattern in self.stats['attention_patterns']:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # 融合权重统计
        avg_vector_weight = np.mean([w['vector'] for w in self.stats['fusion_weights']])
        avg_attention_weight = np.mean([w['attention'] for w in self.stats['fusion_weights']])
        
        # 得分统计
        avg_attention_score = np.mean(self.stats['attention_scores'])
        avg_vector_score = np.mean(self.stats['vector_scores'])
        
        return {
            'query_type_distribution': query_type_counts,
            'attention_pattern_distribution': pattern_counts,
            'average_fusion_weights': {
                'vector': avg_vector_weight,
                'attention': avg_attention_weight
            },
            'average_scores': {
                'attention': avg_attention_score,
                'vector': avg_vector_score
            },
            'total_queries': len(self.stats['query_types'])
        }

def main():
    """主函数"""
    logger.info("=== 优化注意力检索演示 ===")
    
    # 创建示例数据
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
    
    # 不同类型的查询
    queries = [
        "什么是人工智能？",                    # 事实性查询
        "机器学习和深度学习有什么区别？",      # 比较性查询
        "自然语言处理有哪些应用？",            # 事实性查询
        "分析AI在医疗领域的影响",              # 分析性查询
        "如何设计一个智能推荐系统？"           # 创造性查询
    ]
    
    # 初始化系统
    system = OptimizedHybridSystem()
    system.build_index(documents)
    
    # 测试每个查询
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"查询 {i}: {query}")
        logger.info(f"{'='*60}")
        
        # 执行检索
        results = system.search(query, top_k=3)
        
        # 显示结果
        logger.info(f"查询类型: {results['query_analysis']['type_name']}")
        logger.info(f"置信度: {results['query_analysis']['confidence']:.2f}")
        logger.info("")
        
        logger.info("检索结果:")
        for j, result in enumerate(results['top_results'], 1):
            fusion = result['fusion_result']
            attention = result['attention_analysis']
            
            logger.info(f"  排名 {j}:")
            logger.info(f"    文档: {result['document'][:80]}...")
            logger.info(f"    最终得分: {fusion['final_score']:.4f}")
            logger.info(f"    向量得分: {fusion['vector_score']:.4f} (权重: {fusion['vector_weight']:.2f})")
            logger.info(f"    注意力得分: {fusion['attention_score']:.4f} (权重: {fusion['attention_weight']:.2f})")
            logger.info(f"    注意力模式: {attention['pattern_name']}")
            logger.info(f"    重叠度: {attention['overlap_ratio']:.2f}")
            if attention['overlap_words']:
                logger.info(f"    重叠词汇: {', '.join(attention['overlap_words'])}")
            logger.info("")
    
    # 性能分析
    logger.info(f"\n{'='*60}")
    logger.info("性能分析")
    logger.info(f"{'='*60}")
    
    performance = system.analyze_performance()
    
    logger.info("查询类型分布:")
    for query_type, count in performance['query_type_distribution'].items():
        type_name = system.attention_retriever.query_types[query_type]
        logger.info(f"  {type_name}: {count} 次")
    
    logger.info("\n注意力模式分布:")
    for pattern, count in performance['attention_pattern_distribution'].items():
        pattern_name = system.attention_retriever.attention_patterns[pattern]
        logger.info(f"  {pattern_name}: {count} 次")
    
    logger.info(f"\n平均融合权重:")
    logger.info(f"  向量权重: {performance['average_fusion_weights']['vector']:.3f}")
    logger.info(f"  注意力权重: {performance['average_fusion_weights']['attention']:.3f}")
    
    logger.info(f"\n平均得分:")
    logger.info(f"  注意力得分: {performance['average_scores']['attention']:.3f}")
    logger.info(f"  向量得分: {performance['average_scores']['vector']:.3f}")
    
    logger.info(f"\n总查询次数: {performance['total_queries']}")
    
    logger.info("\n演示完成！")

if __name__ == "__main__":
    main() 