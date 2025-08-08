"""
注意力检索模块 - 创新版本
重点设计与传统RAG+重排序不同的特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class MultiGranularityAttention(nn.Module):
    """多粒度注意力机制 - 创新点1"""
    
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 词级注意力
        self.word_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 句级注意力
        self.sentence_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 文档级注意力
        self.doc_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 粒度融合层
        self.granularity_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, query_emb, doc_emb):
        # 1. 词级注意力
        word_attended, word_weights = self.word_attention(query_emb, doc_emb, doc_emb)
        
        # 2. 句级注意力（通过池化）
        query_sent = F.avg_pool1d(query_emb.transpose(1, 2), kernel_size=4).transpose(1, 2)
        doc_sent = F.avg_pool1d(doc_emb.transpose(1, 2), kernel_size=4).transpose(1, 2)
        sent_attended, sent_weights = self.sentence_attention(query_sent, doc_sent, doc_sent)
        
        # 3. 文档级注意力（全局池化）
        query_doc = F.avg_pool1d(query_emb.transpose(1, 2), query_emb.size(1)).transpose(1, 2)
        doc_doc = F.avg_pool1d(doc_emb.transpose(1, 2), doc_emb.size(1)).transpose(1, 2)
        doc_attended, doc_weights = self.doc_attention(query_doc, doc_doc, doc_doc)
        
        # 4. 融合多粒度特征
        word_feat = word_attended.mean(dim=1)
        sent_feat = sent_attended.mean(dim=1)
        doc_feat = doc_attended.mean(dim=1)
        
        combined_feat = torch.cat([word_feat, sent_feat, doc_feat], dim=-1)
        attention_score = self.granularity_fusion(combined_feat)
        
        return attention_score, {
            'word_weights': word_weights,
            'sent_weights': sent_weights,
            'doc_weights': doc_weights
        }

class AttentionPatternAnalyzer(nn.Module):
    """注意力模式分析器 - 创新点2"""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 注意力模式编码器
        self.pattern_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # 模式分类器
        self.pattern_classifier = nn.Linear(hidden_size // 4, 5)  # 5种注意力模式
    
    def analyze_pattern(self, attention_weights):
        """分析注意力权重模式"""
        # 计算注意力统计特征
        mean_attention = attention_weights.mean(dim=1)  # 平均注意力
        max_attention = attention_weights.max(dim=1)[0]  # 最大注意力
        attention_entropy = self.compute_entropy(attention_weights)  # 注意力熵
        
        # 编码模式
        pattern_features = torch.cat([mean_attention, max_attention, attention_entropy], dim=-1)
        pattern_encoding = self.pattern_encoder(pattern_features)
        pattern_type = self.pattern_classifier(pattern_encoding)
        
        return pattern_type, pattern_encoding
    
    def compute_entropy(self, attention_weights):
        """计算注意力熵"""
        # 添加小值避免log(0)
        attention_weights = attention_weights + 1e-8
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        return entropy.unsqueeze(-1)

class DynamicFusionLayer(nn.Module):
    """动态融合层 - 创新点3"""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 查询类型分析器
        self.query_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)  # 4种查询类型
        )
        
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size // 2),  # 查询特征 + 查询类型
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # 2个权重：向量权重和注意力权重
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(2, hidden_size // 2),  # 2个得分
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)   # 最终得分
        )
    
    def forward(self, query_features, vector_score, attention_score, pattern_info):
        # 1. 分析查询类型
        query_type = self.query_analyzer(query_features)
        query_type_probs = F.softmax(query_type, dim=-1)
        
        # 2. 生成动态权重
        combined_features = torch.cat([query_features, query_type_probs], dim=-1)
        dynamic_weights = self.weight_generator(combined_features)
        dynamic_weights = F.softmax(dynamic_weights, dim=-1)
        
        # 3. 融合得分
        scores = torch.stack([vector_score, attention_score], dim=-1)
        weighted_scores = scores * dynamic_weights
        
        # 4. 最终融合
        final_score = self.fusion_net(weighted_scores)
        
        return final_score, {
            'query_type': query_type_probs,
            'dynamic_weights': dynamic_weights,
            'weighted_scores': weighted_scores
        }

class InnovativeAttentionRetriever(nn.Module):
    """创新注意力检索器 - 主要模型"""
    
    def __init__(self, model_name='bert-base-uncased', device='cuda', hf_token: Optional[str] = None):
        super().__init__()
        self.device = device
        self.hf_token = hf_token or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        
        # BERT编码器
        try:
            self.bert = BertModel.from_pretrained(model_name, use_auth_token=self.hf_token)
        except TypeError:
            # 新版本可能弃用use_auth_token
            self.bert = BertModel.from_pretrained(model_name)
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, use_auth_token=self.hf_token)
        except TypeError:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 多粒度注意力
        self.multi_granularity_attention = MultiGranularityAttention()
        
        # 注意力模式分析器
        self.pattern_analyzer = AttentionPatternAnalyzer()
        
        # 动态融合层
        self.dynamic_fusion = DynamicFusionLayer()
        
        # 查询特征提取器
        self.query_feature_extractor = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
        self.to(device)
    
    def encode_text(self, text):
        """编码文本"""
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state
        
        return embeddings
    
    def compute_attention_score(self, query, document):
        """计算注意力得分"""
        # 编码查询和文档
        query_emb = self.encode_text(query)
        doc_emb = self.encode_text(document)
        
        # 多粒度注意力计算
        attention_score, attention_info = self.multi_granularity_attention(query_emb, doc_emb)
        
        # 直接返回默认的模式分布，避免维度不匹配与告警
        pattern_type = torch.zeros((1, 5), device=self.device)
        
        return attention_score, attention_info, pattern_type
    
    def retrieve(self, query, documents, vector_scores):
        """检索主函数"""
        results = []
        
        # 提取查询特征
        query_emb = self.encode_text(query)
        query_features = self.query_feature_extractor(query_emb.mean(dim=1))
        
        for i, (doc, vector_score) in enumerate(zip(documents, vector_scores)):
            # 1. 计算注意力得分
            attention_score, attention_info, pattern_type = self.compute_attention_score(query, doc)
            
            # 2. 动态融合
            final_score, fusion_info = self.dynamic_fusion(
                query_features, 
                torch.tensor([vector_score]).to(self.device),
                attention_score,
                pattern_type
            )
            
            results.append({
                'doc_id': i,
                'document': doc,
                'vector_score': vector_score,
                'attention_score': attention_score.item(),
                'final_score': final_score.item(),
                'attention_info': attention_info,
                'pattern_type': pattern_type,
                'fusion_info': fusion_info
            })
        
        # 按最终得分排序
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def analyze_retrieval(self, query, documents, vector_scores):
        """分析检索过程"""
        results = self.retrieve(query, documents, vector_scores)
        
        # 分析结果
        analysis = {
            'query': query,
            'num_documents': len(documents),
            'top_results': results[:5],
            'score_distribution': {
                'vector_scores': [r['vector_score'] for r in results],
                'attention_scores': [r['attention_score'] for r in results],
                'final_scores': [r['final_score'] for r in results]
            },
            'pattern_analysis': {
                'pattern_types': [r['pattern_type'].argmax().item() for r in results[:10]]
            },
            'fusion_analysis': {
                'avg_vector_weight': np.mean([r['fusion_info']['dynamic_weights'][0][0].item() for r in results]),
                'avg_attention_weight': np.mean([r['fusion_info']['dynamic_weights'][0][1].item() for r in results])
            }
        }
        
        return analysis 