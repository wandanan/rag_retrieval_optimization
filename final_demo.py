"""
最终优化演示 - 使用外部测试数据，重点解决注意力得分过低的问题
"""

import torch
import numpy as np
import logging
from typing import List, Dict
import json
import time
import re
import os
from dotenv import load_dotenv  # type: ignore
# 新增：可选中文分词
try:
    import jieba  # type: ignore
    JIEBA_AVAILABLE = True
except Exception:
    jieba = None
    JIEBA_AVAILABLE = False

# 新增：可选向量检索（带回退）
try:
    from models.vector_retriever import VectorRetriever  # type: ignore
    VECTOR_AVAILABLE = True
except Exception:
    VectorRetriever = None  # type: ignore
    VECTOR_AVAILABLE = False

from data.test_data import get_test_data, get_semantic_relations
from models.attention_retriever import InnovativeAttentionRetriever  # 句子级多头注意力

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalAttentionRetriever:
    """最终优化的注意力检索器"""
    
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
        
        # 加载语义关系（仅作回退）
        self.semantic_relations = get_semantic_relations()
        # 新增：构建反向索引（中文词 -> 语义组键）
        self.token_to_groups = self._build_token_to_groups(self.semantic_relations)
        if not JIEBA_AVAILABLE:
            logger.info("未检测到 jieba，将回退到正则+字符n-gram分词；建议安装 jieba 获得更优中文分词效果")
        # 新增：简单中文停用词（仅用于注意力计算过滤）
        self.stopwords = set([
            '的','了','和','与','及','是','在','为','对','于','有','中','上','下','等','也','都','并','或','把','被','就','而','其','及其','一个','一种','我们','你','我','他','她','它','吗','呢','啊','吧','着','地','得','之'
        ])
        # 新增：可注入的语义编码器（向量后端），以及文档token嵌入缓存
        self.semantic_embedder = None  # 需具有 embed_texts(List[str]) -> np.ndarray
        self._doc_token_embed_cache = {}
        # 新增：语义匹配的token上限
        self.max_doc_sem_tokens = 32
        self.max_query_sem_tokens = 8

    def _build_token_to_groups(self, semantic_relations: Dict[str, List[str]]) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for group_key, related_words in semantic_relations.items():
            # 将组键与相关词都映射回该组键
            for token in [group_key] + list(related_words):
                token_str = str(token)
                mapping.setdefault(token_str, []).append(group_key)
        return mapping
    
    def analyze_query_type(self, query: str) -> Dict:
        """分析查询类型（先判更具体的比较/分析，再回退事实）"""
        query_lower = query.lower()
        
        # 比较性查询（优先于“什么”）
        if any(word in query for word in ['区别', '比较', '差异', 'vs', '对比']):
            query_type = 2
            confidence = 0.9
        # 分析性查询
        elif any(word in query for word in ['分析', '解释', '原因', '影响']):
            query_type = 1
            confidence = 0.75
        # 创造性查询
        elif any(word in query for word in ['创新', '设计', '开发', '创建']):
            query_type = 3
            confidence = 0.65
        # 事实性查询
        elif any(word in query_lower for word in ['什么', '如何', '哪里', '何时', '谁']):
            query_type = 0
            confidence = 0.8
        else:
            query_type = 0  # 默认为事实性
            confidence = 0.55
        
        return {
            'type': query_type,
            'type_name': self.query_types[query_type],
            'confidence': confidence
        }
    
    def _is_cjk(self, ch: str) -> bool:
        return '\u4e00' <= ch <= '\u9fff'

    def _char_ngrams(self, text: str, n_values: List[int] = [2, 3]) -> List[str]:
        compact = re.sub(r"\s+", "", text)
        grams: List[str] = []
        for n in n_values:
            if len(compact) >= n:
                grams.extend([compact[i:i+n] for i in range(len(compact) - n + 1)])
        return grams

    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词 - 优先中文分词，回退字符n-gram与英文单词"""
        text = text.strip()
        tokens: List[str] = []
        
        # 优先使用 jieba 分词
        if JIEBA_AVAILABLE:
            tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        else:
            # 英文/数字词
            tokens = re.findall(r"[A-Za-z0-9_]+", text)
        
        # 对中文文本增加2-3字n-gram，增强子串匹配鲁棒性
        if any(self._is_cjk(ch) for ch in text):
            tokens.extend(self._char_ngrams(text, [2, 3]))
        
        return tokens

    def extract_semantic_tokens(self, text: str, for_document: bool) -> List[str]:
        """提取用于嵌入语义匹配的精简tokens（限制数量以提速）"""
        text = text.strip()
        # 仅用分词，不使用n-gram，以减少token爆炸
        if JIEBA_AVAILABLE:
            tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        else:
            tokens = re.findall(r"[A-Za-z0-9_]+", text)
        # 过滤
        tokens = [t for t in tokens if (len(t) >= 2 and t not in self.stopwords)]
        # 去重且保序
        seen = set()
        unique_tokens: List[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)
        # 按长度降序，优先保留更具信息量的token
        unique_tokens.sort(key=lambda x: len(x), reverse=True)
        # 限制数量
        if for_document:
            return unique_tokens[:self.max_doc_sem_tokens]
        else:
            return unique_tokens[:self.max_query_sem_tokens]
    
    def set_semantic_embedder(self, embedder) -> None:
        """注入向量编码器（必须实现 embed_texts(List[str]) -> np.ndarray）"""
        self.semantic_embedder = embedder

    def precompute_document_semantic_embeddings(self, documents: List[str], batch_size: int = 256) -> None:
        """预计算并缓存每个文档的精简token嵌入，减少查询时的远程调用"""
        if self.semantic_embedder is None:
            return
        token_lists: List[List[str]] = []
        doc_keys: List[int] = []
        for doc in documents:
            d_sem = self.extract_semantic_tokens(doc, for_document=True)
            token_lists.append(d_sem)
            doc_keys.append(hash(doc))
        # 去重全局token以减少总编码量
        unique_tokens: List[str] = []
        token_to_index: Dict[str, int] = {}
        for lst in token_lists:
            for t in lst:
                if t not in token_to_index:
                    token_to_index[t] = len(unique_tokens)
                    unique_tokens.append(t)
        # 分批编码
        all_embs = []
        for i in range(0, len(unique_tokens), batch_size):
            chunk = unique_tokens[i:i+batch_size]
            embs = self.semantic_embedder.embed_texts(chunk)
            # 归一化
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            all_embs.append(embs / norms)
        if all_embs:
            all_embs = np.vstack(all_embs)
        else:
            all_embs = np.zeros((0, 768), dtype=np.float32)
        # 回填每个文档的嵌入矩阵到缓存
        for d_tokens, key in zip(token_lists, doc_keys):
            if not d_tokens:
                self._doc_token_embed_cache[key] = np.zeros((0, all_embs.shape[1] if all_embs.size else 768), dtype=np.float32)
                continue
            idxs = [token_to_index[t] for t in d_tokens if t in token_to_index]
            d_emb = all_embs[idxs] if len(idxs) > 0 else np.zeros((0, all_embs.shape[1]), dtype=np.float32)
            self._doc_token_embed_cache[key] = d_emb

    def _embed_tokens(self, tokens: List[str]) -> np.ndarray:
        if self.semantic_embedder is None:
            raise RuntimeError("未设置语义编码器")
        if not tokens:
            return np.zeros((0, 768), dtype=np.float32)
        embs = self.semantic_embedder.embed_texts(tokens)
        # 归一化
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        return embs / norms

    def _compute_embedding_semantic_score(self, q_tokens: List[str], doc_text: str, d_tokens: List[str]) -> float:
        """基于嵌入的语义匹配分：对每个查询token取与文档token的最大相似度，并做阈值化平均"""
        if not q_tokens or not d_tokens:
            return 0.0
        # 缓存文档token嵌入（按文档内容哈希缓存）
        cache_key = hash(doc_text)
        if cache_key in self._doc_token_embed_cache:
            d_emb = self._doc_token_embed_cache[cache_key]
        else:
            d_emb = self._embed_tokens(d_tokens)
            self._doc_token_embed_cache[cache_key] = d_emb
        q_emb = self._embed_tokens(q_tokens)
        if q_emb.size == 0 or d_emb.size == 0:
            return 0.0
        sims = np.matmul(q_emb, d_emb.T)  # 余弦相似度（已归一化）
        max_sim_per_q = sims.max(axis=1)
        tau = 0.55
        soft = np.clip((max_sim_per_q - tau) / (1.0 - tau), 0.0, 1.0)
        return float(np.mean(soft))

    def compute_lightweight_attention_score(self, query: str, document: str) -> float:
        """轻量注意力：基于嵌入的 scaled dot-product attention，替代启发式注意力得分
        - Q 来自查询语义tokens嵌入；K/V 来自文档语义tokens嵌入（预计算并缓存）
        - logits = QK^T / T（T为温度，默认0.2使分布更锐）
        - 输出 O = softmax(logits) @ V；以 cos(q_i, o_i) 按查询token求平均得到标量注意力分
        """
        if self.semantic_embedder is None:
            # 回退到旧的启发式注意力
            return self.compute_enhanced_attention_score(query, document)
        # 精简tokens
        q_sem = self.extract_semantic_tokens(query, for_document=False)
        d_sem = self.extract_semantic_tokens(document, for_document=True)
        if not q_sem or not d_sem:
            return 0.0
        # 嵌入（已缓存文档）
        cache_key = hash(document)
        if cache_key in self._doc_token_embed_cache:
            k_emb = self._doc_token_embed_cache[cache_key]
        else:
            k_emb = self._embed_tokens(d_sem)
            self._doc_token_embed_cache[cache_key] = k_emb
        q_emb = self._embed_tokens(q_sem)
        if q_emb.size == 0 or k_emb.size == 0:
            return 0.0
        # scaled dot-product attention
        temperature = 0.2  # 较小温度 -> 更锐利的注意力
        logits = np.matmul(q_emb, k_emb.T) / max(1e-6, temperature)
        # 数值稳定：减去行最大值
        logits = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
        weights = weights / weights_sum  # softmax
        # 聚合得到输出
        o_emb = np.matmul(weights, k_emb)
        # 计算与查询的对齐（已归一化，内积即cos）
        sims = np.sum(o_emb * q_emb, axis=1)
        score = float(np.clip(np.mean(sims), 0.0, 1.0))
        return score
    
    def compute_enhanced_attention_score(self, query: str, document: str) -> float:
        """计算增强的注意力得分（中文友好：分词+同义词/嵌入+子串+位置）"""
        query_words_all = self.extract_keywords(query)
        doc_words_all = self.extract_keywords(document)
        
        # 过滤停用词与过短token以提升区分度
        def _filter(tokens: List[str]) -> List[str]:
            return [t for t in tokens if (len(t) >= 2 and t not in self.stopwords)]
        query_words_f = _filter(query_words_all)
        doc_words_f = set(_filter(doc_words_all))
        doc_text = document
        
        # 如果过滤后为空，回退到原tokens
        query_words = query_words_f if query_words_f else query_words_all
        doc_words = doc_words_f if doc_words_f else set(doc_words_all)
        
        if not query_words:
            return 0.0
        
        unique_query_words = list(dict.fromkeys(query_words))
        
        # 1) 直接词汇/子串匹配
        direct_hits = 0
        for qw in unique_query_words:
            if qw in doc_words:
                direct_hits += 1
            elif len(qw) >= 2 and qw in doc_text:
                direct_hits += 0.5
        # 2) 语义关系：优先嵌入匹配（通用），否则回退静态同义词
        use_embedding = self.semantic_embedder is not None
        if use_embedding:
            q_sem = self.extract_semantic_tokens(query, for_document=False)
            d_sem = self.extract_semantic_tokens(document, for_document=True)
            emb_semantic = self._compute_embedding_semantic_score(q_sem, doc_text, d_sem)
            # 权重方案（通用优先）：direct 0.20 / emb 0.60 / pos 0.15 / len 0.05
            direct_score = (direct_hits / max(1, len(unique_query_words))) * 0.20
            semantic_score = emb_semantic * 0.60
            position_score = self._compute_enhanced_position_score(unique_query_words, doc_text) * 0.15
            length_score = self._compute_length_match_score(query, document) * 0.05
        else:
            # 回退静态同义词匹配
            semantic_accum = 0.0
            for qw in unique_query_words:
                groups = self.token_to_groups.get(qw, [])
                if not groups:
                    groups = [g for token, gs in self.token_to_groups.items() if (len(qw) >= 2 and qw in token) for g in gs]
                max_group_score = 0.0
                for g in groups:
                    related_words = self.semantic_relations.get(g, [])
                    if not related_words:
                        continue
                    matches = 0
                    for rw in related_words:
                        if rw in doc_words:
                            matches += 1
                        elif len(rw) >= 2 and rw in doc_text:
                            matches += 0.5
                    group_score = matches / len(related_words)
                    max_group_score = max(max_group_score, group_score)
                semantic_accum += max_group_score
            # 原权重：direct 0.35 / sem 0.40 / pos 0.20 / len 0.05
            direct_score = (direct_hits / max(1, len(unique_query_words))) * 0.35
            semantic_score = (semantic_accum / max(1, len(unique_query_words))) * 0.40
            position_score = self._compute_enhanced_position_score(unique_query_words, doc_text) * 0.20
            length_score = self._compute_length_match_score(query, document) * 0.05
        
        final_score = direct_score + semantic_score + position_score + length_score
        return float(min(1.0, max(0.0, final_score)))
    
    def _compute_enhanced_position_score(self, query_words: List[str], document: str) -> float:
        """计算增强的位置得分（支持同义词与窗口共现）"""
        doc_text = document
        doc_len = max(1, len(doc_text))
        positions: List[int] = []
        
        # 收集每个查询词或其同义词在文档中的最靠前位置
        for qw in query_words:
            # 候选词：自身 + 所在语义组的相关词
            candidates = set([qw])
            for g in self.token_to_groups.get(qw, []):
                for rw in self.semantic_relations.get(g, []):
                    candidates.add(rw)
            
            best_pos = None
            for tok in candidates:
                p = doc_text.find(tok)
                if p != -1:
                    best_pos = p if best_pos is None else min(best_pos, p)
            if best_pos is not None:
                positions.append(best_pos)
        
        if not positions:
            return 0.0
        
        # 靠前得分：位置越靠前越高
        base_scores = [1.0 - (pos / doc_len) for pos in positions]
        base_score = float(np.mean(base_scores))
        
        # 窗口共现加成：多个关键词在近距离内同时出现
        positions_sorted = sorted(positions)
        close_pairs = 0
        total_pairs = 0
        window = max(20, int(0.1 * doc_len))  # 自适应窗口
        for i in range(len(positions_sorted)):
            for j in range(i + 1, len(positions_sorted)):
                total_pairs += 1
                if positions_sorted[j] - positions_sorted[i] <= window:
                    close_pairs += 1
        if total_pairs > 0:
            cooccur_boost = min(0.25, (close_pairs / total_pairs) * 0.25)
        else:
            cooccur_boost = 0.0
        
        return float(min(1.0, base_score * (1.0 + cooccur_boost)))
    
    def _compute_length_match_score(self, query: str, document: str) -> float:
        """计算长度匹配得分"""
        query_length = len(query)
        doc_length = len(document)
        
        # 文档长度适中时得分最高
        if 50 <= doc_length <= 200:
            return 1.0
        elif 20 <= doc_length <= 300:
            return 0.8
        else:
            return 0.5
    
    def compute_attention_pattern(self, query: str, document: str) -> Dict:
        """计算注意力模式"""
        # 计算重叠：若有嵌入，则用精简tokens的语义重叠替代
        if self.semantic_embedder is not None:
            q_sem = self.extract_semantic_tokens(query, for_document=False)
            d_sem = self.extract_semantic_tokens(document, for_document=True)
            sem_score = self._compute_embedding_semantic_score(q_sem, document, d_sem)
            overlap_ratio = min(1.0, sem_score)
            overlap = []
        else:
            query_words = self.extract_keywords(query)
            doc_words = self.extract_keywords(document)
            overlap = set(query_words).intersection(set(doc_words))
            overlap_ratio = len(overlap) / len(query_words) if query_words else 0
        
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
            'overlap_words': list(overlap) if overlap else []
        }
    
    def dynamic_fusion(self, vector_score: float, attention_score: float, 
                      query_analysis: Dict, attention_analysis: Dict) -> Dict:
        """动态融合 - 最终优化版本"""
        
        # 根据查询类型调整权重
        query_type = query_analysis['type']
        if query_type == 0:  # 事实性查询
            vector_weight = 0.4
            attention_weight = 0.6
        elif query_type == 1:  # 分析性查询
            vector_weight = 0.3
            attention_weight = 0.7
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
        
        # 根据注意力得分与重叠度强度调整权重（阈值更敏感）
        overlap_ratio = attention_analysis.get('overlap_ratio', 0.0)
        if attention_score > 0.35:
            attention_weight += 0.15
            vector_weight -= 0.15
        elif overlap_ratio > 0.3:
            attention_weight += 0.10
            vector_weight -= 0.10
        elif attention_score < 0.05:
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

class FinalHybridSystem:
    """最终优化的混合检索系统"""
    
    def __init__(self):
        self.attention_retriever = FinalAttentionRetriever()
        self.documents = []
        self.stats = {
            'query_types': [],
            'attention_patterns': [],
            'fusion_weights': [],
            'attention_scores': [],
            'vector_scores': []
        }
        # 新增：向量检索器（可选）
        self.vector_retriever = None
        self.use_vector = False
        # 新增：候选数（仅对向量Top-N计算注意力）
        self.candidate_k = 20
        # 新增：句子级多头注意力模型（BERT中文）
        self.use_sentence_multihead = True
        self.mh_attn_model = None
    
    def build_index(self, documents: List[str]):
        """构建索引"""
        self.documents = documents
        logger.info(f"Built index for {len(documents)} documents")
        # 尝试启用真实向量检索
        if VECTOR_AVAILABLE:
            try:
                self.vector_retriever = VectorRetriever(
                    model_name='BAAI/bge-large-zh-v1.5',
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    backend='transformers',  # 改为transformers
                    remote_endpoint='http://localhost:11434',
                    query_instruction_for_retrieval='为这个句子生成表示以用于检索相关文章：',
                    hf_token=os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
                )
                self.vector_retriever.build_index(documents)
                self.use_vector = True
                backend_name = 'Ollama' if getattr(self.vector_retriever, 'backend', '') == 'ollama' else 'Transformers'
                logger.info(f"向量检索已启用（{backend_name} + FAISS）")
                # 将语义编码器注入注意力模块（实现通用的嵌入语义匹配）
                self.attention_retriever.set_semantic_embedder(self.vector_retriever)
                # 预计算文档语义token嵌入（一次性，查询复用）
                self.attention_retriever.precompute_document_semantic_embeddings(self.documents)
            except Exception as e:
                self.vector_retriever = None
                self.use_vector = False
                if '401' in str(e) or 'Unauthorized' in str(e):
                    logger.warning("HuggingFace鉴权失败，请设置环境变量 HUGGINGFACE_HUB_TOKEN 或 HF_TOKEN 后重试")
                logger.warning(f"向量检索初始化失败，回退随机分数: {e}")
        else:
            logger.info("未检测到向量检索依赖，回退随机向量分数")

        # 启用句子级多头注意力模型（中文BERT）
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # 使用中文BERT以适配中文句子
            self.mh_attn_model = InnovativeAttentionRetriever(model_name='bert-base-chinese', device=device)
            logger.info("句子级多头注意力已启用（bert-base-chinese）")
        except Exception as e:
            self.mh_attn_model = None
            self.use_sentence_multihead = False
            logger.warning(f"句子级多头注意力初始化失败，将回退轻量注意力: {e}")
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """最终优化检索"""
        start_time = time.time()
        
        # 1. 分析查询类型
        query_analysis = self.attention_retriever.analyze_query_type(query)
        logger.info(f"查询类型: {query_analysis['type_name']} (置信度: {query_analysis['confidence']:.2f})")
        
        # 2. 向量检索/相似度
        if self.use_vector and self.vector_retriever is not None:
            try:
                sims = self.vector_retriever.compute_similarity(query, self.documents)
                # 归一化到[0,1]
                vector_scores = ((sims + 1.0) / 2.0).astype(float)
            except Exception as e:
                logger.warning(f"向量相似度计算失败，回退随机分数: {e}")
                vector_scores = np.random.rand(len(self.documents))
        else:
            # 回退：随机分数
            vector_scores = np.random.rand(len(self.documents))
        
        # 2.5 仅对前N候选计算注意力
        if len(self.documents) > 0:
            candidate_k = self.candidate_k
            candidate_indices = np.argsort(vector_scores)[-candidate_k:][::-1]
        else:
            candidate_indices = []
        
        # 3. 计算注意力得分和模式（仅候选）
        results = []
        for i in candidate_indices:
            doc = self.documents[i]
            # 计算注意力模式（词级重叠或语义重叠）
            attention_analysis = self.attention_retriever.compute_attention_pattern(query, doc)
            
            # 注意力得分：优先句子级多头注意力
            attention_score: float
            if self.use_sentence_multihead and self.mh_attn_model is not None:
                try:
                    # 使用多头注意力融合标量，经sigmoid映射到(0,1)
                    attn_scalar, attn_info, _ = self.mh_attn_model.compute_attention_score(query, doc)
                    attention_score = float(torch.sigmoid(attn_scalar).mean().item())
                except Exception as e:
                    logger.warning(f"句子级多头注意力计算失败，回退轻量注意力: {e}")
                    attention_score = self.attention_retriever.compute_lightweight_attention_score(query, doc)
            elif self.attention_retriever.semantic_embedder is not None:
                attention_score = self.attention_retriever.compute_lightweight_attention_score(query, doc)
            else:
                attention_score = self.attention_retriever.compute_enhanced_attention_score(query, doc)
            
            # 动态融合
            fusion_result = self.attention_retriever.dynamic_fusion(
                float(vector_scores[i]), attention_score, query_analysis, attention_analysis
            )
            
            results.append({
                'doc_id': int(i),
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
    # 加载.env配置（若存在）
    try:
        load_dotenv()
    except Exception:
        pass
    logger.info("=== 最终优化注意力检索演示 ===")
    
    # 加载测试数据
    test_data = get_test_data()
    documents = test_data['documents']
    queries = test_data['queries']
    
    # 初始化系统
    system = FinalHybridSystem()
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