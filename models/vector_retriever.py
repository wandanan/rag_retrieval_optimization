"""
向量检索模块
"""

import torch
import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
import logging
import os

# 新增：可选 Ollama 客户端
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorRetriever:
    """向量检索器（支持 Sentence-Transformers 与 Ollama Embeddings）"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 backend: str = 'sentence-transformers',
                 remote_endpoint: str = 'http://localhost:11434',
                 query_instruction_for_retrieval: Optional[str] = None,
                 hf_token: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.backend = backend  # 'sentence-transformers' | 'ollama'
        self.remote_endpoint = remote_endpoint.rstrip('/')
        
        # BGE 查询指令（仅对encode_query生效）
        if query_instruction_for_retrieval is not None:
            self.query_instruction_for_retrieval = query_instruction_for_retrieval
        else:
            # 自动为中文BGE模型注入推荐指令
            if isinstance(model_name, str) and 'bge' in model_name.lower() and ('zh' in model_name.lower() or 'chinese' in model_name.lower()):
                self.query_instruction_for_retrieval = "为这个句子生成表示以用于检索相关文章："
            else:
                self.query_instruction_for_retrieval = None

        # HF鉴权token：参数优先，其次环境变量
        self.hf_token = hf_token or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        
        # 设置缓存目录
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.dirname(__file__), '..', 'model_cache'))
        os.makedirs(cache_dir, exist_ok=True)
        
        self.encoder = None
        if self.backend == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer  # 延迟导入
            # 传入鉴权token（私有/受限模型需要）
            try:
                logger.info(f"加载模型: {model_name} 到缓存目录: {cache_dir}")
                self.encoder = SentenceTransformer(
                    model_name, 
                    device=device, 
                    use_auth_token=self.hf_token,
                    cache_folder=cache_dir
                )
                logger.info(f"模型加载成功: {model_name}")
            except TypeError:
                # 某些版本不支持use_auth_token参数，回退不传该参数
                self.encoder = SentenceTransformer(
                    model_name, 
                    device=device,
                    cache_folder=cache_dir
                )
                logger.info(f"模型加载成功（无认证）: {model_name}")
            if self.query_instruction_for_retrieval:
                logger.info(f"BGE查询指令已启用: {self.query_instruction_for_retrieval}")
        elif self.backend == 'ollama':
            if not REQUESTS_AVAILABLE:
                raise RuntimeError("requests 未安装，无法使用 Ollama 嵌入后端")
        else:
            raise ValueError(f"未知后端: {self.backend}")
        
        # FAISS索引
        self.index = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Vector retriever initialized with backend={self.backend}, model={model_name}")

    def _encode_ollama(self, texts: List[str]) -> np.ndarray:
        """通过 Ollama /api/embed 生成嵌入"""
        url = f"{self.remote_endpoint}/api/embed"
        payload = {
            "model": self.model_name,
            "input": texts if len(texts) > 1 else texts[0]
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            embeds = data.get("embeddings", [])
            if not embeds:
                raise RuntimeError("Ollama 返回空 embeddings")
            arr = np.array(embeds, dtype=np.float32)
            return arr
        except Exception as e:
            raise RuntimeError(f"调用 Ollama 失败: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """公共批量嵌入接口，供上层进行token级编码"""
        if self.backend == 'sentence-transformers':
            embeddings = self.encoder.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
            return embeddings.astype(np.float32)
        elif self.backend == 'ollama':
            return self._encode_ollama(texts).astype(np.float32)
        else:
            raise ValueError("不支持的编码后端")
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """编码文档"""
        logger.info(f"Encoding {len(documents)} documents via {self.backend}...")
        if self.backend == 'sentence-transformers':
            embeddings = self.encoder.encode(documents, show_progress_bar=True, 
                                           batch_size=32, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        elif self.backend == 'ollama':
            embeddings = self._encode_ollama(documents)
            return embeddings.astype(np.float32)
        else:
            raise ValueError("不支持的编码后端")
    
    def encode_query(self, query: str) -> np.ndarray:
        """编码查询（对BGE模型自动添加检索指令）"""
        if self.backend == 'sentence-transformers':
            if self.query_instruction_for_retrieval:
                query = f"{self.query_instruction_for_retrieval}{query}"
            embedding = self.encoder.encode([query], convert_to_numpy=True)
            return embedding[0].astype(np.float32)
        elif self.backend == 'ollama':
            embedding = self._encode_ollama([query])
            return embedding[0].astype(np.float32)
        else:
            raise ValueError("不支持的编码后端")
    
    def build_index(self, documents: List[str], doc_ids: Optional[List[str]] = None, 
                   index_type: str = 'faiss'):
        """构建索引"""
        logger.info("Building vector index...")
        
        # 编码文档
        self.embeddings = self.encode_documents(documents)
        
        # 存储文档
        self.documents = documents
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        self.doc_ids = doc_ids
        
        # 构建FAISS索引
        if index_type == 'faiss':
            dimension = self.embeddings.shape[1]
            
            # 使用内积索引（余弦相似度）
            self.index = faiss.IndexFlatIP(dimension)
            
            # 归一化向量（用于余弦相似度）
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Index built successfully. Index size: {self.index.ntotal}")
    
    def search(self, query: str, top_k: int = 1000, 
               similarity_threshold: float = 0.3) -> Tuple[List[str], List[float]]:
        """搜索相关文档"""
        # 编码查询
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # 归一化查询向量
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 过滤低相似度结果
        filtered_results = []
        filtered_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold and idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                filtered_results.append(doc_id)
                filtered_scores.append(float(score))
        
        logger.info(f"Vector search completed. Found {len(filtered_results)} relevant documents")
        
        return filtered_results, filtered_scores
    
    def batch_search(self, queries: List[str], top_k: int = 1000, 
                    similarity_threshold: float = 0.3) -> List[Tuple[List[str], List[float]]]:
        """批量搜索"""
        results = []
        
        for query in queries:
            doc_ids, scores = self.search(query, top_k, similarity_threshold)
            results.append((doc_ids, scores))
        
        return results
    
    def get_document_embeddings(self, doc_ids: List[str]) -> np.ndarray:
        """获取文档嵌入"""
        embeddings = []
        
        for doc_id in doc_ids:
            try:
                idx = self.doc_ids.index(doc_id)
                embeddings.append(self.embeddings[idx])
            except ValueError:
                logger.warning(f"Document ID {doc_id} not found in index")
        
        return np.array(embeddings) if embeddings else np.empty((0, self.embeddings.shape[1]))
    
    def compute_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """计算查询与文档的相似度（基于已建索引嵌入）"""
        # 编码查询
        query_embedding = self.encode_query(query)
        
        # 准备文档嵌入（若未编码则即时编码）
        if self.embeddings is None or len(self.documents) != len(documents):
            self.embeddings = self.encode_documents(documents)
        
        # 计算余弦相似度
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        
        similarities = np.dot(self.embeddings, query_embedding) / (doc_norms * query_norm + 1e-8)
        
        return similarities.flatten()
    
    def save_index(self, filepath: str):
        """保存索引"""
        if self.index is None:
            raise RuntimeError("Index not built yet")
        
        # 保存FAISS索引
        faiss.write_index(self.index, filepath)
        
        # 保存元数据
        metadata = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'model_name': self.model_name,
            'backend': self.backend
        }
        
        import pickle
        metadata_path = filepath + '.meta'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """加载索引"""
        # 加载FAISS索引
        self.index = faiss.read_index(filepath)
        
        # 加载元数据
        import pickle
        metadata_path = filepath + '.meta'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        self.doc_ids = metadata['doc_ids']
        self.model_name = metadata['model_name']
        self.backend = metadata['backend']
        
        logger.info(f"Index loaded from {filepath}. Index size: {self.index.ntotal}")

class HybridVectorRetriever:
    """混合向量检索器（支持多种相似度计算）"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 backend: str = 'sentence-transformers',
                 remote_endpoint: str = 'http://localhost:11434',
                 query_instruction_for_retrieval: Optional[str] = None,
                 hf_token: Optional[str] = None):
        self.retriever = VectorRetriever(model_name, device, backend, remote_endpoint, query_instruction_for_retrieval, hf_token)
        self.similarity_methods = ['cosine']
    
    def search_with_multiple_methods(self, query: str, top_k: int = 1000,
                                   methods: List[str] = None) -> Dict[str, Tuple[List[str], List[float]]]:
        """使用多种方法搜索"""
        if methods is None:
            methods = self.similarity_methods
        
        results = {}
        
        for method in methods:
            if method == 'cosine':
                doc_ids, scores = self.retriever.search(query, top_k)
                results[method] = (doc_ids, scores)
        
        return results
    
    def ensemble_search(self, query: str, top_k: int = 1000,
                       weights: Dict[str, float] = None) -> Tuple[List[str], List[float]]:
        """集成搜索"""
        if weights is None:
            weights = {'cosine': 1.0}
        
        method_results = self.search_with_multiple_methods(query, top_k)
        
        # 简单的加权平均
        doc_scores = {}
        for method, (doc_ids, scores) in method_results.items():
            weight = weights.get(method, 1.0)
            for doc_id, score in zip(doc_ids, scores):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += score * weight
        
        # 排序
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        doc_ids = [item[0] for item in sorted_results[:top_k]]
        scores = [item[1] for item in sorted_results[:top_k]]
        
        return doc_ids, scores 