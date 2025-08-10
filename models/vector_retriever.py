"""
向量检索模块
"""

import os
import pickle
import logging
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import time

# 依赖检查
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorRetriever:
    """向量检索引擎 - 使用HuggingFace transformers或Ollama"""
    
    def __init__(self, model_name: str = 'BAAI/bge-small-zh-v1.5', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 backend: str = 'transformers',  # 改为transformers
                 remote_endpoint: str = 'http://localhost:11434',
                 query_instruction_for_retrieval: Optional[str] = None,
                 hf_token: Optional[str] = None):
        """
        初始化向量检索引擎
        
        Args:
            model_name: 模型名称（HF模型名或Ollama模型名）
            device: 计算设备
            backend: 后端类型 ('transformers' 或 'ollama')
            remote_endpoint: Ollama服务端点
            query_instruction_for_retrieval: BGE查询指令
            hf_token: HuggingFace认证token
        """
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.remote_endpoint = remote_endpoint
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.hf_token = hf_token
        
        # 设置缓存目录
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.dirname(__file__), '..', 'model_cache'))
        os.makedirs(cache_dir, exist_ok=True)
        
        self.encoder = None
        if self.backend == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers未安装。请运行: pip install transformers")
            
            try:
                logger.info(f"加载HF模型: {model_name} 到缓存目录: {cache_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    token=hf_token,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    token=hf_token,
                    trust_remote_code=True,
                    device_map=device if device != 'cpu' else None
                )
                if device == 'cpu':
                    self.model = self.model.to(device)
                self.model.eval()
                logger.info(f"HF模型加载成功: {model_name}")
            except Exception as e:
                logger.error(f"加载HF模型失败: {e}")
                raise
                
        elif self.backend == 'ollama':
            if not REQUESTS_AVAILABLE:
                raise RuntimeError("requests 未安装，无法使用 Ollama 嵌入后端")
        else:
            raise ValueError(f"未知后端: {self.backend}")
        
        # FAISS索引
        if not FAISS_AVAILABLE:
            raise ImportError("faiss未安装。请运行: pip install faiss-cpu 或 pip install faiss-gpu")
            
        self.index = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Vector retriever initialized with backend={self.backend}, model={model_name}")

    def _encode_transformers(self, texts: List[str]) -> np.ndarray:
        """使用transformers库生成嵌入"""
        if self.backend != 'transformers':
            raise ValueError("当前后端不是transformers")
            
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 对BGE模型添加查询指令
                if self.query_instruction_for_retrieval:
                    batch_texts = [f"{self.query_instruction_for_retrieval}{text}" for text in batch_texts]
                
                # 编码
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # 平均池化
                attention_mask = inputs['attention_mask']
                embeddings_batch = self._mean_pool(outputs.last_hidden_state, attention_mask)
                
                # L2归一化
                embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                
                embeddings.append(embeddings_batch.cpu().numpy())
        
        return np.vstack(embeddings).astype(np.float32)
    
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """平均池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
        if self.backend == 'transformers':
            return self._encode_transformers(texts)
        elif self.backend == 'ollama':
            return self._encode_ollama(texts).astype(np.float32)
        else:
            raise ValueError("不支持的编码后端")
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """编码文档"""
        logger.info(f"Encoding {len(documents)} documents via {self.backend}...")
        if self.backend == 'transformers':
            return self._encode_transformers(documents)
        elif self.backend == 'ollama':
            embeddings = self._encode_ollama(documents)
            return embeddings.astype(np.float32)
        else:
            raise ValueError("不支持的编码后端")
    
    def encode_query(self, query: str) -> np.ndarray:
        """编码查询（对BGE模型自动添加检索指令）"""
        if self.backend == 'transformers':
            if self.query_instruction_for_retrieval:
                query = f"{self.query_instruction_for_retrieval}{query}"
            embedding = self._encode_transformers([query])
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
    
    def __init__(self, model_name: str = 'BAAI/bge-small-zh-v1.5',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 backend: str = 'transformers',  # 改为transformers
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