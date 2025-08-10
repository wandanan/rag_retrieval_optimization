import os
import io
import time
import json
import hashlib
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入V3引擎
from advanced_zipper_engine_v3 import AdvancedZipperQueryEngineV3, ZipperV3Config, ZipperV3State

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置模型缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
logger.info(f"模型缓存目录: {CACHE_DIR}")

# 设置向量数据库存储目录
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
logger.info(f"向量数据库目录: {VECTOR_DB_DIR}")

# 获取HuggingFace token
HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
if HF_TOKEN:
    logger.info("检测到HuggingFace token，将使用认证模型")
else:
    logger.warning("未检测到HuggingFace token，将使用公开模型")

# -----------------------------
# 文本切割（父/子文档）
# -----------------------------

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须>0")
    if overlap < 0 or overlap >= chunk_size:
        overlap = 0
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks

# -----------------------------
# V3引擎索引状态管理
# -----------------------------

class V3IndexState:
    def __init__(self):
        self.documents: Dict[int, str] = {}
        self.document_hash: Optional[str] = None
        self.chunk_config_hash: Optional[str] = None
        self.v3_engine: Optional[AdvancedZipperQueryEngineV3] = None
        self.index_built: bool = False
        
        # 文档切块配置
        self.chunk_size: int = 500
        self.overlap: int = 100
        
        # V3引擎配置
        self.v3_config: ZipperV3Config = ZipperV3Config(
            encoder_backend="hf",  # 强制使用HF
            hf_model_name="BAAI/bge-small-zh-v1.5"  # 默认HF模型
        )
        
        # 会话状态管理
        self.session_states: Dict[str, ZipperV3State] = {}
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'total_retrieval_time': 0.0,
            'total_llm_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def clear(self):
        self.__init__()
    
    def get_chunk_config_hash(self) -> str:
        """获取切块配置的哈希值"""
        config_str = f"{self.chunk_size}_{self.overlap}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_document_hash(self, text: str) -> str:
        """获取文档内容的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_index_key(self, text: str) -> str:
        """获取索引的唯一标识符"""
        doc_hash = self.get_document_hash(text)
        config_hash = self.get_chunk_config_hash()
        return f"v3_{doc_hash}_{config_hash}"
    
    def get_v3_config_hash(self) -> str:
        """获取V3引擎配置的哈希值"""
        config_str = (
            f"{self.v3_config.encoder_backend}_"
            f"{self.v3_config.hf_model_name}_"
            f"{self.v3_config.bm25_weight}_"
            f"{self.v3_config.colbert_weight}_"
            f"{self.v3_config.num_heads}_"
            f"{self.v3_config.context_influence}_"
            f"{self.v3_config.final_top_k}_"
            f"{self.v3_config.use_hybrid_search}_"
            f"{self.v3_config.use_multi_head}_"
            f"{self.v3_config.use_length_penalty}_"
            f"{self.v3_config.use_stateful_reranking}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_index(self, index_key: str):
        """保存V3索引到磁盘"""
        if not self.index_built or self.v3_engine is None:
            logger.warning("V3索引未构建或引擎未初始化，无法保存")
            return
        
        try:
            logger.info(f"开始保存V3索引: {index_key}")
            
            # 保存文档数据
            docs_data = {
                'documents': self.documents,
                'document_hash': self.document_hash,
                'chunk_config_hash': self.chunk_config_hash,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'v3_config': self.v3_config,
                'index_built': True
            }
            
            docs_path = os.path.join(VECTOR_DB_DIR, f"{index_key}.pkl")
            logger.info(f"保存文档数据到: {docs_path}")
            with open(docs_path, 'wb') as f:
                pickle.dump(docs_data, f)
            
            # 保存BM25索引
            if self.v3_engine.bm25_index is not None:
                bm25_path = os.path.join(VECTOR_DB_DIR, f"{index_key}_bm25.pkl")
                logger.info(f"保存BM25索引到: {bm25_path}")
                with open(bm25_path, 'wb') as f:
                    pickle.dump({
                        'bm25_index': self.v3_engine.bm25_index,
                        'bm25_idx_to_pid': self.v3_engine.bm25_idx_to_pid
                    }, f)
            
            # 保存预计算的token嵌入（如果启用）
            if self.v3_config.precompute_doc_tokens and self.v3_engine.doc_token_embeddings:
                token_path = os.path.join(VECTOR_DB_DIR, f"{index_key}_tokens.pkl")
                logger.info(f"保存token嵌入到: {token_path}")
                with open(token_path, 'wb') as f:
                    pickle.dump(self.v3_engine.doc_token_embeddings, f)
            
            logger.info(f"V3索引保存成功: {index_key}")
            
        except Exception as e:
            logger.error(f"保存V3索引失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
    
    def load_index(self, index_key: str) -> bool:
        """从磁盘加载V3索引"""
        try:
            # 检查文档数据文件
            docs_path = os.path.join(VECTOR_DB_DIR, f"{index_key}.pkl")
            logger.info(f"检查文档数据文件: {docs_path}")
            if not os.path.exists(docs_path):
                logger.info(f"文档数据文件不存在: {docs_path}")
                return False
            
            # 加载文档数据
            logger.info("加载文档数据...")
            with open(docs_path, 'rb') as f:
                docs_data = pickle.load(f)
            
            # 恢复基本状态
            self.documents = docs_data['documents']
            self.document_hash = docs_data['document_hash']
            self.chunk_config_hash = docs_data['chunk_config_hash']
            self.chunk_size = docs_data['chunk_size']
            self.overlap = docs_data['overlap']
            self.v3_config = docs_data['v3_config']
            
            # 初始化V3引擎
            logger.info("初始化V3引擎...")
            self.v3_engine = AdvancedZipperQueryEngineV3(self.v3_config)
            
            # 加载BM25索引
            bm25_path = os.path.join(VECTOR_DB_DIR, f"{index_key}_bm25.pkl")
            if os.path.exists(bm25_path):
                logger.info("加载BM25索引...")
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.v3_engine.bm25_index = bm25_data['bm25_index']
                    self.v3_engine.bm25_idx_to_pid = bm25_data['bm25_idx_to_pid']
            
            # 加载预计算的token嵌入
            token_path = os.path.join(VECTOR_DB_DIR, f"{index_key}_tokens.pkl")
            if os.path.exists(token_path):
                logger.info("加载token嵌入...")
                with open(token_path, 'rb') as f:
                    self.v3_engine.doc_token_embeddings = pickle.load(f)
            else:
                # 如果没有预计算的token嵌入，重新构建
                logger.info("重新构建token嵌入...")
                self.v3_engine.documents = self.documents
                self.v3_engine.build_document_index(self.documents)
            
            self.index_built = True
            logger.info(f"V3索引已加载: {index_key}")
            return True
            
        except Exception as e:
            logger.error(f"加载V3索引失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def get_or_create_session_state(self, session_id: str) -> ZipperV3State:
        """获取或创建会话状态"""
        if session_id not in self.session_states:
            # 创建新的会话状态
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            context_vector = torch.zeros(self.v3_config.embedding_dim, device=device)
            self.session_states[session_id] = ZipperV3State(
                original_query="",
                context_vector=context_vector
            )
        return self.session_states[session_id]
    
    def update_session_state(self, session_id: str, query: str, results: List[Tuple[int, float, str]]):
        """更新会话状态"""
        if self.v3_engine and self.v3_config.use_stateful_reranking:
            session_state = self.get_or_create_session_state(session_id)
            session_state.original_query = query
            self.v3_engine.update_state(session_state, results)
            self.session_states[session_id] = session_state

# 全局V3状态
v3_state = V3IndexState()

# -----------------------------
# FastAPI 应用
# -----------------------------

app = FastAPI(title="V3 RAG 引擎测试系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源
static_dir = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(static_dir, exist_ok=True)
app.mount("/web", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
async def root_page():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)

# -----------------------------
# API 端点
# -----------------------------

@app.get("/api/health")
async def health():
    v3_features = {}
    if v3_state.v3_config:
        v3_features = {
            "hybrid_search": v3_state.v3_config.use_hybrid_search,
            "multi_head": v3_state.v3_config.use_multi_head,
            "stateful_reranking": v3_state.v3_config.use_stateful_reranking,
            "length_penalty": v3_state.v3_config.use_length_penalty,
            "precompute_doc_tokens": v3_state.v3_config.precompute_doc_tokens,
            "enable_amp_if_beneficial": v3_state.v3_config.enable_amp_if_beneficial,
            "encoder_backend": v3_state.v3_config.encoder_backend,
            "bm25_weight": v3_state.v3_config.bm25_weight,
            "colbert_weight": v3_state.v3_config.colbert_weight,
            "num_heads": v3_state.v3_config.num_heads,
            "context_influence": v3_state.v3_config.context_influence,
            "length_penalty_alpha": v3_state.v3_config.length_penalty_alpha,
            "context_memory_decay": v3_state.v3_config.context_memory_decay,
            "bm25_top_n": v3_state.v3_config.bm25_top_n,
            "encode_batch_size": v3_state.v3_config.encode_batch_size,
            "max_length": v3_state.v3_config.max_length
        }
    
    return {
        "status": "ok", 
        "index_built": v3_state.index_built,
        "engine_type": "V3 Advanced Zipper Engine",
        "features": v3_features,
        "session_count": len(v3_state.session_states),
        "total_queries": v3_state.stats.get('total_queries', 0),
        "avg_retrieval_time": v3_state.stats.get('total_retrieval_time', 0) / max(v3_state.stats.get('total_queries', 1), 1)
    }

@app.post("/api/clear")
async def clear():
    v3_state.clear()
    return {"status": "cleared", "message": "V3引擎状态已清空"}

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(100),
    v3_config: str = Form("{}")
):
    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {e}")

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="空文件")

    # 更新配置
    v3_state.chunk_size = chunk_size
    v3_state.overlap = overlap

    # 处理V3引擎配置
    try:
        logger.info(f"原始v3_config参数: {v3_config}")
        v3_config_data = json.loads(v3_config) if v3_config else {}
        logger.info(f"解析后的V3配置: {v3_config_data}")
        logger.info(f"配置类型: {type(v3_config_data)}")
        
        # 更新V3配置
        if v3_config_data:
            v3_state.v3_config = ZipperV3Config(
                encoder_backend=v3_config_data.get("encoder_backend", "hf"),
                hf_model_name=v3_config_data.get("hf_model_name", "BAAI/bge-small-zh-v1.5"),
                embedding_dim=int(v3_config_data.get("embedding_dim", 512)),
                bm25_weight=float(v3_config_data.get("bm25_weight", 1.0)),
                colbert_weight=float(v3_config_data.get("colbert_weight", 1.5)),
                num_heads=int(v3_config_data.get("num_heads", 8)),
                context_influence=float(v3_config_data.get("context_influence", 0.3)),
                final_top_k=int(v3_config_data.get("final_top_k", 10)),
                length_penalty_alpha=float(v3_config_data.get("length_penalty_alpha", 0.05)),
                context_memory_decay=float(v3_config_data.get("context_memory_decay", 0.8)),
                bm25_top_n=int(v3_config_data.get("bm25_top_n", 100)),
                encode_batch_size=int(v3_config_data.get("encode_batch_size", 64)),
                max_length=int(v3_config_data.get("max_length", 256)),
                use_hybrid_search=bool(v3_config_data.get("use_hybrid_search", True)),
                use_multi_head=bool(v3_config_data.get("use_multi_head", True)),
                use_length_penalty=bool(v3_config_data.get("use_length_penalty", True)),
                use_stateful_reranking=bool(v3_config_data.get("use_stateful_reranking", True)),
                precompute_doc_tokens=bool(v3_config_data.get("precompute_doc_tokens", False)),
                enable_amp_if_beneficial=bool(v3_config_data.get("enable_amp_if_beneficial", True)),
                use_reranker=bool(v3_config_data.get("use_reranker", True)),
                reranker_model_name=v3_config_data.get("reranker_model_name", "BAAI/bge-reranker-large"),
                reranker_top_n=int(v3_config_data.get("reranker_top_n", 50)),
                reranker_weight=float(v3_config_data.get("reranker_weight", 1.5)),
                reranker_backend=v3_config_data.get("reranker_backend", "auto")
            )
            logger.info(f"V3配置已更新，模型名称: {v3_state.v3_config.hf_model_name}")
    except Exception as e:
        logger.warning(f"V3配置解析失败，使用默认配置: {e}")

    # 计算文档和配置哈希
    doc_hash = v3_state.get_document_hash(text)
    config_hash = v3_state.get_chunk_config_hash()
    index_key = v3_state.get_index_key(text)
    
    logger.info(f"文档哈希: {doc_hash}")
    logger.info(f"配置哈希: {config_hash}")
    logger.info(f"索引键: {index_key}")
    
    # 检查是否已有缓存的索引
    logger.info(f"检查缓存索引: {index_key}")
    if v3_state.load_index(index_key):
        logger.info(f"使用缓存的索引: {index_key}")
        v3_state.stats['cache_hits'] += 1
        return {
            "message": "使用缓存的V3索引",
            "num_docs": len(v3_state.documents),
            "cached": True,
            "index_key": index_key
        }

    logger.info(f"构建新V3索引: {index_key}")
    v3_state.stats['cache_misses'] += 1
    
    # 切割文档
    chunks = split_text(text, chunk_size, overlap)
    doc_ids = list(range(len(chunks)))
    documents = {doc_id: chunk for doc_id, chunk in zip(doc_ids, chunks)}
    
    # 构建V3引擎索引
    v3_state.documents = documents
    v3_state.document_hash = doc_hash
    v3_state.chunk_config_hash = config_hash
    
    try:
        # 初始化V3引擎并构建索引
        logger.info("初始化V3引擎...")
        v3_state.v3_engine = AdvancedZipperQueryEngineV3(v3_state.v3_config)
        logger.info("构建文档索引...")
        v3_state.v3_engine.build_document_index(documents)
        v3_state.index_built = True
        
        # 保存索引到磁盘
        logger.info("保存索引到磁盘...")
        v3_state.save_index(index_key)

        return {
            "message": "V3索引已构建并保存",
            "num_docs": len(documents),
            "cached": False,
            "index_key": index_key
        }
    except Exception as e:
        logger.error(f"V3引擎初始化失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"V3引擎初始化失败: {str(e)}")

@app.post("/api/v3_query")
async def v3_query(payload: Dict[str, Any]):
    """V3引擎查询接口"""
    if not v3_state.index_built:
        raise HTTPException(status_code=400, detail="尚未上传并构建V3索引")

    question: str = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="缺少问题")

    # 获取V3引擎配置
    v3_config_data = payload.get("v3_config", {})
    
    # 检查是否需要重新初始化引擎（配置变更）
    current_config_hash = v3_state.get_v3_config_hash()
    new_config = ZipperV3Config(
        encoder_backend="hf",  # 强制使用HF
        hf_model_name=v3_config_data.get("hf_model_name") or "BAAI/bge-small-zh-v1.5",
        embedding_dim=int(v3_config_data.get("embedding_dim", 512)),
        bm25_weight=float(v3_config_data.get("bm25_weight", 1.0)),
        colbert_weight=float(v3_config_data.get("colbert_weight", 1.5)),
        num_heads=int(v3_config_data.get("num_heads", 8)),
        context_influence=float(v3_config_data.get("context_influence", 0.3)),
        final_top_k=int(v3_config_data.get("final_top_k", 10)),
        # 新增的高级配置项
        length_penalty_alpha=float(v3_config_data.get("length_penalty_alpha", 0.05)),
        context_memory_decay=float(v3_config_data.get("context_memory_decay", 0.8)),
        bm25_top_n=int(v3_config_data.get("bm25_top_n", 100)),
        encode_batch_size=int(v3_config_data.get("encode_batch_size", 64)),
        max_length=int(v3_config_data.get("max_length", 256)),
        # 功能开关
        use_hybrid_search=bool(v3_config_data.get("use_hybrid_search", True)),
        use_multi_head=bool(v3_config_data.get("use_multi_head", True)),
        use_length_penalty=bool(v3_config_data.get("use_length_penalty", True)),
        use_stateful_reranking=bool(v3_config_data.get("use_stateful_reranking", True)),
        precompute_doc_tokens=bool(v3_config_data.get("precompute_doc_tokens", False)),
        enable_amp_if_beneficial=bool(v3_config_data.get("enable_amp_if_beneficial", True)),
        # 新增重排序配置
        use_reranker=bool(v3_config_data.get("use_reranker", True)),
        reranker_model_name=v3_config_data.get("reranker_model_name", "BAAI/bge-reranker-large"),
        reranker_top_n=int(v3_config_data.get("reranker_top_n", 50)),
        reranker_weight=float(v3_config_data.get("reranker_weight", 1.5)),
        reranker_backend=v3_config_data.get("reranker_backend", "auto")
    )
    
    new_config_hash = hashlib.md5(
        f"{new_config.encoder_backend}_{new_config.hf_model_name}_"
        f"{new_config.embedding_dim}_{new_config.bm25_weight}_{new_config.colbert_weight}_{new_config.num_heads}_"
        f"{new_config.context_influence}_{new_config.final_top_k}_"
        f"{new_config.length_penalty_alpha}_{new_config.context_memory_decay}_"
        f"{new_config.bm25_top_n}_{new_config.encode_batch_size}_{new_config.max_length}_"
        f"{new_config.use_hybrid_search}_{new_config.use_multi_head}_"
        f"{new_config.use_length_penalty}_{new_config.use_stateful_reranking}_"
        f"{new_config.precompute_doc_tokens}_{new_config.enable_amp_if_beneficial}_"
        f"{new_config.use_reranker}_{new_config.reranker_model_name}_{new_config.reranker_top_n}_{new_config.reranker_weight}_{new_config.reranker_backend}".encode()
    ).hexdigest()
    
    # 如果配置变更，重新初始化引擎
    if current_config_hash != new_config_hash:
        logger.info("V3引擎配置变更，重新初始化...")
        v3_state.v3_config = new_config
        v3_state.v3_engine = AdvancedZipperQueryEngineV3(new_config)
        v3_state.v3_engine.build_document_index(v3_state.documents)
        # 清空会话状态
        v3_state.session_states.clear()
    
    # 获取或创建会话状态
    session_id = payload.get("session_id", "default")
    session_state = v3_state.get_or_create_session_state(session_id)
    
    # 执行V3检索
    start_time = time.time()
    try:
        results = v3_state.v3_engine.retrieve(question, state=session_state)
        retrieval_time = time.time() - start_time
        
        # 更新会话状态
        v3_state.update_session_state(session_id, question, results)
        
        # 计算检索指标
        metrics = {
            "retrieval_time": retrieval_time,
            "num_candidates": len(results),
            "top_score": results[0][1] if results else 0.0,
            "score_range": {
                "min": min([r[1] for r in results]) if results else 0.0,
                "max": max([r[1] for r in results]) if results else 0.0
            },
            "engine_config": {
                "hybrid_search": v3_state.v3_config.use_hybrid_search,
                "multi_head": v3_state.v3_config.use_multi_head,
                "stateful_reranking": v3_state.v3_config.use_stateful_reranking,
                "length_penalty": v3_state.v3_config.use_length_penalty,
                "bm25_weight": v3_state.v3_config.bm25_weight,
                "colbert_weight": v3_state.v3_config.colbert_weight,
                "embedding_dim": v3_state.v3_config.embedding_dim,
                "num_heads": v3_state.v3_config.num_heads,
                "context_influence": v3_state.v3_config.context_influence,
                "length_penalty_alpha": v3_state.v3_config.length_penalty_alpha,
                "context_memory_decay": v3_state.v3_config.context_memory_decay,
                "bm25_top_n": v3_state.v3_config.bm25_top_n,
                "encode_batch_size": v3_state.v3_config.encode_batch_size,
                "max_length": v3_state.v3_config.max_length,
                "precompute_doc_tokens": v3_state.v3_config.precompute_doc_tokens,
                "enable_amp_if_beneficial": v3_state.v3_config.enable_amp_if_beneficial,
                "use_reranker": v3_state.v3_config.use_reranker,
                "reranker_model_name": v3_state.v3_config.reranker_model_name,
                "reranker_top_n": v3_state.v3_config.reranker_top_n,
                "reranker_weight": v3_state.v3_config.reranker_weight,
                "reranker_backend": v3_state.v3_config.reranker_backend
            }
        }
        
        # 更新统计信息
        v3_state.stats['total_queries'] += 1
        v3_state.stats['total_retrieval_time'] += retrieval_time
        
    except Exception as e:
        logger.error(f"V3检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"V3检索失败: {e}")

    # 调用LLM生成答案
    llm_cfg = payload.get("llm", {})
    base_url: str = llm_cfg.get("base_url", "").strip()
    api_key: str = llm_cfg.get("api_key", "").strip()
    model: str = llm_cfg.get("model", "gpt-3.5-turbo").strip()
    rag_prompt: str = payload.get("prompt", "").strip()

    answer = ""
    llm_latency = 0.0
    
    if base_url and api_key:
        try:
            import requests
            
            # 组织上下文
            contexts = []
            for i, (doc_id, score, content) in enumerate(results):
                contexts.append({
                    "doc_id": doc_id,
                    "score": score,
                    "content": content[:500] + "..." if len(content) > 500 else content
                })
            
            # 构建LLM请求
            prompt = rag_prompt.strip() or (
                "你是一个严谨的中文助手。请严格基于给定的参考上下文回答用户问题：\n"
                "- 如果上下文无法支持答案，必须直接回答：‘无法根据参考上下文回答。’\n"
                "- 禁止编造或引入上下文以外的信息。\n"
                "- 回答尽量精炼。"
            )

            system_msg = {"role": "system", "content": prompt}
            context_text = "\n\n".join([f"[片段{i+1} - 分数:{c['score']:.3f}]\n{c['content']}" for i, c in enumerate(contexts)])
            user_msg = {
                "role": "user", 
                "content": f"参考上下文如下：\n{context_text}\n\n用户问题：{question}\n请仅依据参考上下文作答。若无法回答，请直接回复：无法根据参考上下文回答。"
            }

            llm_request = {
                "model": model,
                "messages": [system_msg, user_msg],
                "temperature": 0.2,
                "stream": False
            }

            t0 = time.time()
            url = base_url
            headers = {
                "Authorization": f"Bearer {api_key}", 
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity"
            }
            
            resp = requests.post(url, headers=headers, json=llm_request, timeout=120, stream=False)
            resp.raise_for_status()
            
            try:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    answer = data["choices"][0]["message"]["content"].strip()
                else:
                    answer = f"[LLM响应格式错误] {data}"
            except json.JSONDecodeError as e:
                answer = f"[LLM响应解析失败] {e}, 响应内容: {resp.text[:200]}"
                
            llm_latency = time.time() - t0
            v3_state.stats['total_llm_time'] += llm_latency
            
        except requests.exceptions.RequestException as e:
            answer = f"[LLM网络请求失败] {e}"
        except Exception as e:
            answer = f"[LLM调用失败] {e}"
    else:
        answer = "[未配置LLM] 已返回V3引擎检索结果。"

    # 更新指标
    metrics['llm_latency'] = llm_latency
    metrics['total_latency'] = retrieval_time + llm_latency

    return {
        "answer": answer,
        "results": [
            {
                "doc_id": doc_id,
                "score": float(score),
                "content": content
            }
            for doc_id, score, content in results
        ],
        "metrics": metrics,
        "session_id": session_id
    }

@app.get("/api/v3/stats")
async def get_v3_stats():
    """获取V3引擎统计信息"""
    if not v3_state.index_built:
        return {"error": "索引未构建"}
    
    stats = v3_state.stats.copy()
    if stats['total_queries'] > 0:
        stats['avg_retrieval_time'] = stats['total_retrieval_time'] / stats['total_queries']
        stats['avg_llm_time'] = stats['total_llm_time'] / stats['total_queries']
        stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
    
    return {
        "engine_stats": stats,
        "index_info": {
            "num_documents": len(v3_state.documents),
            "document_hash": v3_state.document_hash,
            "chunk_config_hash": v3_state.chunk_config_hash,
            "chunk_size": v3_state.chunk_size,
            "overlap": v3_state.overlap
        },
        "engine_config": {
            "encoder_backend": v3_state.v3_config.encoder_backend,
            "use_hybrid_search": v3_state.v3_config.use_hybrid_search,
            "use_multi_head": v3_state.v3_config.use_multi_head,
            "use_stateful_reranking": v3_state.v3_config.use_stateful_reranking,
            "use_length_penalty": v3_state.v3_config.use_length_penalty,
            "bm25_weight": v3_state.v3_config.bm25_weight,
            "colbert_weight": v3_state.v3_config.colbert_weight,
            "num_heads": v3_state.v3_config.num_heads,
            "context_influence": v3_state.v3_config.context_influence,
            "final_top_k": v3_state.v3_config.final_top_k
        },
        "session_info": {
            "active_sessions": len(v3_state.session_states),
            "session_ids": list(v3_state.session_states.keys())
        }
    }

@app.post("/api/v3/clear_sessions")
async def clear_v3_sessions():
    """清空所有会话状态"""
    v3_state.session_states.clear()
    return {"status": "sessions_cleared", "message": "所有会话状态已清空"}

@app.get("/api/debug/cache")
async def debug_cache():
    """调试缓存状态"""
    cache_files = []
    if os.path.exists(VECTOR_DB_DIR):
        for file in os.listdir(VECTOR_DB_DIR):
            cache_files.append(file)
    
    return {
        "cache_dir": VECTOR_DB_DIR,
        "cache_files": cache_files,
        "index_built": v3_state.index_built,
        "document_hash": v3_state.document_hash,
        "chunk_config_hash": v3_state.chunk_config_hash,
        "num_docs": len(v3_state.documents),
        "engine_type": "V3 Advanced Zipper Engine"
    }

@app.post("/api/debug/save")
async def debug_save():
    """手动保存当前V3索引"""
    if not v3_state.index_built:
        return {"error": "V3索引未构建"}
    
    if not v3_state.document_hash:
        return {"error": "文档哈希未设置"}
    
    index_key = v3_state.get_index_key("")
    v3_state.save_index(index_key)
    
    return {"message": f"V3索引已保存: {index_key}"}

# -----------------------------
# 批量测试接口
# -----------------------------

@app.post("/api/batch_test")
async def batch_test(
    test_file: UploadFile = File(...),
    v3_config: str = Form("{}"),
    llm_config: str = Form("{}"),
    prompt: str = Form(""),
    include_contexts: bool = Form(False)
):
    """批量测试接口"""
    if not v3_state.index_built:
        raise HTTPException(status_code=400, detail="尚未上传并构建V3索引")
    
    try:
        # 解析配置
        v3_config_data = json.loads(v3_config) if v3_config else {}
        llm_config_data = json.loads(llm_config) if llm_config else {}
        
        # 读取测试文件
        test_content = await test_file.read()
        test_data = json.loads(test_content.decode('utf-8'))
        
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(test_file.filename)[0]
        
        # 执行批量测试
        results = []
        total_queries = len(test_data)
        
        for i, test_case in enumerate(test_data):
            logger.info(f"执行测试用例 {i+1}/{total_queries}: {test_case.get('id', 'Unknown')}")
            
            # 执行查询
            query = test_case.get('query', '')
            if not query:
                continue
                
            try:
                # 构建查询payload
                payload = {
                    "question": query,
                    "v3_config": v3_config_data,
                    "llm": llm_config_data,
                    "prompt": prompt,
                    "session_id": f"batch_test_{i}"
                }
                
                # 调用V3查询接口
                query_result = await v3_query(payload)
                
                # 构建测试结果
                test_result = {
                    "id": test_case.get('id', f"test_{i+1}"),
                    "category": test_case.get('category', ''),
                    "difficulty": test_case.get('difficulty', 1),
                    "query": query,
                    "expected_answer_keywords": test_case.get('expected_answer_keywords', []),
                    "ai_answer": query_result.get('answer', ''),
                    "metrics": query_result.get('metrics', {}),
                    "session_id": query_result.get('session_id', ''),
                    "timestamp": time.time()
                }
                
                # 根据用户选择决定是否包含召回调段
                if include_contexts:
                    test_result["retrieval_results"] = query_result.get('results', [])
                else:
                    # 不包含召回调段，只保存基本信息
                    test_result["retrieval_results"] = []
                    # 移除详细的检索信息，只保留基本指标
                    if "metrics" in test_result:
                        basic_metrics = {}
                        for key in ["total_time", "retrieval_time", "generation_time"]:
                            if key in test_result["metrics"]:
                                basic_metrics[key] = test_result["metrics"][key]
                        test_result["metrics"] = basic_metrics
                
                results.append(test_result)
                
                # 添加延迟避免过快请求
                if i < total_queries - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"测试用例 {test_case.get('id', 'Unknown')} 执行失败: {e}")
                test_result = {
                    "id": test_case.get('id', f"test_{i+1}"),
                    "category": test_case.get('category', ''),
                    "difficulty": test_case.get('difficulty', 1),
                    "query": query,
                    "expected_answer_keywords": test_case.get('expected_answer_keywords', []),
                    "ai_answer": f"[执行失败] {e}",
                    "retrieval_results": [],
                    "metrics": {},
                    "session_id": "",
                    "timestamp": time.time(),
                    "error": str(e)
                }
                results.append(test_result)
        
        # 保存测试结果
        result_filename = f"{filename}_jg.json"
        result_path = os.path.join("results", result_filename)
        os.makedirs("results", exist_ok=True)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成测试摘要
        summary = {
            "total_tests": total_queries,
            "successful_tests": len([r for r in results if 'error' not in r]),
            "failed_tests": len([r for r in results if 'error' in r]),
            "average_retrieval_time": sum([r.get('metrics', {}).get('retrieval_time', 0) for r in results if 'error' not in r]) / max(1, len([r for r in results if 'error' not in r])),
            "average_llm_latency": sum([r.get('metrics', {}).get('llm_latency', 0) for r in results if 'error' not in r]) / max(1, len([r for r in results if 'error' not in r])),
            "test_timestamp": time.time(),
            "result_file": result_filename,
            "include_contexts": include_contexts,
            "contexts_info": "包含召回调段" if include_contexts else "仅包含AI回答，不包含召回调段"
        }
        
        # 计算召回成功统计率
        successful_results = [r for r in results if 'error' not in r]
        recall_failures = 0
        
        for result in successful_results:
            ai_answer = result.get('ai_answer', '').lower()
            # 检查是否包含召回失败的关键字
            if '无法根据参考上下文回答' in ai_answer or '无法' in ai_answer:
                recall_failures += 1
        
        total_successful = len(successful_results)
        recall_success_count = total_successful - recall_failures
        recall_success_rate = (recall_success_count / total_successful * 100) if total_successful > 0 else 0
        
        # 更新摘要信息
        summary.update({
            "recall_success_count": recall_success_count,
            "recall_failure_count": recall_failures,
            "recall_success_rate": round(recall_success_rate, 2),
            "total_successful_tests": total_successful
        })
        
        # 保存摘要
        summary_filename = f"{filename}_jg_summary.json"
        summary_path = os.path.join("results", summary_filename)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": f"批量测试完成，共执行 {total_queries} 个测试用例",
            "summary": summary,
            "result_file": result_filename,
            "download_url": f"/api/download_result/{result_filename}"
        }
        
    except Exception as e:
        logger.error(f"批量测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量测试失败: {e}")

@app.get("/api/download_result/{filename}")
async def download_result(filename: str):
    """下载测试结果文件"""
    file_path = os.path.join("results", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/json'
    )

@app.get("/api/list_results")
async def list_results():
    """列出所有可下载的测试结果文件"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return {"files": []}
    
    files = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_jg.json') or filename.endswith('_jg_summary.json'):
            file_path = os.path.join(results_dir, filename)
            file_stat = os.stat(file_path)
            files.append({
                "filename": filename,
                "size": file_stat.st_size,
                "modified": file_stat.st_mtime,
                "download_url": f"/api/download_result/{filename}"
            })
    
    # 按修改时间排序
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    return {"files": files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 