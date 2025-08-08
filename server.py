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

from models.vector_retriever import VectorRetriever
from final_demo import FinalAttentionRetriever

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
# 索引状态管理
# -----------------------------

class IndexState:
    def __init__(self):
        self.parent_docs: List[str] = []
        self.sub_docs: List[str] = []
        self.sub_to_parent: List[int] = []
        self.parent_ids: List[str] = []
        self.sub_ids: List[str] = []
        self.sub_id_to_idx: Dict[str, int] = {}

        self.vector: Optional[VectorRetriever] = None
        self.attn: Optional[FinalAttentionRetriever] = None
        self.index_built: bool = False

        # 配置
        self.parent_chunk_size: int = 1000
        self.parent_overlap: int = 200
        self.sub_chunk_size: int = 200
        self.sub_overlap: int = 50
        
        # 文档哈希（用于检查是否需要重新构建索引）
        self.document_hash: Optional[str] = None
        self.chunk_config_hash: Optional[str] = None

    def clear(self):
        self.__init__()
    
    def get_chunk_config_hash(self) -> str:
        """获取切块配置的哈希值"""
        config_str = f"{self.parent_chunk_size}_{self.parent_overlap}_{self.sub_chunk_size}_{self.sub_overlap}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_document_hash(self, text: str) -> str:
        """获取文档内容的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_index_key(self, text: str) -> str:
        """获取索引的唯一标识符"""
        doc_hash = self.get_document_hash(text)
        config_hash = self.get_chunk_config_hash()
        return f"{doc_hash}_{config_hash}"
    
    def save_index(self, index_key: str):
        """保存索引到磁盘"""
        if not self.index_built or self.vector is None:
            return
        
        try:
            # 保存向量索引
            vector_path = os.path.join(VECTOR_DB_DIR, f"{index_key}.faiss")
            self.vector.save_index(vector_path)
            
            # 保存文档状态
            state_data = {
                'parent_docs': self.parent_docs,
                'sub_docs': self.sub_docs,
                'sub_to_parent': self.sub_to_parent,
                'parent_ids': self.parent_ids,
                'sub_ids': self.sub_ids,
                'sub_id_to_idx': self.sub_id_to_idx,
                'parent_chunk_size': self.parent_chunk_size,
                'parent_overlap': self.parent_overlap,
                'sub_chunk_size': self.sub_chunk_size,
                'sub_overlap': self.sub_overlap,
                'document_hash': self.document_hash,
                'chunk_config_hash': self.chunk_config_hash,
                'index_built': True
            }
            
            state_path = os.path.join(VECTOR_DB_DIR, f"{index_key}.pkl")
            with open(state_path, 'wb') as f:
                pickle.dump(state_data, f)
            
            logger.info(f"索引已保存: {index_key}")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def load_index(self, index_key: str) -> bool:
        """从磁盘加载索引"""
        try:
            # 加载向量索引
            vector_path = os.path.join(VECTOR_DB_DIR, f"{index_key}.faiss")
            if not os.path.exists(vector_path):
                return False
            
            # 确保向量检索器已初始化
            _ensure_embedder()
            self.vector.load_index(vector_path)
            
            # 加载文档状态
            state_path = os.path.join(VECTOR_DB_DIR, f"{index_key}.pkl")
            if not os.path.exists(state_path):
                return False
            
            with open(state_path, 'rb') as f:
                state_data = pickle.load(f)
            
            # 恢复状态
            self.parent_docs = state_data['parent_docs']
            self.sub_docs = state_data['sub_docs']
            self.sub_to_parent = state_data['sub_to_parent']
            self.parent_ids = state_data['parent_ids']
            self.sub_ids = state_data['sub_ids']
            self.sub_id_to_idx = state_data['sub_id_to_idx']
            self.parent_chunk_size = state_data['parent_chunk_size']
            self.parent_overlap = state_data['parent_overlap']
            self.sub_chunk_size = state_data['sub_chunk_size']
            self.sub_overlap = state_data['sub_overlap']
            self.document_hash = state_data['document_hash']
            self.chunk_config_hash = state_data['chunk_config_hash']
            self.index_built = True
            
            # 预计算父文档token嵌入
            if self.attn:
                self.attn.precompute_document_semantic_embeddings(self.parent_docs)
            
            logger.info(f"索引已加载: {index_key}")
            return True
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

state = IndexState()

# -----------------------------
# FastAPI 应用
# -----------------------------

app = FastAPI(title="RAG对比演示：注意力RAG vs 向量RAG")
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
# 工具函数
# -----------------------------

def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _ensure_embedder():
    if state.vector is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"初始化向量检索器，设备: {device}")
        
        # 根据是否有token选择不同的模型
        if HF_TOKEN:
            # 有token时优先使用BGE中文模型
            model_options = [
                'BAAI/bge-large-zh-v1.5',   # 中文BGE大模型
                'BAAI/bge-base-zh-v1.5',    # 中文BGE基础模型
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 多语言模型
            ]
        else:
            # 无token时使用公开模型
            model_options = [
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 多语言模型
                'sentence-transformers/all-MiniLM-L6-v2',  # 英文模型作为备选
            ]
        
        for model_name in model_options:
            try:
                logger.info(f"尝试加载模型: {model_name}")
                state.vector = VectorRetriever(
                    model_name=model_name,
                    device=device,
                    backend='sentence-transformers',
                    hf_token=HF_TOKEN  # 传入token
                )
                logger.info(f"成功加载模型: {model_name}")
                break
            except Exception as e:
                logger.warning(f"模型 {model_name} 加载失败: {e}")
                continue
        
        if state.vector is None:
            raise RuntimeError("所有模型都无法加载，请检查网络连接或模型缓存")
    
    if state.attn is None:
        state.attn = FinalAttentionRetriever()
        if state.vector:
            state.attn.set_semantic_embedder(state.vector)

# -----------------------------
# API 端点
# -----------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "index_built": state.index_built}

@app.post("/api/clear")
async def clear():
    state.clear()
    return {"status": "cleared"}

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    parent_chunk_size: int = Form(1000),
    parent_overlap: int = Form(200),
    sub_chunk_size: int = Form(200),
    sub_overlap: int = Form(50)
):
    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {e}")

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="空文件")

    # 更新配置
    state.parent_chunk_size = parent_chunk_size
    state.parent_overlap = parent_overlap
    state.sub_chunk_size = sub_chunk_size
    state.sub_overlap = sub_overlap

    # 计算文档和配置哈希
    doc_hash = state.get_document_hash(text)
    config_hash = state.get_chunk_config_hash()
    index_key = state.get_index_key(text)
    
    # 检查是否已有缓存的索引
    if state.load_index(index_key):
        logger.info(f"使用缓存的索引: {index_key}")
        return {
            "message": "使用缓存的索引",
            "num_parents": len(state.parent_docs),
            "num_subs": len(state.sub_docs),
            "cached": True
        }

    logger.info(f"构建新索引: {index_key}")
    
    # 切割父文档
    parent_docs = split_text(text, parent_chunk_size, parent_overlap)
    parent_ids = [f"parent_{i}" for i in range(len(parent_docs))]

    # 子文档：基于每个父文档二次切割
    sub_docs: List[str] = []
    sub_to_parent: List[int] = []
    for pid, ptext in enumerate(parent_docs):
        parts = split_text(ptext, sub_chunk_size, sub_overlap)
        sub_docs.extend(parts)
        sub_to_parent.extend([pid] * len(parts))

    sub_ids = [f"sub_{i}" for i in range(len(sub_docs))]
    # 建立反向索引，便于查询阶段 O(1) 查找
    sub_id_to_idx = {sid: i for i, sid in enumerate(sub_ids)}

    # 构建向量索引（对子文档）
    _ensure_embedder()
    state.vector.build_index(sub_docs, doc_ids=sub_ids)

    # 预计算父文档token嵌入供注意力轻量机制使用
    state.attn.precompute_document_semantic_embeddings(parent_docs)

    # 保存状态
    state.parent_docs = parent_docs
    state.parent_ids = parent_ids
    state.sub_docs = sub_docs
    state.sub_ids = sub_ids
    state.sub_id_to_idx = sub_id_to_idx
    state.sub_to_parent = sub_to_parent
    state.document_hash = doc_hash
    state.chunk_config_hash = config_hash
    state.index_built = True

    # 保存索引到磁盘
    state.save_index(index_key)

    return {
        "message": "索引已构建并保存",
        "num_parents": len(parent_docs),
        "num_subs": len(sub_docs),
        "cached": False
    }

@app.post("/api/query")
async def query(
    payload: Dict[str, Any]
):
    if not state.index_built:
        raise HTTPException(status_code=400, detail="尚未上传并构建索引")

    question: str = payload.get("question", "").strip()
    mode: str = payload.get("mode", "attention")  # 'attention' | 'vector'
    top_k_parents: int = int(payload.get("top_k_parents", 4))
    top_k_sub: int = int(payload.get("top_k_sub", 50))

    llm_cfg = payload.get("llm", {})
    base_url: str = llm_cfg.get("base_url", "")
    api_key: str = llm_cfg.get("api_key", "")
    model: str = llm_cfg.get("model", "gpt-3.5-turbo")
    temperature: float = float(llm_cfg.get("temperature", 0.2))
    rag_prompt: str = payload.get("prompt", "")

    if not question:
        raise HTTPException(status_code=400, detail="缺少问题")

    # 1) 子文档相似度检索
    doc_ids, scores = state.vector.search(question, top_k=max(top_k_sub, top_k_parents))

    # 2) 聚合到父文档
    parent_best: Dict[int, float] = {}
    parent_hits: Dict[int, List[Tuple[str, float]]] = {}
    for did, sc in zip(doc_ids, scores):
        idx = state.sub_id_to_idx.get(did)
        if idx is None:
            continue
        pid = state.sub_to_parent[idx]
        parent_best[pid] = max(parent_best.get(pid, -1e9), float(sc))
        parent_hits.setdefault(pid, []).append((did, float(sc)))

    parent_candidates = sorted(parent_best.items(), key=lambda x: x[1], reverse=True)

    # 3) 重新排序（注意力 or 仅向量）
    selected: List[Tuple[int, float, float]] = []  # (pid, vec_score, attn_score)
    if mode == 'attention':
        for pid, vsc in parent_candidates[: max(5*top_k_parents, top_k_parents)]:
            ptext = state.parent_docs[pid]
            attn = state.attn.compute_lightweight_attention_score(question, ptext)
            selected.append((pid, float(vsc), float(attn)))
        # 以注意力分为主、向量为辅
        selected.sort(key=lambda x: (x[2], x[1]), reverse=True)
    else:
        for pid, vsc in parent_candidates:
            selected.append((pid, float(vsc), 0.0))
        selected.sort(key=lambda x: x[1], reverse=True)

    selected = selected[:top_k_parents]

    # 4) 组织上下文
    contexts: List[Dict[str, Any]] = []
    for pid, vsc, asc in selected:
        ctxt = state.parent_docs[pid]
        contexts.append({
            "parent_id": state.parent_ids[pid],
            "vector_score": vsc,
            "attention_score": asc,
            "content": ctxt
        })

    # 5) 调用LLM（OpenAI兼容）
    prompt = rag_prompt.strip() or (
        "你是一个严谨的中文助手。请严格基于给定的参考上下文回答用户问题：\n"
        "- 如果上下文无法支持答案，必须直接回答：‘无法根据参考上下文回答。’\n"
        "- 禁止编造或引入上下文以外的信息。\n"
        "- 回答尽量精炼。"
    )

    system_msg = {"role": "system", "content": prompt}
    context_text = "\n\n".join([f"[片段{i+1} - {c['parent_id']}]\n{c['content']}" for i, c in enumerate(contexts)])
    user_msg = {
        "role": "user",
        "content": f"参考上下文如下：\n{context_text}\n\n用户问题：{question}\n请仅依据参考上下文作答。若无法回答，请直接回复：无法根据参考上下文回答。"
    }

    llm_request = {
        "model": model,
        "messages": [system_msg, user_msg],
        "temperature": temperature,
        "stream": False
    }

    answer = ""
    llm_latency = 0.0
    if base_url and api_key:
        try:
            import requests
            t0 = time.time()
            url = base_url.rstrip('/') + "/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}", 
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity"  # 禁用压缩，避免gzip解压问题
            }
            
            # 发送请求
            resp = requests.post(
                url, 
                headers=headers, 
                json=llm_request,  # 使用json参数而不是data
                timeout=120,
                stream=False
            )
            resp.raise_for_status()
            
            # 解析响应
            try:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    answer = data["choices"][0]["message"]["content"].strip()
                else:
                    answer = f"[LLM响应格式错误] {data}"
            except json.JSONDecodeError as e:
                answer = f"[LLM响应解析失败] {e}, 响应内容: {resp.text[:200]}"
                
            llm_latency = time.time() - t0
            
        except requests.exceptions.RequestException as e:
            answer = f"[LLM网络请求失败] {e}"
        except Exception as e:
            answer = f"[LLM调用失败] {e}"
    else:
        answer = "[未配置LLM] 已返回候选上下文与排序结果。"

    return {
        "mode": mode,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "top_k_parents": top_k_parents,
        "llm_latency": llm_latency
    }

@app.post("/api/compare")
async def compare(payload: Dict[str, Any]):
    # 运行注意力与向量两种模式
    attn_payload = dict(payload)
    attn_payload["mode"] = "attention"
    vec_payload = dict(payload)
    vec_payload["mode"] = "vector"

    attn_res = await query(attn_payload)
    vec_res = await query(vec_payload)

    return {"attention": attn_res, "vector": vec_res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 