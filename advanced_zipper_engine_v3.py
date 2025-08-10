# advanced_zipper_engine_v3.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import logging
from dataclasses import dataclass
from contextlib import nullcontext

# 依赖项检查
try:
    from FlagEmbedding import FlagModel
except ImportError:
    raise ImportError("FlagEmbedding未安装。请运行: pip install FlagEmbedding")

# --- 新增: 引入HF模型 ---
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25未安装。请运行: pip install rank_bm25")

# --- 日志和设备配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- V3 配置定义 (更丰富，可控制) ---

@dataclass
class ZipperV3Config:
    bge_model_path: str = "models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    embedding_dim: int = 512
    
    # --- 新增: 编码后端选择 ('bge' 或 'hf') 与 HF 模型名 ---
    encoder_backend: str = "bge"
    hf_model_name: Optional[str] = None
    
    bm25_top_n: int = 100
    final_top_k: int = 10
    
    # 优化策略开关
    use_hybrid_search: bool = True
    bm25_weight: float = 1.0
    colbert_weight: float = 1.0

    use_length_penalty: bool = True
    length_penalty_alpha: float = 0.05

    use_multi_head: bool = True
    num_heads: int = 8

    use_stateful_reranking: bool = True
    context_memory_decay: float = 0.8
    context_influence: float = 0.3

    # 编码性能与兼容性
    encode_batch_size: int = 64
    max_length: int = 256
    precompute_doc_tokens: bool = False
    enable_amp_if_beneficial: bool = True
    
    # --- 新增: 重排序配置 ---
    use_reranker: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-large"
    reranker_top_n: int = 50
    reranker_weight: float = 1.5
    reranker_backend: str = "auto"  # 新增：重排序后端选择


@dataclass
class ZipperV3State:
    original_query: str
    context_vector: torch.Tensor
    
class TokenLevelEncoder:
    def __init__(self, model_path: str, use_fp16: bool = True, enable_amp_if_beneficial: bool = True, backend: str = "bge", hf_model_name: Optional[str] = None):
        self.backend = backend.lower()
        self.enable_amp_if_beneficial = enable_amp_if_beneficial
        self.gpu_name = (torch.cuda.get_device_name(0).upper() if torch.cuda.is_available() else "CPU")
        self.use_amp = (torch.cuda.is_available() and "GTX" not in self.gpu_name and self.enable_amp_if_beneficial)
        self.autocast_dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

        if self.backend == "bge":
            logger.info(f"正在加载BGE模型用于Token级编码: {model_path}")
            self.model = FlagModel(
                model_path,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=use_fp16 if device.type == 'cuda' else False
            )
            self.model.model.to(device).eval()
            self.tokenizer = self.model.tokenizer
            logger.info("Token级编码器(BGE)加载成功。")
        elif self.backend == "hf":
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("未安装 transformers。请运行: pip install transformers")
            model_id = hf_model_name or model_path
            logger.info(f"正在加载HF模型用于Token级编码: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()
            # 自动前缀：对E5家族自动加 query:/passage:
            self.query_prefix = ""
            self.doc_prefix = ""
            _name = model_id.lower()
            if "e5" in _name:
                self.query_prefix = "query: "
                self.doc_prefix = "passage: "
            logger.info("Token级编码器(HF)加载成功。")
        else:
            raise ValueError(f"不支持的编码后端: {self.backend}")

        # --- 新增：显卡/AMP 自适应信息日志 ---
        try:
            logger.info(
                f"设备: {device.type.upper()}, GPU: {self.gpu_name}, AMP启用: {self.use_amp}, autocast_dtype: "
                f"{('bf16' if self.autocast_dtype==torch.bfloat16 else 'fp16')}"
            )
        except Exception:
            pass

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_query(self, query: str) -> torch.Tensor:
        if self.backend == "bge":
            query_embeddings = self.model.encode_queries([query])
            return torch.tensor(query_embeddings, device=device)
        # HF: 使用平均池化并L2归一化
        text_for_encode = (getattr(self, 'query_prefix', '') + query) if getattr(self, 'query_prefix', '') else query
        inputs = self.tokenizer([text_for_encode], return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            use_amp = self.use_amp
            ctx = torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                outputs = self.model(**inputs)
                sent = self._mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
                sent = torch.nn.functional.normalize(sent, p=2, dim=1)
        return sent

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        if self.backend == "bge":
            doc_embeddings = self.model.encode(documents, batch_size=64)
            return torch.tensor(doc_embeddings, device=device)
        all_vecs = []
        bs = 64
        for i in range(0, len(documents), bs):
            batch = documents[i:i+bs]
            if getattr(self, 'doc_prefix', ''):
                batch = [self.doc_prefix + t for t in batch]
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                use_amp = self.use_amp
                ctx = torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    outputs = self.model(**inputs)
                    sent = self._mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    sent = torch.nn.functional.normalize(sent, p=2, dim=1)
            all_vecs.append(sent)
        return torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0, 0, device=device)

    def encode_tokens(self, text: str, max_length: int = 512) -> torch.Tensor:
        if self.backend == "bge":
            try:
                output = self.model.encode(text, return_dense=False, return_token_embeddings=True)
                return torch.tensor(output['token_embeddings'], device=device)
            except Exception:
                tokens = self.tokenize(text)
                if not tokens:
                    return torch.empty(0, self.model.model.config.hidden_size, device=device)
                inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True).to(device)
                with torch.no_grad():
                    outputs = self.model.model(**inputs, output_hidden_states=True)
                    token_embeddings = outputs.hidden_states[-1].squeeze(0)
                if token_embeddings.size(0) > 2:
                    return token_embeddings[1:-1]
                return token_embeddings
        else:
            text_for_encode = (getattr(self, 'doc_prefix', '') + text) if getattr(self, 'doc_prefix', '') else text
            inputs = self.tokenizer(text_for_encode, return_tensors='pt', max_length=max_length, truncation=True).to(device)
            with torch.no_grad():
                use_amp = self.use_amp
                ctx = torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    token_embeddings = outputs.hidden_states[-1].squeeze(0)
            if token_embeddings.size(0) > 2:
                token_embeddings = token_embeddings[1:-1]
            return token_embeddings

    def _forward_hidden(self, inputs):
        with torch.inference_mode():
            use_amp = self.use_amp
            ctx = torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                if self.backend == "bge":
                    outputs = self.model.model(**inputs)
                    return outputs.last_hidden_state
                else:
                    outputs = self.model(**inputs)
                    return outputs.last_hidden_state

    def encode_tokens_batch(self, texts: List[str], max_length: int = 384) -> List[torch.Tensor]:
        if not texts:
            return []
        out: List[torch.Tensor] = []
        if self.backend == "bge":
            model_device = next(self.model.model.parameters()).device
        else:
            model_device = next(self.model.parameters()).device
        left, right = 0, len(texts)
        bs = len(texts)
        step = bs
        while left < right:
            step = min(step, right - left)
            chunk = texts[left:left+step]
            try:
                if self.backend == 'hf' and getattr(self, 'doc_prefix', ''):
                    chunk = [self.doc_prefix + t for t in chunk]
                inputs = self.tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(model_device)
                last_hidden = self._forward_hidden(inputs)
                attn = inputs['attention_mask']
                for i in range(last_hidden.size(0)):
                    valid = attn[i].bool()
                    vecs = last_hidden[i][valid]
                    if vecs.size(0) > 2:
                        vecs = vecs[1:-1]
                    out.append(vecs.contiguous())
                left += step
                step = bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and step > 1 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    step = max(1, step // 2)
                    self.use_amp = False
                    continue
                raise
        return out


# --- 新增: 重排序器类 ---
class CrossEncoderReranker:
    """交叉编码器重排序器，支持多种后端：FlagEmbedding、HuggingFace transformers、ONNX"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True, backend: str = "auto"):
        self.model_name = model_name
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.backend = backend
        self.model = None
        self.tokenizer = None
        
        # 尝试按优先级加载不同的后端
        self._load_model()
    
    def _load_model(self):
        """按优先级加载重排序模型"""
        backends_to_try = []
        
        if self.backend == "auto":
            # 自动选择：优先FlagEmbedding，然后是HF，最后是ONNX
            backends_to_try = ["flagembedding", "transformers", "onnx"]
        else:
            backends_to_try = [self.backend]
        
        for backend_type in backends_to_try:
            if self._try_load_backend(backend_type):
                logger.info(f"重排序模型加载成功，使用后端: {backend_type}")
                return
        
        logger.error("所有重排序后端都加载失败，重排序功能将被禁用")
    
    def _try_load_backend(self, backend_type: str) -> bool:
        """尝试加载指定的后端"""
        try:
            if backend_type == "flagembedding":
                return self._load_flagembedding()
            elif backend_type == "transformers":
                return self._load_transformers()
            elif backend_type == "onnx":
                return self._load_onnx()
            return False
        except Exception as e:
            logger.warning(f"加载{backend_type}后端失败: {e}")
            return False
    
    def _load_flagembedding(self) -> bool:
        """加载FlagEmbedding后端"""
        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16
            )
            return True
        except ImportError:
            logger.warning("FlagEmbedding未安装")
            return False
    
    def _load_transformers(self) -> bool:
        """加载HuggingFace transformers后端"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            if self.use_fp16 and device.type == 'cuda':
                self.model = self.model.half()
            self.model = self.model.to(device)
            return True
        except Exception as e:
            logger.warning(f"加载transformers后端失败: {e}")
            return False
    
    def _load_onnx(self) -> bool:
        """加载ONNX后端"""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # 尝试加载ONNX模型，如果失败则回退到原始模型
            try:
                self.model = ORTModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    file_name="onnx/model.onnx"
                )
            except:
                # 如果没有ONNX文件，使用原始模型
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.eval()
                if self.use_fp16 and device.type == 'cuda':
                    self.model = self.model.half()
                self.model = self.model.to(device)
            return True
        except Exception as e:
            logger.warning(f"加载ONNX后端失败: {e}")
            return False
    
    def rerank(self, query: str, documents: List[str], top_n: int = None) -> List[Tuple[int, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_n: 返回前N个结果，如果为None则返回所有
            
        Returns:
            List[Tuple[int, float]]: (文档索引, 重排序分数) 的列表，按分数降序排列
        """
        if self.model is None or not documents:
            return []
        
        try:
            if hasattr(self.model, 'compute_score'):  # FlagEmbedding
                return self._rerank_flagembedding(query, documents, top_n)
            else:  # Transformers/ONNX
                return self._rerank_transformers(query, documents, top_n)
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return []
    
    def _rerank_flagembedding(self, query: str, documents: List[str], top_n: int = None) -> List[Tuple[int, float]]:
        """使用FlagEmbedding进行重排序"""
        # 准备查询-文档对
        pairs = [[query, doc] for doc in documents]
        
        # 批量计算重排序分数
        scores = self.model.compute_score(pairs)
        
        # 如果scores是单个值，转换为列表
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        # 创建(索引, 分数)对并排序
        scored_pairs = [(i, score) for i, score in enumerate(scores)]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 限制返回数量
        if top_n is not None:
            scored_pairs = scored_pairs[:top_n]
        
        return scored_pairs
    
    def _rerank_transformers(self, query: str, documents: List[str], top_n: int = None) -> List[Tuple[int, float]]:
        """使用Transformers/ONNX进行重排序"""
        # 准备查询-文档对
        pairs = [['query', query], ['passage', doc] for doc in documents]
        
        # 使用tokenizer处理输入
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )
            
            # 将输入移到正确的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 计算分数
            outputs = self.model(**inputs, return_dict=True)
            scores = outputs.logits.view(-1, ).float()
            
            # 创建(索引, 分数)对并排序
            scored_pairs = [(i, score.item()) for i, score in enumerate(scores)]
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 限制返回数量
            if top_n is not None:
                scored_pairs = scored_pairs[:top_n]
            
            return scored_pairs
    
    def is_available(self) -> bool:
        """检查重排序器是否可用"""
        return self.model is not None
    
    def get_backend_info(self) -> str:
        """获取当前使用的后端信息"""
        if hasattr(self.model, 'compute_score'):
            return "FlagEmbedding"
        elif hasattr(self.model, 'forward'):
            return "Transformers/ONNX"
        else:
            return "Unknown"


# --- V3 主引擎 ---
class AdvancedZipperQueryEngineV3:
    def __init__(self, config: ZipperV3Config):
        self.config = config
        self.encoder = TokenLevelEncoder(
            config.bge_model_path,
            enable_amp_if_beneficial=config.enable_amp_if_beneficial,
            backend=config.encoder_backend,
            hf_model_name=config.hf_model_name
        )
        
        # --- 新增: 初始化重排序器 ---
        if config.use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name=config.reranker_model_name,
                use_fp16=config.enable_amp_if_beneficial,
                backend=config.reranker_backend
            )
            if self.reranker.is_available():
                logger.info(f"重排序器初始化成功: {config.reranker_model_name}")
            else:
                logger.warning("重排序器初始化失败，将禁用重排序功能")
                config.use_reranker = False
        else:
            self.reranker = None
        
        if config.use_multi_head and config.embedding_dim % config.num_heads != 0:
            print(config.embedding_dim, config.num_heads)
            raise ValueError("embedding_dim 必须能被 num_heads 整除")
        
        self.documents: Dict[int, str] = {}
        self.doc_token_embeddings: Dict[int, torch.Tensor] = {}
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_idx_to_pid: Dict[int, int] = {}
        
        logger.info("AdvancedZipperQueryEngine V3 初始化完成 (多策略优化版 + 重排序)")

    def build_document_index(self, documents: Dict[int, str]):
        logger.info(f"开始构建V3索引，共 {len(documents)} 个文档...")
        start_time = time.time()
        self.documents = documents
        
        doc_ids_sorted = sorted(self.documents.keys())
        corpus_list = [self.documents[pid] for pid in doc_ids_sorted]
        
        logger.info("构建BM25稀疏索引...")
        tokenized_corpus = [self.encoder.tokenize(doc) for doc in corpus_list]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_idx_to_pid = {i: pid for i, pid in enumerate(doc_ids_sorted)}
        
        logger.info("构建Token级稠密索引...")
        self.doc_token_embeddings = {}
        if self.config.precompute_doc_tokens:
            bs = self.config.encode_batch_size
            max_len = self.config.max_length
            for i in range(0, len(corpus_list), bs):
                batch_texts = corpus_list[i:i+bs]
                batch_pids = doc_ids_sorted[i:i+bs]
                batch_vecs = self.encoder.encode_tokens_batch(batch_texts, max_length=max_len)
                for pid, vecs in zip(batch_pids, batch_vecs):
                    self.doc_token_embeddings[pid] = vecs
                if (i + bs) % 100 == 0 or i + bs >= len(corpus_list):
                    logger.info(f"  已处理 {min(i+bs, len(corpus_list))}/{len(corpus_list)} 个文档的Token向量化")
        else:
            logger.info("跳过全量token向量化，将在检索时按需编码候选文档。")

        logger.info(f"V3索引构建完成，总耗时: {time.time() - start_time:.3f}秒")

    def _calculate_colbert_score(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        if query_emb.nelement() == 0 or doc_emb.nelement() == 0: return 0.0

        if self.config.use_multi_head:
            num_heads = self.config.num_heads
            head_dim = self.config.embedding_dim // num_heads
            
            # 检查维度兼容性
            q_size = query_emb.size(-1)
            d_size = doc_emb.size(-1)
            
            if q_size != self.config.embedding_dim or d_size != self.config.embedding_dim:
                # 如果维度不匹配，回退到单头模式
                logger.warning(f"维度不匹配，回退到单头模式: query_dim={q_size}, doc_dim={d_size}, expected={self.config.embedding_dim}")
                sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
                max_sim = sim_matrix.max(dim=1).values
                score = max_sim.sum().item()
            else:
                # 安全的reshape操作
                try:
                    q_heads = query_emb.view(-1, num_heads, head_dim)
                    d_heads = doc_emb.view(-1, num_heads, head_dim)
                    head_scores = []
                    for i in range(num_heads):
                        sim_matrix_head = F.cosine_similarity(q_heads[:, i, :].unsqueeze(1), d_heads[:, i, :].unsqueeze(0), dim=-1)
                        max_sim_head = sim_matrix_head.max(dim=1).values
                        head_scores.append(max_sim_head.sum())
                    score = sum(head_scores).item()
                except RuntimeError as e:
                    logger.warning(f"多头计算失败，回退到单头模式: {e}")
                    sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
                    max_sim = sim_matrix.max(dim=1).values
                    score = max_sim.sum().item()
        else:
            sim_matrix = F.cosine_similarity(query_emb.unsqueeze(1), doc_emb.unsqueeze(0), dim=-1)
            max_sim = sim_matrix.max(dim=1).values
            score = max_sim.sum().item()
            
        if self.config.use_length_penalty:
            num_doc_tokens = doc_emb.size(0)
            penalty = 1.0 + self.config.length_penalty_alpha * np.log(num_doc_tokens + 1)
            score /= penalty
            
        return score

    def retrieve(self, query: str, state: Optional[ZipperV3State] = None) -> List[Tuple[int, float, str]]:
        if self.bm25_index is None: return []

        logger.info(f"V3 开始检索，查询: '{query[:50]}...'")
        
        # 1. BM25召回
        query_tokens_list = self.encoder.tokenize(query)
        bm25_raw_scores = self.bm25_index.get_scores(query_tokens_list)
        bm25_candidate_indices = np.argsort(bm25_raw_scores)[::-1][:self.config.bm25_top_n]
        candidate_pids = [self.bm25_idx_to_pid[idx] for idx in bm25_candidate_indices]
        bm25_scores_map = {pid: bm25_raw_scores[idx] for idx, pid in zip(bm25_candidate_indices, candidate_pids)}

        # 1.5 按需补齐候选文档的 Token 向量
        missing_pids = [pid for pid in candidate_pids if pid not in self.doc_token_embeddings]
        if missing_pids:
            texts = [self.documents[pid] for pid in missing_pids]
            bs = self.config.encode_batch_size
            max_len = self.config.max_length
            for i in range(0, len(texts), bs):
                batch_texts = texts[i:i+bs]
                batch_pids = missing_pids[i:i+bs]
                batch_vecs = self.encoder.encode_tokens_batch(batch_texts, max_length=max_len)
                for pid, vecs in zip(batch_pids, batch_vecs):
                    self.doc_token_embeddings[pid] = vecs

        # 2. 准备查询向量 (可能被状态化调整)
        query_tokens_emb = self.encoder.encode_tokens(query, max_length=self.config.max_length)
        if state and self.config.use_stateful_reranking and state.context_vector.sum() != 0:
            logger.info("应用状态化上下文调整查询...")
            influence = self.config.context_influence
            dynamic_adjustment = state.context_vector.unsqueeze(0) * influence
            query_tokens_emb += dynamic_adjustment

        # 3. 计算ColBERT分数
        colbert_scores_map = {pid: self._calculate_colbert_score(query_tokens_emb, self.doc_token_embeddings.get(pid, torch.empty(0))) for pid in candidate_pids}

        # 4. 分数融合
        fused_scores = {}
        bm25_vals = np.array(list(bm25_scores_map.values()))
        colbert_vals = np.array(list(colbert_scores_map.values()))
        bm25_min, bm25_max = (bm25_vals.min(), bm25_vals.max()) if bm25_vals.size > 1 else (0, 1)
        colbert_min, colbert_max = (colbert_vals.min(), colbert_vals.max()) if colbert_vals.size > 1 else (0, 1)
        
        for pid in candidate_pids:
            norm_bm25 = (bm25_scores_map.get(pid, 0) - bm25_min) / (bm25_max - bm25_min + 1e-9)
            norm_colbert = (colbert_scores_map.get(pid, 0) - colbert_min) / (colbert_max - colbert_min + 1e-9)
            
            if self.config.use_hybrid_search:
                fused_scores[pid] = (self.config.bm25_weight * norm_bm25 + self.config.colbert_weight * norm_colbert)
            else:
                fused_scores[pid] = norm_colbert
        
        # 5. --- 新增: 重排序阶段 ---
        if self.config.use_reranker and self.reranker and self.reranker.is_available():
            logger.info("开始重排序阶段...")
            rerank_start_time = time.time()
            
            # 选择前N个候选文档进行重排序
            top_candidates_for_rerank = sorted(candidate_pids, key=lambda pid: fused_scores.get(pid, 0), reverse=True)[:self.config.reranker_top_n]
            candidate_docs = [self.documents[pid] for pid in top_candidates_for_rerank]
            
            # 执行重排序
            rerank_results = self.reranker.rerank(query, candidate_docs, top_n=self.config.reranker_top_n)
            
            if rerank_results:
                # 创建重排序分数映射
                rerank_scores_map = {}
                for idx, score in rerank_results:
                    pid = top_candidates_for_rerank[idx]
                    rerank_scores_map[pid] = score
                
                # 归一化重排序分数
                rerank_vals = np.array(list(rerank_scores_map.values()))
                if rerank_vals.size > 1:
                    rerank_min, rerank_max = rerank_vals.min(), rerank_vals.max()
                    for pid in rerank_scores_map:
                        if rerank_max > rerank_min:
                            rerank_scores_map[pid] = (rerank_scores_map[pid] - rerank_min) / (rerank_max - rerank_min)
                        else:
                            rerank_scores_map[pid] = 1.0
                
                # 融合重排序分数
                for pid in top_candidates_for_rerank:
                    if pid in rerank_scores_map:
                        fused_scores[pid] = fused_scores.get(pid, 0) + self.config.reranker_weight * rerank_scores_map[pid]
                
                logger.info(f"重排序完成，处理了 {len(rerank_results)} 个文档，耗时: {time.time() - rerank_start_time:.3f}秒")
            else:
                logger.warning("重排序失败，使用原始融合分数")
        
        # 6. 最终排序和返回
        sorted_pids = sorted(candidate_pids, key=lambda pid: fused_scores.get(pid, 0), reverse=True)
        
        return [(pid, fused_scores.get(pid, 0), self.documents[pid]) for pid in sorted_pids[:self.config.final_top_k]]

    def update_state(self, state: ZipperV3State, results: List[Tuple[int, float, str]]) -> ZipperV3State:
        if not results or not self.config.use_stateful_reranking: return state
        top_doc_pid = results[0][0]
        top_doc_tokens_emb = self.doc_token_embeddings.get(top_doc_pid)
        if top_doc_tokens_emb is not None and top_doc_tokens_emb.nelement() > 0:
            top_doc_avg_emb = top_doc_tokens_emb.mean(dim=0)
            decay = self.config.context_memory_decay
            state.context_vector = decay * state.context_vector + (1-decay) * top_doc_avg_emb
        return state