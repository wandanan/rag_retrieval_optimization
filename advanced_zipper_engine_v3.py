# advanced_zipper_engine_v3.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
import logging
from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import pickle
import os

# 依赖项检查
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("transformers未安装。请运行: pip install transformers")

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
    # 强制使用HF模型，移除本地模型选择
    encoder_backend: str = "hf"  # 固定为HF
    hf_model_name: str = "BAAI/bge-small-zh-v1.5"  # 默认HF模型
    embedding_dim: int = 512
    
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
    precompute_doc_tokens: bool = True  # 默认改为True，避免每次查询都编码
    enable_amp_if_beneficial: bool = True
    
    # --- 新增: 重排序配置 ---
    use_reranker: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-large"
    reranker_top_n: int = 50
    reranker_weight: float = 1.5
    reranker_backend: str = "auto"  # 新增：重排序后端选择
    
    # --- 新增: 索引缓存配置 ---
    enable_index_cache: bool = True
    cache_dir: str = "index_cache"
    cache_version: str = "v3.0"
    
    # --- 新增: 索引构建策略配置 ---
    auto_build_index: bool = True  # 自动构建索引
    incremental_update: bool = True  # 启用增量更新
    warmup_on_first_query: bool = True  # 首次查询时预热
    index_rebuild_threshold: float = 0.3  # 文档变化超过30%时重建索引


@dataclass
class ZipperV3State:
    original_query: str
    context_vector: torch.Tensor


# --- 新增: 索引缓存管理器 ---
class IndexCacheManager:
    """索引缓存管理器，避免重复构建索引"""
    
    def __init__(self, cache_dir: str = "index_cache", cache_version: str = "v3.0"):
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, documents: Dict[int, str], config_hash: str) -> str:
        """生成缓存键"""
        # 基于文档内容和配置生成唯一标识
        doc_content = "".join([f"{k}:{v[:100]}" for k, v in sorted(documents.items())])
        content_hash = hashlib.md5(doc_content.encode()).hexdigest()
        return f"{self.cache_version}_{config_hash}_{content_hash}"
    
    def _get_config_hash(self, config: ZipperV3Config) -> str:
        """生成配置哈希"""
        config_str = f"{config.hf_model_name}_{config.embedding_dim}_{config.max_length}_{config.precompute_doc_tokens}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_index(self, cache_key: str, index_data: dict) -> bool:
        """保存索引到缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(index_data, f)
            logger.info(f"索引缓存保存成功: {cache_file}")
            return True
        except Exception as e:
            logger.warning(f"索引缓存保存失败: {e}")
            return False
    
    def load_index(self, cache_key: str) -> Optional[dict]:
        """从缓存加载索引"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    index_data = pickle.load(f)
                logger.info(f"索引缓存加载成功: {cache_file}")
                return index_data
        except Exception as e:
            logger.warning(f"索引缓存加载失败: {e}")
        return None
    
    def clear_cache(self):
        """清理所有缓存"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("索引缓存清理完成")
        except Exception as e:
            logger.warning(f"索引缓存清理失败: {e}")


class TokenLevelEncoder:
    def __init__(self, model_name: str, use_fp16: bool = True, enable_amp_if_beneficial: bool = True, backend: str = "hf", hf_model_name: Optional[str] = None):
        # 强制使用HF后端
        self.backend = "hf"
        self.enable_amp_if_beneficial = enable_amp_if_beneficial
        self.gpu_name = (torch.cuda.get_device_name(0).upper() if torch.cuda.is_available() else "CPU")
        self.use_amp = (torch.cuda.is_available() and "GTX" not in self.gpu_name and self.enable_amp_if_beneficial)
        self.autocast_dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

        # 强制使用HF模型
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("未安装 transformers。请运行: pip install transformers")
        
        # 使用传入的模型名称或默认名称
        model_id = hf_model_name or model_name
        if not model_id or model_id.strip() == "":
            # 如果都没有提供，使用默认的BGE模型
            model_id = "BAAI/bge-small-zh-v1.5"
            logger.warning(f"未提供有效的HF模型名称，使用默认模型: {model_id}")
        
        # 确保模型名称有效
        if not model_id or model_id.strip() == "":
            raise ValueError("必须提供有效的HF模型名称")
        
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
        # 只使用HF模型：使用平均池化并L2归一化
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
        # 只使用HF模型
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
        return torch.cat(all_vecs, dim=0)

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
    
    def switch_model(self, new_model_name: str, new_backend: str = "auto") -> bool:
        """
        动态切换重排序模型
        
        Args:
            new_model_name: 新的模型名称
            new_backend: 新的后端类型
            
        Returns:
            bool: 切换是否成功
        """
        try:
            logger.info(f"正在切换重排序模型: {self.model_name} -> {new_model_name}")
            
            # 清理当前模型
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # 更新配置
            self.model_name = new_model_name
            self.backend = new_backend
            
            # 重新加载模型
            self._load_model()
            
            if self.model is not None:
                logger.info(f"重排序模型切换成功: {new_model_name}")
                return True
            else:
                logger.error(f"重排序模型切换失败: {new_model_name}")
                return False
                
        except Exception as e:
            logger.error(f"重排序模型切换时发生错误: {e}")
            return False
    
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
        pairs = [['query', query]] + [['passage', doc] for doc in documents]
        
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
    
    def get_model_info(self) -> Dict[str, str]:
        """获取当前模型信息"""
        return {
            'model_name': self.model_name,
            'backend': self.backend,
            'is_available': self.is_available(),
            'device': str(device),
            'use_fp16': self.use_fp16
        }


# --- V3 主引擎 ---
class AdvancedZipperQueryEngineV3:
    def __init__(self, config: ZipperV3Config):
        self.config = config
        self.encoder = TokenLevelEncoder(
            config.hf_model_name, # 强制使用HF模型
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
        
        # --- 新增: 索引状态管理 ---
        self.documents: Dict[int, str] = {}
        self.doc_token_embeddings: Dict[int, torch.Tensor] = {}
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_idx_to_pid: Dict[int, int] = {}
        self.pid_to_bm25_idx: Dict[int, int] = {}  # 新增：反向映射，提高查找效率
        self.index_built: bool = False  # 新增：索引构建状态标志
        
        # --- 新增: 索引缓存管理器 ---
        if config.enable_index_cache:
            self.cache_manager = IndexCacheManager(
                cache_dir=config.cache_dir,
                cache_version=config.cache_version
            )
        else:
            self.cache_manager = None
        
        # --- 新增: 索引状态跟踪 ---
        self.documents_hash: str = ""  # 文档内容哈希
        self.last_query_time: float = 0  # 上次查询时间
        self.query_count: int = 0  # 查询次数统计
        
        logger.info("AdvancedZipperQueryEngine V3 初始化完成 (多策略优化版 + 重排序 + 索引缓存 + 智能索引管理)")

    def build_document_index(self, documents: Dict[int, str], force_rebuild: bool = False):
        """
        智能构建文档索引，支持缓存、增量更新和智能重建
        
        Args:
            documents: 文档字典 {doc_id: content}
            force_rebuild: 是否强制重新构建索引
        """
        # 计算文档内容哈希
        current_hash = self._calculate_documents_hash(documents)
        
        # 检查是否已经构建过索引且内容未变化
        if self.index_built and not force_rebuild and current_hash == self.documents_hash:
            logger.info("索引已存在且文档未变化，跳过重复构建")
            return
        
        # 检查是否需要增量更新
        if (self.index_built and self.config.incremental_update and 
            not force_rebuild and current_hash != self.documents_hash):
            logger.info("检测到文档变化，执行增量更新...")
            self._incremental_update_index(documents, current_hash)
            return
        
        # 检查是否需要重建索引
        if self.index_built and not force_rebuild:
            change_ratio = self._calculate_change_ratio(documents)
            if change_ratio > self.config.index_rebuild_threshold:
                logger.info(f"文档变化率({change_ratio:.1%})超过阈值，重建索引...")
                force_rebuild = True
        
        logger.info(f"开始构建V3索引，共 {len(documents)} 个文档...")
        start_time = time.time()
        
        # 尝试从缓存加载索引
        if self.cache_manager and not force_rebuild:
            config_hash = self.cache_manager._get_config_hash(self.config)
            cache_key = self.cache_manager._get_cache_key(documents, config_hash)
            cached_index = self.cache_manager.load_index(cache_key)
            
            if cached_index:
                # 从缓存恢复索引
                self._restore_index_from_cache(cached_index)
                self.documents_hash = current_hash
                logger.info(f"索引从缓存加载成功，总耗时: {time.time() - start_time:.3f}秒")
                return
        
        # 构建新索引
        self._build_full_index(documents, start_time)
        self.documents_hash = current_hash
        
        # 保存索引到缓存
        if self.cache_manager:
            self._save_index_to_cache()
        
        logger.info(f"V3索引构建完成，总耗时: {time.time() - start_time:.3f}秒")
    
    def _calculate_documents_hash(self, documents: Dict[int, str]) -> str:
        """计算文档内容的哈希值"""
        content = "".join([f"{k}:{v}" for k, v in sorted(documents.items())])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_change_ratio(self, new_documents: Dict[int, str]) -> float:
        """计算文档变化率"""
        if not self.documents:
            return 1.0
        
        total_docs = len(set(self.documents.keys()) | set(new_documents.keys()))
        changed_docs = 0
        
        for doc_id in set(self.documents.keys()) | set(new_documents.keys()):
            old_content = self.documents.get(doc_id, "")
            new_content = new_documents.get(doc_id, "")
            if old_content != new_content:
                changed_docs += 1
        
        return changed_docs / total_docs if total_docs > 0 else 0.0
    
    def _incremental_update_index(self, new_documents: Dict[int, str], new_hash: str):
        """增量更新索引"""
        start_time = time.time()
        
        # 找出新增、删除和修改的文档
        added_pids = set(new_documents.keys()) - set(self.documents.keys())
        removed_pids = set(self.documents.keys()) - set(new_documents.keys())
        modified_pids = {pid for pid in set(self.documents.keys()) & set(new_documents.keys()) 
                        if self.documents[pid] != new_documents[pid]}
        
        logger.info(f"增量更新: 新增{len(added_pids)}个, 删除{len(removed_pids)}个, 修改{len(modified_pids)}个文档")
        
        # 更新文档字典
        self.documents = new_documents.copy()
        
        # 删除已移除文档的索引
        for pid in removed_pids:
            self.doc_token_embeddings.pop(pid, None)
        
        # 重新构建BM25索引（因为文档变化会影响整个索引）
        doc_ids_sorted = sorted(self.documents.keys())
        corpus_list = [self.documents[pid] for pid in doc_ids_sorted]
        tokenized_corpus = [self.encoder.tokenize(doc) for doc in corpus_list]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_idx_to_pid = {i: pid for i, pid in enumerate(doc_ids_sorted)}
        self.pid_to_bm25_idx = {pid: i for i, pid in enumerate(doc_ids_sorted)}  # 构建反向映射
        
        # 更新token embeddings（仅处理新增和修改的文档）
        pids_to_update = added_pids | modified_pids
        if pids_to_update and self.config.precompute_doc_tokens:
            self._update_token_embeddings(pids_to_update)
        
        self.documents_hash = new_hash
        logger.info(f"增量更新完成，耗时: {time.time() - start_time:.3f}秒")
    
    def _build_full_index(self, documents: Dict[int, str], start_time: float):
        """构建完整索引"""
        self.documents = documents
        
        doc_ids_sorted = sorted(self.documents.keys())
        corpus_list = [self.documents[pid] for pid in doc_ids_sorted]
        
        logger.info("构建BM25稀疏索引...")
        tokenized_corpus = [self.encoder.tokenize(doc) for doc in corpus_list]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_idx_to_pid = {i: pid for i, pid in enumerate(doc_ids_sorted)}
        self.pid_to_bm25_idx = {pid: i for i, pid in enumerate(doc_ids_sorted)}  # 构建反向映射
        
        logger.info("构建Token级稠密索引...")
        self.doc_token_embeddings = {}
        if self.config.precompute_doc_tokens:
            self._update_token_embeddings(doc_ids_sorted)
        else:
            logger.info("跳过全量token向量化，将在检索时按需编码候选文档。")
        
        self.index_built = True
    
    def _update_token_embeddings(self, pids: List[int]):
        """更新指定文档的token embeddings"""
        bs = self.config.encode_batch_size
        max_len = self.config.max_length
        
        for i in range(0, len(pids), bs):
            batch_pids = pids[i:i+bs]
            batch_texts = [self.documents[pid] for pid in batch_pids]
            batch_vecs = self.encoder.encode_tokens_batch(batch_texts, max_length=max_len)
            
            for pid, vecs in zip(batch_pids, batch_vecs):
                self.doc_token_embeddings[pid] = vecs
            
            if (i + bs) % 100 == 0 or i + bs >= len(pids):
                logger.info(f"  已处理 {min(i+bs, len(pids))}/{len(pids)} 个文档的Token向量化")
    
    def _restore_index_from_cache(self, cached_index: dict):
        """从缓存恢复索引"""
        self.documents = cached_index['documents']
        self.doc_token_embeddings = cached_index['doc_token_embeddings']
        self.bm25_index = cached_index['bm25_index']
        self.bm25_idx_to_pid = cached_index['bm25_idx_to_pid']
        self.pid_to_bm25_idx = cached_index.get('pid_to_bm25_idx', {})  # 兼容旧版本缓存
        self.index_built = True
    
    def _save_index_to_cache(self):
        """保存索引到缓存"""
        try:
            config_hash = self.cache_manager._get_config_hash(self.config)
            cache_key = self.cache_manager._get_cache_key(self.documents, config_hash)
            index_data = {
                'documents': self.documents,
                'doc_token_embeddings': self.doc_token_embeddings,
                'bm25_index': self.bm25_index,
                'bm25_idx_to_pid': self.bm25_idx_to_pid,
                'pid_to_bm25_idx': self.pid_to_bm25_idx  # 保存反向映射
            }
            self.cache_manager.save_index(cache_key, index_data)
        except Exception as e:
            logger.warning(f"索引缓存保存失败: {e}")

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
        # 智能索引检查和预热
        self._ensure_index_ready()
        
        # 更新查询统计
        self.query_count += 1
        self.last_query_time = time.time()
        
        if self.bm25_index is None: return []

        logger.info(f"V3 开始检索，查询: '{query[:50]}...'")
        
        # 1. BM25召回
        query_tokens_list = self.encoder.tokenize(query)
        bm25_raw_scores = self.bm25_index.get_scores(query_tokens_list)
        bm25_candidate_indices = np.argsort(bm25_raw_scores)[::-1][:self.config.bm25_top_n]
        candidate_pids = [self.bm25_idx_to_pid[idx] for idx in bm25_candidate_indices]
        bm25_scores_map = {pid: bm25_raw_scores[idx] for idx, pid in zip(bm25_candidate_indices, candidate_pids)}

        # 1.5 按需补齐候选文档的 Token 向量（仅当precompute_doc_tokens=False时）
        if not self.config.precompute_doc_tokens:
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
            # 使用反向映射快速找到对应的BM25索引位置
            bm25_idx = self.pid_to_bm25_idx.get(pid)
            
            if bm25_idx is not None:
                norm_bm25 = (bm25_raw_scores[bm25_idx] - bm25_min) / (bm25_max - bm25_min + 1e-9)
            else:
                norm_bm25 = 0.0  # 如果找不到对应的索引，使用默认分数
                
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
    
    def _ensure_index_ready(self):
        """确保索引已准备就绪，必要时进行预热"""
        if not self.index_built:
            if self.config.auto_build_index and self.documents:
                logger.info("索引未构建，自动构建索引...")
                self.build_document_index(self.documents)
            else:
                raise RuntimeError("索引尚未构建，请先调用 build_document_index()")
        
        # 首次查询预热
        if (self.config.warmup_on_first_query and self.query_count == 1 and 
            self.config.precompute_doc_tokens and not self.doc_token_embeddings):
            logger.info("首次查询，预热token embeddings...")
            self._warmup_token_embeddings()
    
    def _warmup_token_embeddings(self):
        """预热token embeddings"""
        if not self.documents:
            return
        
        start_time = time.time()
        doc_ids = list(self.documents.keys())
        
        # 选择前几个文档进行预热
        warmup_count = min(10, len(doc_ids))
        warmup_pids = doc_ids[:warmup_count]
        
        logger.info(f"预热 {warmup_count} 个文档的token embeddings...")
        self._update_token_embeddings(warmup_pids)
        
        logger.info(f"预热完成，耗时: {time.time() - start_time:.3f}秒")

    def update_state(self, state: ZipperV3State, results: List[Tuple[int, float, str]]) -> ZipperV3State:
        if not results or not self.config.use_stateful_reranking: return state
        top_doc_pid = results[0][0]
        top_doc_tokens_emb = self.doc_token_embeddings.get(top_doc_pid)
        if top_doc_tokens_emb is not None and top_doc_tokens_emb.nelement() > 0:
            top_doc_avg_emb = top_doc_tokens_emb.mean(dim=0)
            decay = self.config.context_memory_decay
            state.context_vector = decay * state.context_vector + (1-decay) * top_doc_avg_emb
        return state
    
    # --- 新增: 索引管理方法 ---
    def is_index_ready(self) -> bool:
        """检查索引是否已准备就绪"""
        return self.index_built and self.bm25_index is not None
    
    def clear_index(self):
        """清理索引"""
        self.documents.clear()
        self.doc_token_embeddings.clear()
        self.bm25_index = None
        self.bm25_idx_to_pid.clear()
        self.pid_to_bm25_idx.clear()
        self.index_built = False
        self.documents_hash = ""
        self.query_count = 0
        self.last_query_time = 0
        logger.info("索引已清理")
    
    def get_index_stats(self) -> Dict[str, any]:
        """获取索引统计信息"""
        return {
            'index_built': self.index_built,
            'num_documents': len(self.documents),
            'num_token_embeddings': len(self.doc_token_embeddings),
            'bm25_index_exists': self.bm25_index is not None,
            'cache_enabled': self.cache_manager is not None,
            'documents_hash': self.documents_hash,
            'query_count': self.query_count,
            'last_query_time': self.last_query_time,
            'precompute_tokens': self.config.precompute_doc_tokens,
            'incremental_update': self.config.incremental_update
        }
    
    def add_documents(self, new_documents: Dict[int, str]):
        """添加新文档到索引"""
        if not self.index_built:
            logger.warning("索引未构建，请先调用 build_document_index()")
            return
        
        if self.config.incremental_update:
            # 合并文档并增量更新
            merged_docs = {**self.documents, **new_documents}
            self.build_document_index(merged_docs)
        else:
            # 强制重建索引
            merged_docs = {**self.documents, **new_documents}
            self.build_document_index(merged_docs, force_rebuild=True)
    
    def remove_documents(self, doc_ids: List[int]):
        """从索引中移除文档"""
        if not self.index_built:
            logger.warning("索引未构建，请先调用 build_document_index()")
            return
        
        if self.config.incremental_update:
            # 移除指定文档并增量更新
            remaining_docs = {k: v for k, v in self.documents.items() if k not in doc_ids}
            self.build_document_index(remaining_docs)
        else:
            # 强制重建索引
            remaining_docs = {k: v for k, v in self.documents.items() if k not in doc_ids}
            self.build_document_index(remaining_docs, force_rebuild=True)
    
    def force_rebuild_index(self):
        """强制重建索引"""
        if self.documents:
            logger.info("强制重建索引...")
            self.build_document_index(self.documents, force_rebuild=True)
        else:
            logger.warning("没有文档可重建索引")

    def update_reranker_config(self, use_reranker: bool = None, reranker_model_name: str = None, 
                              reranker_top_n: int = None, reranker_weight: float = None, 
                              reranker_backend: str = None) -> bool:
        """
        动态更新重排序配置
        
        Args:
            use_reranker: 是否启用重排序
            reranker_model_name: 重排序模型名称
            reranker_top_n: 重排序候选数量
            reranker_weight: 重排序权重
            reranker_backend: 重排序后端
            
        Returns:
            bool: 更新是否成功
        """
        try:
            logger.info("正在更新重排序配置...")
            
            # 更新配置
            if use_reranker is not None:
                self.config.use_reranker = use_reranker
                logger.info(f"重排序开关已更新: {use_reranker}")
            
            if reranker_top_n is not None:
                self.config.reranker_top_n = reranker_top_n
                logger.info(f"重排序候选数量已更新: {reranker_top_n}")
            
            if reranker_weight is not None:
                self.config.reranker_weight = reranker_weight
                logger.info(f"重排序权重已更新: {reranker_weight}")
            
            if reranker_backend is not None:
                self.config.reranker_backend = reranker_backend
                logger.info(f"重排序后端已更新: {reranker_backend}")
            
            # 如果模型名称发生变化，需要重新加载模型
            if reranker_model_name is not None and reranker_model_name != self.config.reranker_model_name:
                self.config.reranker_model_name = reranker_model_name
                logger.info(f"重排序模型名称已更新: {reranker_model_name}")
                
                # 如果重排序器已存在，尝试切换模型
                if self.reranker and self.config.use_reranker:
                    success = self.reranker.switch_model(reranker_model_name, self.config.reranker_backend)
                    if success:
                        logger.info(f"重排序模型切换成功: {reranker_model_name}")
                    else:
                        logger.warning(f"重排序模型切换失败: {reranker_model_name}")
                        # 如果切换失败，禁用重排序功能
                        self.config.use_reranker = False
                        return False
                elif self.config.use_reranker:
                    # 如果重排序器不存在但需要启用，创建新的重排序器
                    self.reranker = CrossEncoderReranker(
                        model_name=reranker_model_name,
                        use_fp16=self.config.enable_amp_if_beneficial,
                        backend=self.config.reranker_backend
                    )
                    if self.reranker.is_available():
                        logger.info(f"重排序器初始化成功: {reranker_model_name}")
                    else:
                        logger.warning("重排序器初始化失败，将禁用重排序功能")
                        self.config.use_reranker = False
                        return False
            
            logger.info("重排序配置更新完成")
            return True
            
        except Exception as e:
            logger.error(f"更新重排序配置时发生错误: {e}")
            return False
    
    def get_reranker_status(self) -> Dict[str, Any]:
        """获取重排序器状态信息"""
        status = {
            'enabled': self.config.use_reranker,
            'model_name': self.config.reranker_model_name,
            'top_n': self.config.reranker_top_n,
            'weight': self.config.reranker_weight,
            'backend': self.config.reranker_backend,
            'available': False,
            'model_info': None
        }
        
        if self.reranker:
            status['available'] = self.reranker.is_available()
            status['model_info'] = self.reranker.get_model_info()
        
        return status

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置V3引擎
    config = ZipperV3Config(
        precompute_doc_tokens=True,  # 启用全量token化
        incremental_update=True,      # 启用增量更新
        warmup_on_first_query=True,  # 首次查询时预热
        auto_build_index=True        # 自动构建索引
    )
    
    # 初始化引擎
    engine = AdvancedZipperQueryEngineV3(config)
    
    # 示例文档
    documents = {
        1: "人工智能是计算机科学的一个分支",
        2: "机器学习是人工智能的重要技术",
        3: "深度学习是机器学习的一个子集"
    }
    
    # 构建索引（只会执行一次）
    engine.build_document_index(documents)
    
    # 多次查询（不会重复构建索引）
    for i in range(3):
        results = engine.retrieve("什么是机器学习？")
        print(f"查询 {i+1}: 找到 {len(results)} 个结果")
    
    # 添加新文档（增量更新）
    new_docs = {4: "自然语言处理是AI的重要应用"}
    engine.add_documents(new_docs)
    
    # 查看索引统计
    stats = engine.get_index_stats()
    print(f"索引统计: {stats}")
    
    # 再次查询（使用更新后的索引）
    results = engine.retrieve("自然语言处理")
    print(f"更新后查询: 找到 {len(results)} 个结果")