"""
数据处理工具函数
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import BertTokenizer
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, tokenizer_name: str = 'bert-base-uncased', max_length: int = 512):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def tokenize_query(self, query: str) -> Dict[str, torch.Tensor]:
        """对查询进行分词"""
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding
    
    def tokenize_document(self, document: str) -> Dict[str, torch.Tensor]:
        """对文档进行分词"""
        encoding = self.tokenizer(
            document,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """批量分词"""
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding
    
    def chunk_document(self, document: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """将长文档分块"""
        words = document.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

class DocumentIndex:
    """文档索引管理器"""
    
    def __init__(self):
        self.documents = []
        self.doc_ids = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None, 
                     metadata: Optional[List[Dict]] = None):
        """添加文档到索引"""
        self.documents.extend(documents)
        
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(self.documents) - len(documents), len(self.documents))]
        self.doc_ids.extend(doc_ids)
        
        if metadata is None:
            metadata = [{} for _ in documents]
        self.metadata.extend(metadata)
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """根据ID获取文档"""
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None
    
    def get_documents(self, doc_ids: List[str]) -> List[str]:
        """批量获取文档"""
        documents = []
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc is not None:
                documents.append(doc)
        return documents
    
    def get_all_documents(self) -> List[str]:
        """获取所有文档"""
        return self.documents.copy()
    
    def __len__(self):
        return len(self.documents)

class QueryProcessor:
    """查询处理器"""
    
    def __init__(self, processor: DataProcessor):
        self.processor = processor
    
    def process_query(self, query: str) -> Dict[str, torch.Tensor]:
        """处理查询"""
        # 清理查询
        query = self.clean_query(query)
        
        # 分词
        encoding = self.processor.tokenize_query(query)
        
        return encoding
    
    def clean_query(self, query: str) -> str:
        """清理查询文本"""
        # 移除多余空格
        query = ' '.join(query.split())
        
        # 转换为小写
        query = query.lower()
        
        return query
    
    def expand_query(self, query: str, expansion_terms: List[str]) -> str:
        """扩展查询"""
        expanded_query = query + " " + " ".join(expansion_terms)
        return expanded_query

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
    
    def create_batches(self, items: List, batch_size: Optional[int] = None) -> List[List]:
        """创建批次"""
        if batch_size is None:
            batch_size = self.batch_size
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def process_batches(self, items: List, process_func, batch_size: Optional[int] = None):
        """批量处理"""
        batches = self.create_batches(items, batch_size)
        results = []
        
        for batch in batches:
            batch_result = process_func(batch)
            results.extend(batch_result)
        
        return results 