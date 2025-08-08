# RAG检索优化项目

## 项目概述

本项目实现了三种创新的RAG检索优化方案，旨在解决传统RAG系统在检索精度和效率方面的局限性。

## 三种设想

### 设想1：向量化基础上的注意力计算
- 在传统向量检索基础上增加注意力机制
- 实现快速筛选 + 精确匹配的两阶段检索
- 平衡效率和精度的混合方案

### 设想2：拉链查询器
- 基于注意力机制的纯检索方案
- 避免向量检索的局限性
- 支持百万级文档的高效检索

### 设想3：Mamba机制检索器
- 利用Mamba的状态空间模型
- 线性复杂度的高效检索
- 选择性机制提供智能记忆

## 项目结构

```
rag_retrieval_optimization/
├── requirements.txt          # 项目依赖
├── README.md                # 项目说明
├── config/                  # 配置文件
├── models/                  # 模型实现
├── data/                    # 数据处理
├── utils/                   # 工具函数
├── experiments/             # 实验脚本
└── results/                 # 结果输出
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行实验：
```bash
python experiments/run_attention_retrieval.py
```

## 实验结果

详细的实验结果和性能分析请参考 `results/` 目录。 