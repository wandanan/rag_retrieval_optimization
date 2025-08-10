#!/usr/bin/env python3
"""
RAG系统调试脚本
用于诊断和修复RAG对比系统的问题
"""

import os
import sys
import logging
import json
import time
from typing import List, Dict, Any
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def create_test_document():
    """创建测试文档（小说片段）"""
    test_text = """
    易书元坐在船舱里，望着窗外的江水。这是一艘从元江县开往阔南山的客船，船上载着各色人等，有商人、有书生、也有像他这样的说书人。

    阔南山是岭东大地的一座名山，山势雄伟，云雾缭绕。传说山中有神仙居住，但易书元从不相信这些传说。他此行的目的是为了收集一些民间故事，为他的说书增添新的素材。

    船行至半途，易书元听到甲板上传来一阵喧哗声。他走出船舱，看到几个船客正围着一个樵夫模样的人。那樵夫背着柴禾，唱着山歌，声音洪亮而悠扬。

    "这位大哥，您唱的可是阔南山的山歌？"易书元上前问道。

    樵夫停下脚步，看了易书元一眼，笑道："正是。小兄弟也是去阔南山？"

    "是的，我是说书人，想去收集一些故事。"

    "说书人？"樵夫眼中闪过一丝异样的光芒，"那您可知道阔南山的传说？"

    易书元摇头："还请大哥指教。"

    樵夫神秘地笑了笑："阔南山有山神，每逢月圆之夜，山神会显灵。不过，只有有缘人才能见到。"

    易书元不以为然，但也没有反驳。他目送樵夫越走越远，心中却升起一丝疑惑。这个樵夫，似乎有些不同寻常。

    船终于到达了阔南山脚下。易书元下了船，沿着山路向上走去。山路崎岖，但风景优美。他一边走一边观察，希望能找到一些有趣的故事素材。

    走到半山腰时，易书元看到一座破旧的山神庙。庙门虚掩，里面传来阵阵香火味。他推门而入，发现庙内供奉着一尊山神像，神像前点着香烛。

    易书元正要离开，突然听到身后传来一个熟悉的声音："小兄弟，我们又见面了。"

    他回头一看，正是之前在船上遇到的那个樵夫。但此时，樵夫的形象似乎有些模糊，仿佛笼罩在一层薄雾中。

    "您...您是谁？"易书元有些紧张地问道。

    樵夫笑了笑："我是阔南山的山神，黄宏川。小兄弟，你我有缘，所以我才显现在你面前。"

    易书元震惊不已："山神？这怎么可能？"

    "世间万物，皆有灵性。阔南山连通地脉，与山同感，我能察觉到这座山处于某种风暴的中心。"黄宏川说道，"而你，易书元，你的名字在元江县的生死册上找不到，这说明你不是普通人。"

    易书元更加震惊："您怎么知道我的名字？"

    "因为我是山神，能感知到很多事情。"黄宏川说道，"你是一个仙道真人，虽然你自己可能还不知道。"

    易书元摇头："我只是一个普通的说书人。"

    "说书人？"黄宏川笑道，"那你可知道，你的说书能影响现实？你在《治灾记》中讲述的故事，正在岭东大地上真实发生。"

    易书元想起自己确实在说书时讲过《治灾记》，那是一个关于神仙治灾的故事。难道说，那些故事真的在现实中发生了？

    "这...这太不可思议了。"易书元说道。

    "世间本就有很多不可思议的事情。"黄宏川说道，"小兄弟，你我有缘，不如我们在这阔南山上对饮畅谈，如何？"

    易书元点头同意。于是，一个说书人和一个山神，在阔南山的山神庙中，开始了他们的对话。

    在元江县的阴司中，老城隍正在翻看生死册。他皱着眉头，因为他在生死册上找不到一个叫易书元的人的名字，但是县里的凡人却都记得这个人。

    "奇怪，这个易书元到底是什么人？"老城隍自言自语道。

    而在阔南山上，易书元正在听黄宏川讲述着岭东大地的秘密。原来，这个世界比易书元想象的要复杂得多，神仙、妖怪、凡人，都在这个大舞台上扮演着自己的角色。

    易书元第一次遇见超凡之事，就是在阔南山遇见黄宏川的那一刻。从此，他的生活将不再平凡，他将卷入一个更大的故事中。
    """
    return test_text.strip()

def test_vector_retriever():
    """测试向量检索器"""
    logger.info("=== 测试向量检索器 ===")
    
    try:
        from models.vector_retriever import VectorRetriever
        
        # 检查环境变量
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        if not hf_token:
            logger.warning("未设置HuggingFace token，将使用公开模型")
        
        # 尝试不同的模型
        model_options = [
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 多语言模型
            'sentence-transformers/all-MiniLM-L6-v2',  # 英文模型
        ]
        
        if hf_token:
            model_options.insert(0, 'BAAI/bge-large-zh-v1.5')  # 中文BGE模型
        
        vector_retriever = None
        for model_name in model_options:
            try:
                logger.info(f"尝试加载模型: {model_name}")
                vector_retriever = VectorRetriever(
                    model_name=model_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    backend='sentence-transformers',
                    hf_token=hf_token
                )
                logger.info(f"成功加载模型: {model_name}")
                break
            except Exception as e:
                logger.warning(f"模型 {model_name} 加载失败: {e}")
                continue
        
        if vector_retriever is None:
            logger.error("所有模型都无法加载")
            return False
        
        # 测试文档
        test_docs = [
            "易书元是一个说书人，擅长讲述各种故事。",
            "阔南山是岭东大地的一座名山，山势雄伟。",
            "黄宏川是阔南山的山神，能感知到很多事情。",
            "人工智能是计算机科学的一个分支。"
        ]
        
        # 构建索引
        vector_retriever.build_index(test_docs)
        
        # 测试查询
        query = "易书元是谁"
        doc_ids, scores = vector_retriever.search(query, top_k=2)
        
        logger.info(f"查询: {query}")
        logger.info(f"检索结果: {doc_ids}")
        logger.info(f"相似度分数: {scores}")
        
        return True
        
    except Exception as e:
        logger.error(f"向量检索器测试失败: {e}")
        return False

def test_attention_retriever():
    """测试注意力检索器"""
    logger.info("=== 测试注意力检索器 ===")
    
    try:
        from final_demo import FinalAttentionRetriever
        
        attention_retriever = FinalAttentionRetriever()
        
        # 测试查询类型分析
        test_queries = [
            "易书元是谁",
            "阔南山的山神可能是谁",
            "易书元在阔南山遇见了谁",
            "易书元第一次遇见超凡之事是什么"
        ]
        
        for query in test_queries:
            analysis = attention_retriever.analyze_query_type(query)
            logger.info(f"查询: {query}")
            logger.info(f"类型: {analysis['type_name']} (置信度: {analysis['confidence']:.2f})")
        
        # 测试注意力得分计算
        query = "易书元是谁"
        document = "易书元是一个说书人，擅长讲述各种故事，包括神话和现实事件。"
        
        # 测试不同的注意力计算方法
        enhanced_score = attention_retriever.compute_enhanced_attention_score(query, document)
        logger.info(f"增强注意力得分: {enhanced_score:.4f}")
        
        # 测试注意力模式
        pattern = attention_retriever.compute_attention_pattern(query, document)
        logger.info(f"注意力模式: {pattern['pattern_name']}")
        logger.info(f"重叠度: {pattern['overlap_ratio']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"注意力检索器测试失败: {e}")
        return False

def test_document_chunking():
    """测试文档切块"""
    logger.info("=== 测试文档切块 ===")
    
    test_text = create_test_document()
    logger.info(f"测试文档长度: {len(test_text)} 字符")
    
    # 测试不同的切块参数
    chunk_configs = [
        {"parent_size": 500, "parent_overlap": 100, "sub_size": 100, "sub_overlap": 20},
        {"parent_size": 1000, "parent_overlap": 200, "sub_size": 200, "sub_overlap": 50},
        {"parent_size": 1500, "parent_overlap": 300, "sub_size": 300, "sub_overlap": 75},
    ]
    
    for config in chunk_configs:
        logger.info(f"测试配置: {config}")
        
        # 父文档切块
        parent_chunks = split_text(test_text, config["parent_size"], config["parent_overlap"])
        logger.info(f"父文档块数: {len(parent_chunks)}")
        
        # 子文档切块
        sub_chunks = []
        sub_to_parent = []
        for i, parent_chunk in enumerate(parent_chunks):
            subs = split_text(parent_chunk, config["sub_size"], config["sub_overlap"])
            sub_chunks.extend(subs)
            sub_to_parent.extend([i] * len(subs))
        
        logger.info(f"子文档块数: {len(sub_chunks)}")
        
        # 检查切块质量
        avg_parent_len = np.mean([len(chunk) for chunk in parent_chunks])
        avg_sub_len = np.mean([len(chunk) for chunk in sub_chunks])
        logger.info(f"平均父块长度: {avg_parent_len:.1f}")
        logger.info(f"平均子块长度: {avg_sub_len:.1f}")
        
        # 检查是否包含关键信息
        key_terms = ["易书元", "阔南山", "黄宏川", "山神"]
        coverage = {}
        for term in key_terms:
            parent_coverage = sum(1 for chunk in parent_chunks if term in chunk)
            sub_coverage = sum(1 for chunk in sub_chunks if term in chunk)
            coverage[term] = {"parent": parent_coverage, "sub": sub_coverage}
        
        logger.info(f"关键信息覆盖: {coverage}")

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """文本切块函数"""
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

def test_llm_integration():
    """测试LLM集成"""
    logger.info("=== 测试LLM集成 ===")
    
    # 检查环境变量
    base_url = os.environ.get('LLM_BASE_URL', '')
    api_key = os.environ.get('LLM_API_KEY', '')
    model = os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')
    
    if not base_url or not api_key:
        logger.warning("未配置LLM环境变量，跳过LLM测试")
        logger.info("请设置以下环境变量:")
        logger.info("  LLM_BASE_URL: LLM服务的base URL")
        logger.info("  LLM_API_KEY: LLM服务的API key")
        logger.info("  LLM_MODEL: 模型名称")
        return False
    
    try:
        import requests
        
        # 测试LLM连接
        test_prompt = "请简单介绍一下自己"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": test_prompt}],
            "temperature": 0.2
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(f"{base_url}/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"LLM测试成功，回答: {answer[:100]}...")
            return True
        else:
            logger.error(f"LLM测试失败，状态码: {response.status_code}")
            logger.error(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"LLM测试异常: {e}")
        return False

def generate_test_queries():
    """生成测试查询"""
    logger.info("=== 生成测试查询 ===")
    
    test_queries = [
        "易书元是谁",
        "阔南山的山神可能是谁", 
        "易书元在阔南山遇见了谁",
        "易书元第一次遇见超凡之事是什么",
        "主角是谁",
        "易书元出场是什么场景",
        "易书元第一次出现时坐的什么交通工具",
        "易书元第一次出现时坐的什么船",
        "谁是易书元"
    ]
    
    logger.info("测试查询列表:")
    for i, query in enumerate(test_queries, 1):
        logger.info(f"  {i}. {query}")
    
    return test_queries

def main():
    """主函数"""
    logger.info("开始RAG系统调试...")
    
    # 1. 测试向量检索器
    vector_ok = test_vector_retriever()
    
    # 2. 测试注意力检索器
    attention_ok = test_attention_retriever()
    
    # 3. 测试文档切块
    test_document_chunking()
    
    # 4. 测试LLM集成
    llm_ok = test_llm_integration()
    
    # 5. 生成测试查询
    test_queries = generate_test_queries()
    
    # 6. 总结和建议
    logger.info("\n=== 调试总结 ===")
    logger.info(f"向量检索器: {'✅ 正常' if vector_ok else '❌ 异常'}")
    logger.info(f"注意力检索器: {'✅ 正常' if attention_ok else '❌ 异常'}")
    logger.info(f"LLM集成: {'✅ 正常' if llm_ok else '❌ 异常'}")
    
    logger.info("\n=== 建议 ===")
    if not vector_ok:
        logger.info("1. 检查HuggingFace token设置")
        logger.info("2. 确保网络连接正常")
        logger.info("3. 尝试使用不同的模型")
    
    if not llm_ok:
        logger.info("1. 设置LLM环境变量")
        logger.info("2. 检查LLM服务是否可用")
        logger.info("3. 验证API key是否正确")
    
    logger.info("4. 建议使用较小的文档切块参数（父块500字符，子块100字符）")
    logger.info("5. 确保上传的文档包含测试查询相关的信息")
    
    # 保存测试查询到文件
    with open('test_queries.json', 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
    logger.info("测试查询已保存到 test_queries.json")

if __name__ == "__main__":
    main() 