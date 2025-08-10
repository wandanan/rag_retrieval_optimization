// 全局状态
let isIndexBuilt = false;
let history = [];

// 配置项列表
const CONFIG_KEYS = {
  // 文档切割配置
  PARENT_CHUNK_SIZE: 'parent_chunk_size',
  PARENT_OVERLAP: 'parent_overlap', 
  SUB_CHUNK_SIZE: 'sub_chunk_size',
  SUB_OVERLAP: 'sub_overlap',
  
  // LLM配置
  BASE_URL: 'base_url',
  MODEL: 'model',
  API_KEY: 'api_key',
  PROMPT: 'prompt',
  
  // 历史记录
  HISTORY: 'history'
};

// 默认配置
const DEFAULT_CONFIG = {
  [CONFIG_KEYS.PARENT_CHUNK_SIZE]: '1000',
  [CONFIG_KEYS.PARENT_OVERLAP]: '200',
  [CONFIG_KEYS.SUB_CHUNK_SIZE]: '200',
  [CONFIG_KEYS.SUB_OVERLAP]: '50',
  [CONFIG_KEYS.BASE_URL]: '',
  [CONFIG_KEYS.MODEL]: '',
  [CONFIG_KEYS.API_KEY]: '',
  [CONFIG_KEYS.PROMPT]: '',
  [CONFIG_KEYS.HISTORY]: 'history'
};

// V3引擎默认配置
const V3_DEFAULT_CONFIG = {
  'encoder_backend': 'bge',
  'bge_model_path': 'models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620',
  'hf_model_name': '',
  'embedding_dim': '512',
  'bm25_weight': '1.0',
  'colbert_weight': '1.5',
  'num_heads': '8',
  'context_influence': '0.3',
  'length_penalty_alpha': '0.05',
  'context_memory_decay': '0.8',
  'bm25_top_n': '100',
  'final_top_k': '10',
  'encode_batch_size': '64',
  'max_length': '256',
  'use_hybrid_search': true,
  'use_multi_head': true,
  'use_length_penalty': true,
  'use_stateful_reranking': true,
  'precompute_doc_tokens': false,
  'enable_amp_if_beneficial': true,
  'include_contexts': false,
  // 重排序默认配置
  'use_reranker': true,
  'reranker_model_name': 'BAAI/bge-reranker-large',
  'reranker_top_n': '50',
  'reranker_weight': '1.5',
  'reranker_backend': 'auto'
};

// DOM 元素
const elements = {
  uploadForm: document.getElementById('uploadForm'),
  fileInput: document.getElementById('fileInput'),
  uploadStatus: document.getElementById('uploadStatus'),
  question: document.getElementById('question'),
  askBtn: document.getElementById('askBtn'),
  v3Answer: document.getElementById('v3Answer'),
  v3Ctx: document.getElementById('v3Ctx'),
  v3Status: document.getElementById('v3Status'),
  v3Metrics: document.getElementById('v3Metrics'),
  historyList: document.getElementById('historyList'),
  clearConfigBtn: document.getElementById('clearConfigBtn'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn')
};

// 批量测试相关元素
const batchTestElements = {
  batchTestForm: document.getElementById('batchTestForm'),
  testFileInput: document.getElementById('testFileInput'),
  batchTestStatus: document.getElementById('batchTestStatus'),
  testProgressCard: document.getElementById('testProgressCard'),
  progressFill: document.getElementById('progressFill'),
  progressText: document.getElementById('progressText'),
  testSummary: document.getElementById('testSummary'),
  resultsList: document.getElementById('resultsList'),
  refreshResultsBtn: document.getElementById('refreshResultsBtn')
};

// 配置管理
function saveConfig() {
  const config = {};
  
  // 只保存实际存在的元素
  const configElements = {
    [CONFIG_KEYS.PARENT_CHUNK_SIZE]: 'parent_chunk_size',
    [CONFIG_KEYS.PARENT_OVERLAP]: 'parent_overlap',
    [CONFIG_KEYS.SUB_CHUNK_SIZE]: 'sub_chunk_size',
    [CONFIG_KEYS.SUB_OVERLAP]: 'sub_overlap',
    [CONFIG_KEYS.BASE_URL]: 'base_url',
    [CONFIG_KEYS.MODEL]: 'model',
    [CONFIG_KEYS.API_KEY]: 'api_key',
    [CONFIG_KEYS.PROMPT]: 'prompt'
  };
  
  // 安全地获取元素值
  Object.entries(configElements).forEach(([key, elementId]) => {
    const element = document.getElementById(elementId);
    if (element) {
      config[key] = element.value;
    }
  });
  
  // 保存V3引擎配置
  const v3ConfigElements = [
    'encoder_backend', 'bge_model_path', 'hf_model_name', 'embedding_dim',
    'bm25_weight', 'colbert_weight', 'num_heads', 'context_influence',
    'length_penalty_alpha', 'context_memory_decay', 'bm25_top_n', 'final_top_k',
    'encode_batch_size', 'max_length', 'use_hybrid_search', 'use_multi_head',
    'use_length_penalty', 'use_stateful_reranking', 'precompute_doc_tokens',
    'enable_amp_if_beneficial', 'include_contexts',
    // 新增重排序配置
    'use_reranker', 'reranker_model_name', 'reranker_top_n', 'reranker_weight', 'reranker_backend'
  ];
  
  v3ConfigElements.forEach(elementId => {
    const element = document.getElementById(elementId);
    if (element) {
      if (element.type === 'checkbox') {
        config[elementId] = element.checked;
      } else {
        config[elementId] = element.value;
      }
    }
  });
  
  localStorage.setItem('rag_config', JSON.stringify(config));
  console.log('配置已保存');
}

function loadConfig() {
  try {
    const saved = localStorage.getItem('rag_config');
    if (saved) {
      const config = JSON.parse(saved);
      
      // 恢复配置到表单
      Object.keys(config).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
          if (element.type === 'checkbox') {
            element.checked = config[key];
          } else {
            element.value = config[key];
          }
        }
      });
      
      console.log('配置已恢复');
    } else {
      // 如果没有保存的配置，使用默认值
      Object.keys(DEFAULT_CONFIG).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
          element.value = DEFAULT_CONFIG[key];
        }
      });
    }
  } catch (error) {
    console.warn('恢复配置失败:', error);
    // 出错时使用默认配置
    Object.keys(DEFAULT_CONFIG).forEach(key => {
      const element = document.getElementById(key);
      if (element) {
        element.value = DEFAULT_CONFIG[key];
      }
    });
  }
}

function clearConfig() {
  if (confirm('确定要清除所有配置吗？这将重置所有设置到默认值。')) {
    localStorage.removeItem('rag_config');
    
    // 重置到默认值
    Object.keys(DEFAULT_CONFIG).forEach(key => {
      const element = document.getElementById(key);
      if (element) {
        element.value = DEFAULT_CONFIG[key];
      }
    });
    
    // 重置V3引擎配置到默认值
    const v3Defaults = {
      'encoder_backend': 'bge',
      'bge_model_path': 'models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620',
      'hf_model_name': '',
      'embedding_dim': '512',
      'bm25_weight': '1.0',
      'colbert_weight': '1.5',
      'num_heads': '8',
      'context_influence': '0.3',
      'length_penalty_alpha': '0.05',
      'context_memory_decay': '0.8',
      'bm25_top_n': '100',
      'final_top_k': '10',
      'encode_batch_size': '64',
      'max_length': '256',
      'use_hybrid_search': true,
      'use_multi_head': true,
      'use_length_penalty': true,
      'use_stateful_reranking': true,
      'precompute_doc_tokens': false,
      'enable_amp_if_beneficial': true,
      'include_contexts': false
    };
    
    Object.entries(v3Defaults).forEach(([elementId, defaultValue]) => {
      const element = document.getElementById(elementId);
      if (element) {
        if (element.type === 'checkbox') {
          element.checked = defaultValue;
        } else {
          element.value = defaultValue;
        }
      }
    });
    
    alert('配置已清除');
  }
}

function saveHistory() {
  try {
    localStorage.setItem(CONFIG_KEYS.HISTORY, JSON.stringify(history));
  } catch (error) {
    console.warn('保存历史记录失败:', error);
  }
}

function loadHistory() {
  try {
    const saved = localStorage.getItem(CONFIG_KEYS.HISTORY);
    if (saved) {
      history = JSON.parse(saved);
      renderHistory();
      console.log('历史记录已恢复');
    }
  } catch (error) {
    console.warn('恢复历史记录失败:', error);
  }
}

function clearHistory() {
  if (confirm('确定要清除所有历史记录吗？')) {
    history = [];
    localStorage.removeItem(CONFIG_KEYS.HISTORY);
    renderHistory();
    alert('历史记录已清除');
  }
}

// 状态管理
function updateStatus(element, status, type = 'default') {
  element.textContent = status;
  element.className = `status-indicator ${type}`;
}

function setLoading(loading) {
  elements.askBtn.disabled = loading;
  elements.askBtn.textContent = loading ? '处理中...' : '同时提问';
}

// 文件上传
async function uploadFile(e) {
  e.preventDefault();
  
  if (!elements.fileInput.files || elements.fileInput.files.length === 0) {
    alert('请选择txt文件');
    return;
  }

  const formData = new FormData();
  formData.append('file', elements.fileInput.files[0]);
  formData.append('parent_chunk_size', document.getElementById('parent_chunk_size').value || '1000');
  formData.append('parent_overlap', document.getElementById('parent_overlap').value || '200');
  formData.append('sub_chunk_size', document.getElementById('sub_chunk_size').value || '200');
  formData.append('sub_overlap', document.getElementById('sub_overlap').value || '50');

  elements.uploadStatus.textContent = '上传中...';
  elements.uploadStatus.className = 'status';

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (!res.ok) throw new Error(data.detail || '上传失败');
    
    isIndexBuilt = true;
    elements.uploadStatus.textContent = `✅ 索引已构建：父块 ${data.num_parents}，子块 ${data.num_subs}`;
    elements.uploadStatus.className = 'status success';
    
    // 更新状态指示器
    updateStatus(elements.v3Status, '就绪', 'success');
    
  } catch (err) {
    elements.uploadStatus.textContent = `❌ 失败：${err.message}`;
    elements.uploadStatus.className = 'status error';
  }
}

// 渲染上下文
function renderContexts(container, contexts) {
  container.innerHTML = '';
  
  if (!contexts || contexts.length === 0) {
    container.innerHTML = '<div class="context-item"><div class="context-content">无相关上下文</div></div>';
    return;
  }

  contexts.forEach((ctx, i) => {
    const item = document.createElement('div');
    item.className = 'context-item';
    
    const header = document.createElement('div');
    header.className = 'context-header';
    
    // 构建头部信息，适应V3引擎的上下文格式
    let headerText = `片段${i + 1}`;
    if (ctx.parent_id) headerText += ` (${ctx.parent_id})`;
    if (ctx.vector_score !== undefined) headerText += ` | 向量分: ${ctx.vector_score.toFixed(4)}`;
    if (ctx.attention_score !== undefined) headerText += ` | 注意力分: ${ctx.attention_score.toFixed(4)}`;
    if (ctx.bm25_score !== undefined) headerText += ` | BM25分: ${ctx.bm25_score.toFixed(4)}`;
    if (ctx.colbert_score !== undefined) headerText += ` | ColBERT分: ${ctx.colbert_score.toFixed(4)}`;
    if (ctx.final_score !== undefined) headerText += ` | 综合分: ${ctx.final_score.toFixed(4)}`;
    
    header.textContent = headerText;
    
    const content = document.createElement('div');
    content.className = 'context-content';
    content.textContent = ctx.content;
    
    item.appendChild(header);
    item.appendChild(content);
    container.appendChild(item);
  });
}

// 同时提问
async function askQuestion() {
  const question = elements.question.value.trim();
  
  if (!question) {
    alert('请输入问题');
    return;
  }

  if (!isIndexBuilt) {
    alert('请先上传文档并构建索引');
    return;
  }

  // 获取V3引擎配置
  const encoder_backend = document.getElementById('encoder_backend').value;
  const bge_model_path = document.getElementById('bge_model_path').value;
  const hf_model_name = document.getElementById('hf_model_name').value;
  const embedding_dim = Number(document.getElementById('embedding_dim').value);
  const bm25_weight = Number(document.getElementById('bm25_weight').value);
  const colbert_weight = Number(document.getElementById('colbert_weight').value);
  const num_heads = Number(document.getElementById('num_heads').value);
  const context_influence = Number(document.getElementById('context_influence').value);
  const length_penalty_alpha = Number(document.getElementById('length_penalty_alpha').value);
  const context_memory_decay = Number(document.getElementById('context_memory_decay').value);
  const bm25_top_n = Number(document.getElementById('bm25_top_n').value);
  const final_top_k = Number(document.getElementById('final_top_k').value);
  const encode_batch_size = Number(document.getElementById('encode_batch_size').value);
  const max_length = Number(document.getElementById('max_length').value);
  
  // 获取重排序配置
  const use_reranker = document.getElementById('use_reranker').checked;
  const reranker_model_name = document.getElementById('reranker_model_name').value;
  const reranker_top_n = Number(document.getElementById('reranker_top_n').value);
  const reranker_weight = Number(document.getElementById('reranker_weight').value);
  const reranker_backend = document.getElementById('reranker_backend').value;
  
  // 获取功能开关配置
  const use_hybrid_search = document.getElementById('use_hybrid_search').checked;
  const use_multi_head = document.getElementById('use_multi_head').checked;
  const use_length_penalty = document.getElementById('use_length_penalty').checked;
  const use_stateful_reranking = document.getElementById('use_stateful_reranking').checked;
  const precompute_doc_tokens = document.getElementById('precompute_doc_tokens').checked;
  const enable_amp_if_beneficial = document.getElementById('enable_amp_if_beneficial').checked;
  
  // 获取LLM配置
  const base_url = document.getElementById('base_url').value.trim();
  const model = document.getElementById('model').value.trim();
  const api_key = document.getElementById('api_key').value.trim();
  const prompt = document.getElementById('prompt').value.trim();
  const top_k_parents = 4; // 使用固定值，因为HTML中没有这个元素

  // 设置加载状态
  setLoading(true);
  updateStatus(elements.v3Status, '处理中...', 'loading');
  
  elements.v3Answer.textContent = '正在生成答案...';
  elements.v3Ctx.innerHTML = '';
  elements.v3Metrics.innerHTML = '';

  // 根据编码后端选择性地设置模型配置
  const modelConfig = {};
  if (encoder_backend === 'bge') {
    modelConfig.bge_model_path = bge_model_path;
    modelConfig.hf_model_name = '';
  } else if (encoder_backend === 'hf') {
    modelConfig.bge_model_path = '';
    modelConfig.hf_model_name = hf_model_name;
  }

  const payload = {
    question,
    top_k_parents,
    top_k_sub: Math.max(50, top_k_parents * 20),
    prompt,
    llm: { base_url, model, api_key, temperature: 0.2 },
    v3_config: {
      encoder_backend,
      ...modelConfig,
      embedding_dim,
      bm25_weight,
      colbert_weight,
      num_heads,
      context_influence,
      length_penalty_alpha,
      context_memory_decay,
      bm25_top_n,
      final_top_k,
      encode_batch_size,
      max_length,
      use_reranker,
      reranker_model_name,
      reranker_top_n,
      reranker_weight,
      reranker_backend,
      use_hybrid_search,
      use_multi_head,
      use_length_penalty,
      use_stateful_reranking,
      precompute_doc_tokens,
      enable_amp_if_beneficial
    }
  };

  try {
    const res = await fetch('/api/v3_query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    if (!res.ok) throw new Error(data.detail || '请求失败');

    // 更新答案
    elements.v3Answer.textContent = data.answer || '未获取到答案';

    // 渲染上下文
    renderContexts(elements.v3Ctx, data.contexts || []);

    // 渲染指标
    if (data.metrics) {
      elements.v3Metrics.innerHTML = `
        <div class="metric-item">
          <span class="metric-label">检索时间:</span>
          <span class="metric-value">${data.metrics.retrieval_time?.toFixed(3) || 'N/A'}s</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">生成时间:</span>
          <span class="metric-value">${data.metrics.generation_time?.toFixed(3) || 'N/A'}s</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">总时间:</span>
          <span class="metric-value">${data.metrics.total_time?.toFixed(3) || 'N/A'}s</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">候选数量:</span>
          <span class="metric-value">${data.metrics.num_candidates || 'N/A'}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">最高分数:</span>
          <span class="metric-value">${data.metrics.top_score?.toFixed(3) || 'N/A'}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">重排序状态:</span>
          <span class="metric-value">${data.metrics.engine_config?.use_reranker ? '启用' : '禁用'}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">重排序模型:</span>
          <span class="metric-value">${data.metrics.engine_config?.reranker_model_name || 'N/A'}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">重排序后端:</span>
          <span class="metric-value">${data.metrics.engine_config?.reranker_backend || 'N/A'}</span>
        </div>
      `;
    }

    // 更新状态
    updateStatus(elements.v3Status, '完成', 'success');

    // 添加到历史记录
    addToHistory(question, data);

  } catch (err) {
    const errorMsg = `❌ 失败：${err.message}`;
    elements.v3Answer.textContent = errorMsg;
    
    updateStatus(elements.v3Status, '错误', 'error');
  } finally {
    setLoading(false);
  }
}

// 添加到历史记录
function addToHistory(question, result) {
  const historyItem = {
    id: Date.now(),
    question,
    answer: result.answer || '无答案',
    contexts: result.contexts || [],
    metrics: result.metrics || {},
    timestamp: new Date().toLocaleString()
  };

  history.unshift(historyItem);
  if (history.length > 10) history.pop(); // 保留最近10条
  
  renderHistory();
  saveHistory(); // 保存到localStorage
}

// 渲染历史记录
function renderHistory() {
  if (history.length === 0) {
    elements.historyList.innerHTML = '<div style="text-align: center; color: var(--muted); padding: 20px;">暂无历史记录</div>';
    return;
  }

  elements.historyList.innerHTML = history.map(item => `
    <div class="history-item">
      <div class="history-question">Q: ${item.question}</div>
      <div class="history-answer">
        <strong>🚀 V3引擎答案:</strong><br>
        ${item.answer}
      </div>
      <div style="font-size: 12px; color: var(--muted); margin-top: 8px;">
        ${item.timestamp}
      </div>
    </div>
  `).join('');
}

// 键盘事件
function handleKeyPress(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
}

// 配置变更事件
function handleConfigChange() {
  saveConfig();
}

// 模型输入切换函数
function toggleModelInputs() {
  const encoderBackend = document.getElementById('encoder_backend').value;
  const bgeGroup = document.getElementById('bge_model_group');
  const hfGroup = document.getElementById('hf_model_group');
  
  if (encoderBackend === 'bge') {
    bgeGroup.style.display = 'block';
    hfGroup.style.display = 'none';
    // 清空HF模型名
    document.getElementById('hf_model_name').value = '';
  } else {
    bgeGroup.style.display = 'none';
    hfGroup.style.display = 'block';
    // 清空BGE模型路径
    document.getElementById('bge_model_path').value = '';
  }
}

// 页面加载时初始化模型输入显示
document.addEventListener('DOMContentLoaded', function() {
  toggleModelInputs();
  
  // 设置重排序模型的默认值
  const rerankerModelInput = document.getElementById('reranker_model_name');
  if (rerankerModelInput && !rerankerModelInput.value.trim()) {
    rerankerModelInput.value = 'BAAI/bge-reranker-large';
  }
});

// 预设配置应用函数
function applyPreset() {
  const presetSelect = document.getElementById('preset_config');
  const selectedPreset = presetSelect.value;
  
  if (!selectedPreset) return; // 自定义配置，不做任何更改
  
  const presets = {
    balanced: {
      encoder_backend: 'bge',
      bm25_weight: 1.0,
      colbert_weight: 1.5,
      num_heads: 8,
      context_influence: 0.3,
      length_penalty_alpha: 0.05,
      context_memory_decay: 0.8,
      bm25_top_n: 100,
      final_top_k: 10,
      encode_batch_size: 64,
      max_length: 256,
      // 新增重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 50,
      reranker_weight: 1.5,
      reranker_backend: "auto"
    },
    precision: {
      encoder_backend: 'bge',
      bm25_weight: 0.8,
      colbert_weight: 2.0,
      num_heads: 12,
      context_influence: 0.5,
      length_penalty_alpha: 0.1,
      context_memory_decay: 0.9,
      bm25_top_n: 150,
      final_top_k: 15,
      encode_batch_size: 32,
      max_length: 384,
      // 新增重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 100,
      reranker_weight: 2.0,
      reranker_backend: "auto"
    },
    speed: {
      encoder_backend: 'bge',
      bm25_weight: 1.2,
      colbert_weight: 1.0,
      num_heads: 4,
      context_influence: 0.2,
      length_penalty_alpha: 0.02,
      context_memory_decay: 0.7,
      bm25_top_n: 50,
      final_top_k: 5,
      encode_batch_size: 128,
      max_length: 192,
      // 新增重排序配置
      use_reranker: false,
      reranker_model_name: "",
      reranker_top_n: 30,
      reranker_weight: 1.0,
      reranker_backend: "auto"
    },
    conversational: {
      encoder_backend: 'bge',
      bm25_weight: 0.9,
      colbert_weight: 1.8,
      num_heads: 10,
      context_influence: 0.4,
      length_penalty_alpha: 0.03,
      context_memory_decay: 0.85,
      bm25_top_n: 120,
      final_top_k: 12,
      encode_batch_size: 48,
      max_length: 320,
      // 新增重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 80,
      reranker_weight: 1.8,
      reranker_backend: "auto"
    },
    hf_optimized: {
      encoder_backend: 'hf',
      bm25_weight: 1.1,
      colbert_weight: 1.6,
      num_heads: 6,
      context_influence: 0.25,
      length_penalty_alpha: 0.04,
      context_memory_decay: 0.75,
      bm25_top_n: 80,
      final_top_k: 8,
      encode_batch_size: 96,
      max_length: 256,
      // 新增重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 60,
      reranker_weight: 1.6,
      reranker_backend: "auto"
    }
  };
  
  const preset = presets[selectedPreset];
  if (preset) {
    Object.keys(preset).forEach(key => {
      const element = document.getElementById(key);
      if (element) {
        if (element.type === 'checkbox') {
          element.checked = preset[key];
        } else {
          element.value = preset[key];
        }
      }
    });
    
    // 更新UI状态
    toggleModelInputs();
    
    // 保存配置
    saveConfig();
    
    // 显示提示
    const status = document.getElementById('batchTestStatus');
    if (status) {
      status.textContent = `✅ 已应用${presetSelect.options[presetSelect.selectedIndex].text}`;
      status.className = 'status success';
      setTimeout(() => {
        status.textContent = '';
        status.className = 'status';
      }, 2000);
    }
  }
}

// 验证并显示配置
function validateAndShowConfig() {
  const config = {};
  const errors = [];
  
  // 收集所有配置
  const allConfigElements = [
    'parent_chunk_size', 'parent_overlap', 'sub_chunk_size', 'sub_overlap',
    'base_url', 'model', 'api_key', 'prompt',
    'encoder_backend', 'bge_model_path', 'hf_model_name', 'embedding_dim',
    'bm25_weight', 'colbert_weight', 'num_heads', 'context_influence',
    'length_penalty_alpha', 'context_memory_decay', 'bm25_top_n', 'final_top_k',
    'encode_batch_size', 'max_length', 'use_hybrid_search', 'use_multi_head',
    'use_length_penalty', 'use_stateful_reranking', 'precompute_doc_tokens',
    'enable_amp_if_beneficial', 'include_contexts',
    // 新增重排序配置
    'use_reranker', 'reranker_model_name', 'reranker_top_n', 'reranker_weight', 'reranker_backend'
  ];
  
  allConfigElements.forEach(elementId => {
    const element = document.getElementById(elementId);
    if (element) {
      if (element.type === 'checkbox') {
        config[elementId] = element.checked;
      } else {
        config[elementId] = element.value;
      }
    }
  });
  
  // 验证模型配置
  const encoderBackend = config.encoder_backend;
  if (encoderBackend === 'bge' && !config.bge_model_path.trim()) {
    errors.push('❌ BGE模型路径不能为空');
  } else if (encoderBackend === 'hf' && !config.hf_model_name.trim()) {
    errors.push('❌ HF模型名不能为空');
  }
  
  // 验证重排序配置
  if (config.use_reranker && !config.reranker_model_name.trim()) {
    errors.push('❌ 重排序模型名不能为空');
  }
  
  // 显示配置对话框
  const dialog = document.createElement('div');
  dialog.className = 'config-dialog';
  
  const errorHtml = errors.length > 0 ? 
    `<div class="config-errors" style="color: #ef4444; margin-bottom: 15px; padding: 10px; background: #fef2f2; border-radius: 5px;">
      <strong>配置验证错误：</strong><br>
      ${errors.join('<br>')}
    </div>` : '';
  
  dialog.innerHTML = `
    <div class="config-dialog-content">
      <div class="config-dialog-header">
        <h3>📋 当前配置</h3>
        <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">&times;</button>
      </div>
      <div class="config-dialog-body">
        ${errorHtml}
        <pre>${JSON.stringify(config, null, 2)}</pre>
      </div>
      <div class="config-dialog-footer">
        <button class="btn btn-primary" onclick="this.parentElement.parentElement.parentElement.remove()">关闭</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(dialog);
  
  // 点击背景关闭对话框
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) {
      dialog.remove();
    }
  });
}

// 批量测试功能
async function handleBatchTest(e) {
  e.preventDefault();
  
  // 重新获取DOM元素（确保在函数执行时能访问到）
  const batchTestElements = {
    batchTestForm: document.getElementById('batchTestForm'),
    testFileInput: document.getElementById('testFileInput'),
    batchTestStatus: document.getElementById('batchTestStatus'),
    testProgressCard: document.getElementById('testProgressCard'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    testSummary: document.getElementById('testSummary'),
    resultsList: document.getElementById('resultsList'),
    refreshResultsBtn: document.getElementById('refreshResultsBtn')
  };
  
  // 检查必要的DOM元素是否存在
  if (!batchTestElements.testFileInput || !batchTestElements.batchTestStatus) {
    console.error('批量测试相关DOM元素未找到');
    return;
  }
  
  if (!isIndexBuilt) {
    updateStatus(batchTestElements.batchTestStatus, '请先上传文档并构建索引', 'error');
    return;
  }
  
  const testFile = batchTestElements.testFileInput.files[0];
  if (!testFile) {
    updateStatus(batchTestElements.batchTestStatus, '请选择测试文件', 'error');
    return;
  }
  
  // 检查进度相关元素
  if (!batchTestElements.testProgressCard || !batchTestElements.progressFill || !batchTestElements.progressText || !batchTestElements.testSummary) {
    console.error('进度显示相关DOM元素未找到');
    return;
  }
  
  // 显示进度卡片
  batchTestElements.testProgressCard.style.display = 'block';
  batchTestElements.progressFill.style.width = '0%';
  batchTestElements.progressText.textContent = '准备中...';
  batchTestElements.testSummary.innerHTML = '';
  
  updateStatus(batchTestElements.batchTestStatus, '开始批量测试...', 'info');
  
  try {
    // 构建表单数据
    const formData = new FormData();
    formData.append('test_file', testFile);
    
    // 获取当前配置
    const v3Config = {
      encoder_backend: document.getElementById('encoder_backend')?.value || 'bge',
      bge_model_path: document.getElementById('bge_model_path')?.value || '',
      hf_model_name: document.getElementById('hf_model_name')?.value || '',
      embedding_dim: parseInt(document.getElementById('embedding_dim')?.value || '512'),
      bm25_weight: parseFloat(document.getElementById('bm25_weight')?.value || '1.0'),
      colbert_weight: parseFloat(document.getElementById('colbert_weight')?.value || '1.5'),
      num_heads: parseInt(document.getElementById('num_heads')?.value || '8'),
      context_influence: parseFloat(document.getElementById('context_influence')?.value || '0.3'),
      final_top_k: parseInt(document.getElementById('final_top_k')?.value || '10'),
      length_penalty_alpha: parseFloat(document.getElementById('length_penalty_alpha')?.value || '0.05'),
      context_memory_decay: parseFloat(document.getElementById('context_memory_decay')?.value || '0.8'),
      bm25_top_n: parseInt(document.getElementById('bm25_top_n')?.value || '100'),
      encode_batch_size: parseInt(document.getElementById('encode_batch_size')?.value || '64'),
      max_length: parseInt(document.getElementById('max_length')?.value || '256'),
      use_hybrid_search: document.getElementById('use_hybrid_search')?.checked || true,
      use_multi_head: document.getElementById('use_multi_head')?.checked || true,
      use_length_penalty: document.getElementById('use_length_penalty')?.checked || true,
      use_stateful_reranking: document.getElementById('use_stateful_reranking')?.checked || true,
      precompute_doc_tokens: document.getElementById('precompute_doc_tokens')?.checked || false,
      enable_amp_if_beneficial: document.getElementById('enable_amp_if_beneficial')?.checked || true
    };
    
    const llmConfig = {
      base_url: document.getElementById('base_url')?.value || '',
      api_key: document.getElementById('api_key')?.value || '',
      model: document.getElementById('model')?.value || ''
    };
    
    const prompt = document.getElementById('prompt')?.value || '';
    
    // 获取是否包含召回调段的选项
    const includeContexts = document.getElementById('include_contexts')?.checked || false;
    
    formData.append('v3_config', JSON.stringify(v3Config));
    formData.append('llm_config', JSON.stringify(llmConfig));
    formData.append('prompt', prompt);
    formData.append('include_contexts', includeContexts);
    
    // 模拟进度更新
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 90) progress = 90;
      if (batchTestElements.progressFill) {
        batchTestElements.progressFill.style.width = progress + '%';
      }
      if (batchTestElements.progressText) {
        batchTestElements.progressText.textContent = `测试进行中... ${Math.round(progress)}%`;
      }
    }, 1000);
    
    // 发送请求
    const response = await fetch('/api/batch_test', {
      method: 'POST',
      body: formData
    });
    
    clearInterval(progressInterval);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // 完成进度
    if (batchTestElements.progressFill) {
      batchTestElements.progressFill.style.width = '100%';
    }
    if (batchTestElements.progressText) {
      batchTestElements.progressText.textContent = '测试完成！';
    }
    
    // 显示摘要
    if (batchTestElements.testSummary) {
      const summary = result.summary;
      batchTestElements.testSummary.innerHTML = `
        <h4>测试摘要</h4>
        <p><strong>总测试数：</strong>${summary.total_tests}</p>
        <p><strong>成功测试：</strong>${summary.successful_tests}</p>
        <p><strong>失败测试：</strong>${summary.failed_tests}</p>
        <p><strong>平均检索时间：</strong>${summary.average_retrieval_time.toFixed(3)}s</p>
        <p><strong>平均LLM延迟：</strong>${summary.average_llm_latency.toFixed(3)}s</p>
        <p><strong>结果文件：</strong>${summary.result_file}</p>
        <p><strong>结果配置：</strong><span style="color: ${summary.include_contexts ? '#28a745' : '#ffc107'}; font-weight: bold;">${summary.contexts_info || (summary.include_contexts ? '包含召回调段' : '仅包含AI回答')}</span></p>
        <hr>
        <h4>召回统计</h4>
        <p><strong>召回成功数：</strong>${summary.recall_success_count || 0}</p>
        <p><strong>召回失败数：</strong>${summary.recall_failure_count || 0}</p>
        <p><strong>召回成功率：</strong><span style="color: ${(summary.recall_success_rate || 0) >= 80 ? '#28a745' : (summary.recall_success_rate || 0) >= 60 ? '#ffc107' : '#dc3545'}; font-weight: bold;">${summary.recall_success_rate || 0}%</span></p>
      `;
    }
    
    updateStatus(batchTestElements.batchTestStatus, result.message, 'success');
    
    // 刷新结果列表
    await loadResultsList();
    
  } catch (error) {
    console.error('批量测试失败:', error);
    if (batchTestElements.progressFill) {
      batchTestElements.progressFill.style.width = '0%';
    }
    if (batchTestElements.progressText) {
      batchTestElements.progressText.textContent = '测试失败';
    }
    updateStatus(batchTestElements.batchTestStatus, `批量测试失败: ${error.message}`, 'error');
  }
}

// 加载结果列表
async function loadResultsList() {
  try {
    const response = await fetch('/api/list_results');
    const data = await response.json();
    
    // 重新获取DOM元素
    const resultsList = document.getElementById('resultsList');
    if (!resultsList) {
      console.error('结果列表容器未找到');
      return;
    }
    
    if (data.files.length === 0) {
      resultsList.innerHTML = '<div class="empty-state">暂无测试结果文件</div>';
      return;
    }
    
    const resultsHtml = data.files.map(file => {
      const fileSize = (file.size / 1024).toFixed(1);
      const modifiedDate = new Date(file.modified * 1000).toLocaleString('zh-CN');
      
      return `
        <div class="result-item">
          <div class="result-info">
            <div class="result-filename">${file.filename}</div>
            <div class="result-meta">大小: ${fileSize} KB | 修改时间: ${modifiedDate}</div>
          </div>
          <div class="result-actions">
            <button class="btn-download" onclick="downloadFile('${file.download_url}', '${file.filename}')">
              下载
            </button>
          </div>
        </div>
      `;
    }).join('');
    
    resultsList.innerHTML = resultsHtml;
    
  } catch (error) {
    console.error('加载结果列表失败:', error);
    const resultsList = document.getElementById('resultsList');
    if (resultsList) {
      resultsList.innerHTML = '<div class="error">加载失败</div>';
    }
  }
}

// 下载文件
function downloadFile(downloadUrl, filename) {
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// 刷新结果列表
async function refreshResultsList() {
  await loadResultsList();
}

// 事件监听
elements.uploadForm.addEventListener('submit', uploadFile);
elements.askBtn.addEventListener('click', askQuestion);
elements.question.addEventListener('keypress', handleKeyPress);
elements.clearConfigBtn.addEventListener('click', clearConfig);
elements.clearHistoryBtn.addEventListener('click', clearHistory);

// 为所有配置输入框添加变更监听
Object.values(CONFIG_KEYS).forEach(key => {
  if (key !== CONFIG_KEYS.HISTORY) {
    const element = document.getElementById(key);
    if (element) {
      element.addEventListener('change', handleConfigChange);
      element.addEventListener('input', handleConfigChange);
    }
  }
});

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  // 重新获取批量测试相关元素（确保DOM已加载）
  const batchTestElements = {
    batchTestForm: document.getElementById('batchTestForm'),
    testFileInput: document.getElementById('testFileInput'),
    batchTestStatus: document.getElementById('batchTestStatus'),
    testProgressCard: document.getElementById('testProgressCard'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    testSummary: document.getElementById('testSummary'),
    resultsList: document.getElementById('resultsList'),
    refreshResultsBtn: document.getElementById('refreshResultsBtn')
  };

  // 批量测试事件监听器
  if (batchTestElements.batchTestForm) {
    batchTestElements.batchTestForm.addEventListener('submit', handleBatchTest);
  }

  if (batchTestElements.refreshResultsBtn) {
    batchTestElements.refreshResultsBtn.addEventListener('click', refreshResultsList);
  }

  // 加载保存的配置
  loadConfig();
  loadHistory();
  
  // 检查服务状态
  fetch('/api/health')
    .then(res => res.json())
    .then(data => {
      if (data.index_built) {
        isIndexBuilt = true;
        updateStatus(elements.v3Status, '就绪', 'success');
        elements.uploadStatus.textContent = '✅ 索引已就绪';
        elements.uploadStatus.className = 'status success';
      }
    })
    .catch(() => {
      updateStatus(elements.v3Status, '未连接', 'error');
    });
  
  // 初始加载结果列表
  loadResultsList();
}); 