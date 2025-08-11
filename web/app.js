// 全局状态
let isIndexBuilt = false;
let history = [];
let modelStatus = 'online'; // 新增：模型状态管理
let currentV3Config = null; // 新增：保存当前使用的V3配置

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
  'encoder_backend': 'hf',  // 固定为HF
  'hf_model_name': 'BAAI/bge-small-zh-v1.5',   // HF模型名称
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

// 模型状态管理
function updateModelStatus(status) {
  modelStatus = status;
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.querySelector('.status-text');
  
  if (statusDot && statusText) {
    statusDot.className = `status-dot ${status}`;
    
    switch (status) {
      case 'online':
        statusText.textContent = '在线模型模式';
        break;
      case 'loading':
        statusText.textContent = '模型加载中...';
        break;
      case 'offline':
        statusText.textContent = '模型离线';
        break;
    }
  }
}

// 显示优化提示
function showOptimizationTip(message, type = 'info') {
  const tip = document.createElement('div');
  tip.className = `notification ${type}`;
  tip.textContent = message;
  
  document.body.appendChild(tip);
  
  setTimeout(() => {
    tip.remove();
  }, 5000);
}

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

// 显示当前V3配置信息
function showCurrentV3Config() {
  if (!currentV3Config) {
    console.log('当前没有保存的V3配置');
    return;
  }
  
  console.log('=== 当前V3引擎配置 ===');
  console.log('编码后端:', currentV3Config.encoder_backend);
  console.log('HF模型名称:', currentV3Config.hf_model_name);
  console.log('嵌入维度:', currentV3Config.embedding_dim);
  console.log('BM25权重:', currentV3Config.bm25_weight);
  console.log('ColBERT权重:', currentV3Config.colbert_weight);
  console.log('注意力头数:', currentV3Config.num_heads);
  console.log('上下文影响:', currentV3Config.context_influence);
  console.log('重排序模型:', currentV3Config.reranker_model_name);
  console.log('重排序权重:', currentV3Config.reranker_weight);
  console.log('========================');
}

function validateV3ConfigConsistency() {
  if (!currentV3Config) {
    return { consistent: false, message: '未找到保存的V3配置' };
  }
  
  // 检查关键配置项是否完整
  const requiredFields = ['hf_model_name', 'embedding_dim', 'encoder_backend'];
  for (const field of requiredFields) {
    if (!currentV3Config[field]) {
      return { consistent: false, message: `缺少关键配置项: ${field}` };
    }
  }
  
  // 检查配置值的合理性
  if (currentV3Config.embedding_dim < 128 || currentV3Config.embedding_dim > 1024) {
    return { consistent: false, message: `嵌入维度值不合理: ${currentV3Config.embedding_dim}` };
  }
  
  if (!currentV3Config.hf_model_name.includes('/')) {
    return { consistent: false, message: `模型名称格式不正确: ${currentV3Config.hf_model_name}` };
  }
  
  return { consistent: true, message: 'V3配置验证通过' };
}

// 配置管理
function saveConfig() {
  try {
    // 收集所有配置项
    const config = {};
    
    // 文档切割配置
    config[CONFIG_KEYS.PARENT_CHUNK_SIZE] = document.getElementById('parent_chunk_size').value;
    config[CONFIG_KEYS.PARENT_OVERLAP] = document.getElementById('parent_overlap').value;
    config[CONFIG_KEYS.SUB_CHUNK_SIZE] = document.getElementById('sub_chunk_size').value;
    config[CONFIG_KEYS.SUB_OVERLAP] = document.getElementById('sub_overlap').value;
    
    // LLM配置
    config[CONFIG_KEYS.BASE_URL] = document.getElementById('base_url').value;
    config[CONFIG_KEYS.MODEL] = document.getElementById('model').value;
    config[CONFIG_KEYS.API_KEY] = document.getElementById('api_key').value;
    config[CONFIG_KEYS.PROMPT] = document.getElementById('prompt').value;
    
    // 保存到localStorage
    Object.keys(config).forEach(key => {
      localStorage.setItem(key, config[key]);
    });
    
    // 保存历史记录
    localStorage.setItem(CONFIG_KEYS.HISTORY, JSON.stringify(history));
    
    showOptimizationTip('配置已保存到本地存储', 'success');
    
    // 新增：如果索引已构建，立即更新后台重排序配置
    if (isIndexBuilt) {
      updateBackendRerankerConfig();
    }
    
  } catch (error) {
    console.error('保存配置时发生错误:', error);
    showOptimizationTip('保存配置失败', 'error');
  }
}

function loadConfig() {
  try {
    // 加载基本配置
    Object.keys(CONFIG_KEYS).forEach(key => {
      const savedValue = localStorage.getItem(key);
      if (savedValue !== null) {
        const element = document.getElementById(CONFIG_KEYS[key]);
        if (element) {
          element.value = savedValue;
        }
      }
    });
    
    // 加载V3引擎配置
    const savedV3Config = localStorage.getItem('rag_config');
    if (savedV3Config) {
      try {
        const v3Config = JSON.parse(savedV3Config);
        
        // 应用V3配置到界面
        Object.keys(v3Config).forEach(key => {
          const element = document.getElementById(key);
          if (element) {
            if (element.type === 'checkbox') {
              element.checked = Boolean(v3Config[key]);
            } else {
              element.value = v3Config[key];
            }
          }
        });
        
        // 更新当前V3配置
        currentV3Config = { ...v3Config };
        
        console.log('V3配置已加载:', currentV3Config);
        
        // 新增：如果索引已构建，同步配置到后台
        if (isIndexBuilt) {
          setTimeout(() => {
            updateBackendRerankerConfig();
          }, 1000); // 延迟1秒执行，确保界面完全加载
        }
        
      } catch (parseError) {
        console.error('解析V3配置失败:', parseError);
      }
    }
    
    // 加载历史记录
    const savedHistory = localStorage.getItem(CONFIG_KEYS.HISTORY);
    if (savedHistory) {
      try {
        history = JSON.parse(savedHistory);
        renderHistory();
      } catch (parseError) {
        console.error('解析历史记录失败:', parseError);
      }
    }
    
    showOptimizationTip('配置已从本地存储加载', 'success');
    
  } catch (error) {
    console.error('加载配置时发生错误:', error);
    showOptimizationTip('加载配置失败', 'error');
  }
}

function clearConfig() {
  try {
    // 清除基本配置
    Object.keys(CONFIG_KEYS).forEach(key => {
      localStorage.removeItem(key);
      const element = document.getElementById(CONFIG_KEYS[key]);
      if (element) {
        element.value = DEFAULT_CONFIG[key] || '';
      }
    });
    
    // 清除V3引擎配置
    localStorage.removeItem('rag_config');
    
    // 重置V3引擎配置到默认值
    Object.keys(V3_DEFAULT_CONFIG).forEach(key => {
      const element = document.getElementById(key);
      if (element) {
        if (element.type === 'checkbox') {
          element.checked = V3_DEFAULT_CONFIG[key];
        } else {
          element.value = V3_DEFAULT_CONFIG[key];
        }
      }
    });
    
    // 更新当前V3配置
    currentV3Config = { ...V3_DEFAULT_CONFIG };
    
    // 清除历史记录
    history = [];
    localStorage.removeItem(CONFIG_KEYS.HISTORY);
    renderHistory();
    
    console.log('配置已清除，使用默认值');
    showOptimizationTip('配置已清除并重置为默认值', 'success');
    
    // 新增：如果索引已构建，同步默认配置到后台
    if (isIndexBuilt) {
      setTimeout(() => {
        updateBackendRerankerConfig();
      }, 1000); // 延迟1秒执行，确保界面完全加载
    }
    
  } catch (error) {
    console.error('清除配置时发生错误:', error);
    showOptimizationTip('清除配置失败', 'error');
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

  // 获取V3引擎配置
  const encoder_backend = document.getElementById('encoder_backend').value;
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

  // 构建V3配置
  const v3Config = {
    encoder_backend,
    hf_model_name,
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
  };

  // 保存当前V3配置，供查询时使用
  currentV3Config = { ...v3Config };
  
  // 保存V3配置到本地存储
  localStorage.setItem('rag_config', JSON.stringify(v3Config));
  
  // 显示上传配置信息
  console.log('上传文档使用的V3配置:', currentV3Config);
  console.log('配置已保存，查询时将使用完全一致的配置');
  
  // 显示当前配置到界面
  showCurrentV3Config();
  
  // 在界面上显示当前使用的配置
  showOptimizationTip(`已保存V3配置：模型=${currentV3Config.hf_model_name}, 维度=${currentV3Config.embedding_dim}`, 'success');

  const formData = new FormData();
  formData.append('file', elements.fileInput.files[0]);
  formData.append('parent_chunk_size', document.getElementById('parent_chunk_size').value || '1000');
  formData.append('parent_overlap', document.getElementById('parent_overlap').value || '200');
  formData.append('sub_chunk_size', document.getElementById('sub_chunk_size').value || '200');
  formData.append('sub_overlap', document.getElementById('sub_overlap').value || '50');
  
  // 添加V3引擎配置
  const v3ConfigToSend = {
    encoder_backend,
    hf_model_name,
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
  };
  
  // 调试：显示配置信息
  console.log('上传前的V3配置:', v3ConfigToSend);
  console.log('HF模型名称:', hf_model_name);
  
  formData.append('v3_config', JSON.stringify(v3ConfigToSend));

  elements.uploadStatus.textContent = '上传中...';
  elements.uploadStatus.className = 'status';
  
  // 更新模型状态为加载中
  updateModelStatus('loading');
  showOptimizationTip('正在构建索引，首次使用将自动下载模型...', 'info');

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (!res.ok) throw new Error(data.detail || '上传失败');
    
    isIndexBuilt = true;
    elements.uploadStatus.textContent = `✅ 索引已构建：父块 ${data.num_parents}，子块 ${data.num_subs}`;
    elements.uploadStatus.className = 'status success';
    
    // 更新状态指示器
    updateStatus(elements.v3Status, '就绪', 'success');
    
    // 恢复模型状态为在线
    updateModelStatus('online');
    showOptimizationTip('索引构建完成！系统已优化为使用HuggingFace在线模型', 'success');
    
  } catch (err) {
    elements.uploadStatus.textContent = `❌ 失败：${err.message}`;
    elements.uploadStatus.className = 'status error';
    
    // 更新模型状态为离线
    updateModelStatus('offline');
    showOptimizationTip('索引构建失败，请检查网络连接和模型配置', 'error');
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

  // 检查是否有保存的V3配置
  if (!currentV3Config) {
    alert('未找到V3配置信息，请重新上传文档');
    return;
  }
  
  // 验证V3配置一致性
  const configValidation = validateV3ConfigConsistency();
  if (!configValidation.consistent) {
    alert(`V3配置验证失败: ${configValidation.message}\n请重新上传文档以获取正确的配置`);
    return;
  }

  // 使用保存的V3配置，确保与上传文档时完全一致
  const v3Config = { ...currentV3Config };
  
  // 显示使用的配置信息
  console.log('查询时使用的V3配置:', v3Config);
  console.log('模型名称:', v3Config.hf_model_name);
  console.log('嵌入维度:', v3Config.embedding_dim);
  console.log('确保配置与上传文档时一致，避免重新初始化');

  // 设置加载状态
  setLoading(true);
  updateStatus(elements.v3Status, '处理中...', 'loading');
  
  // 更新模型状态为加载中
  updateModelStatus('loading');
  showOptimizationTip('正在使用V3引擎处理查询，模型推理中...', 'info');
  
  elements.v3Answer.textContent = '正在生成答案...';
  elements.v3Ctx.innerHTML = '';
  elements.v3Metrics.innerHTML = '';

  // 根据编码后端选择性地设置模型配置
  const modelConfig = {};
  if (v3Config.encoder_backend === 'hf') {
    modelConfig.hf_model_name = v3Config.hf_model_name;
  }

  const payload = {
    question,
    top_k_parents: 4, // 使用固定值，因为HTML中没有这个元素
    top_k_sub: Math.max(50, 4 * 20),
    prompt: document.getElementById('prompt').value.trim(),
    llm: { 
      base_url: document.getElementById('base_url').value.trim(), 
      model: document.getElementById('model').value.trim(), 
      api_key: document.getElementById('api_key').value.trim(), 
      temperature: 0.2 
    },
    v3_config: v3Config  // 使用完全一致的配置
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
    
    // 恢复模型状态为在线
    updateModelStatus('online');
    showOptimizationTip('查询处理完成！V3引擎表现优异', 'success');

    // 添加到历史记录
    addToHistory(question, data);

  } catch (err) {
    const errorMsg = `❌ 失败：${err.message}`;
    elements.v3Answer.textContent = errorMsg;
    
    updateStatus(elements.v3Status, '错误', 'error');
    
    // 更新模型状态为离线
    updateModelStatus('offline');
    showOptimizationTip('查询处理失败，请检查网络连接和模型配置', 'error');
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
  const hfGroup = document.getElementById('hf_model_group');
  
  if (encoderBackend === 'hf') {
    hfGroup.style.display = 'block';
  } else {
    hfGroup.style.display = 'none';
  }
}

// 页面加载时初始化模型输入显示
document.addEventListener('DOMContentLoaded', function() {
  // 强制显示HF模型输入
  const hfGroup = document.getElementById('hf_model_group');
  if (hfGroup) {
    hfGroup.style.display = 'block';
  }
  
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
  
  if (!selectedPreset) {
    showOptimizationTip('请选择一个预设配置', 'warning');
    return;
  }
  
  try {
    // 获取预设配置
    const presetConfig = getPresetConfig(selectedPreset);
    if (!presetConfig) {
      showOptimizationTip('预设配置不存在', 'error');
      return;
    }
    
    // 应用预设配置到界面
    Object.keys(presetConfig).forEach(key => {
      const element = document.getElementById(key);
      if (element) {
        if (element.type === 'checkbox') {
          element.checked = presetConfig[key];
        } else {
          element.value = presetConfig[key];
        }
      }
    });
    
    // 更新当前V3配置
    currentV3Config = { ...presetConfig };
    
    // 保存配置到本地存储
    localStorage.setItem('rag_config', JSON.stringify(presetConfig));
    
    // 显示成功提示
    const presetName = getPresetDisplayName(selectedPreset);
    showOptimizationTip(`预设配置"${presetName}"已应用并保存`, 'success');
    
    // 新增：如果索引已构建，同步预设配置到后台
    if (isIndexBuilt) {
      setTimeout(() => {
        updateBackendRerankerConfig();
      }, 1000); // 延迟1秒执行，确保界面完全加载
    }
    
    // 重置预设选择器
    presetSelect.value = '';
    
  } catch (error) {
    console.error('应用预设配置时发生错误:', error);
    showOptimizationTip('应用预设配置失败', 'error');
  }
}

// 新增：获取预设配置的显示名称
function getPresetDisplayName(presetKey) {
  const presetNames = {
    'balanced': '平衡模式',
    'precision': '精确模式',
    'speed': '快速模式',
    'conversational': '对话模式',
    'hf_optimized': 'HF优化模式'
  };
  return presetNames[presetKey] || presetKey;
}

// 验证并显示配置
function validateAndShowConfig() {
  const config = {};
  const errors = [];
  
  // 收集所有配置
  const allConfigElements = [
    'parent_chunk_size', 'parent_overlap', 'sub_chunk_size', 'sub_overlap',
    'base_url', 'model', 'api_key', 'prompt',
    'encoder_backend', 'hf_model_name', 'embedding_dim',
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
  if (encoderBackend === 'hf' && !config.hf_model_name.trim()) {
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
      encoder_backend: document.getElementById('encoder_backend')?.value || 'hf',
      hf_model_name: document.getElementById('hf_model_name')?.value || 'BAAI/bge-small-zh-v1.5',
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

// 重排序配置管理
async function updateRerankerConfig() {
  try {
    if (!isIndexBuilt) {
      showOptimizationTip('请先上传文档构建索引', 'warning');
      return;
    }
    
    // 获取重排序配置
    const use_reranker = document.getElementById('use_reranker').checked;
    const reranker_model_name = document.getElementById('reranker_model_name').value.trim();
    const reranker_top_n = Number(document.getElementById('reranker_top_n').value);
    const reranker_weight = Number(document.getElementById('reranker_weight').value);
    const reranker_backend = document.getElementById('reranker_backend').value;
    
    // 验证配置
    if (use_reranker && !reranker_model_name) {
      showOptimizationTip('启用重排序时必须指定模型名称', 'error');
      return;
    }
    
    if (reranker_top_n < 1 || reranker_top_n > 200) {
      showOptimizationTip('重排序候选数量必须在1-200之间', 'error');
      return;
    }
    
    if (reranker_weight < 0 || reranker_weight > 5) {
      showOptimizationTip('重排序权重必须在0-5之间', 'error');
      return;
    }
    
    // 显示加载状态
    const updateBtn = document.querySelector('.reranker-actions .btn-primary');
    const originalText = updateBtn.textContent;
    updateBtn.textContent = '🔄 更新中...';
    updateBtn.disabled = true;
    
    // 发送更新请求
    const response = await fetch('/api/v3/update_reranker', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        use_reranker,
        reranker_model_name,
        reranker_top_n,
        reranker_weight,
        reranker_backend
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      showOptimizationTip('重排序配置更新成功！', 'success');
      
      // 更新当前配置
      if (currentV3Config) {
        currentV3Config.use_reranker = use_reranker;
        currentV3Config.reranker_model_name = reranker_model_name;
        currentV3Config.reranker_top_n = reranker_top_n;
        currentV3Config.reranker_weight = reranker_weight;
        currentV3Config.reranker_backend = reranker_backend;
      }
      
      // 刷新重排序状态显示
      await refreshRerankerStatus();
      
      // 显示重排序状态区域
      document.getElementById('rerankerStatus').style.display = 'block';
      
    } else {
      showOptimizationTip(`重排序配置更新失败: ${result.message || '未知错误'}`, 'error');
    }
    
  } catch (error) {
    console.error('更新重排序配置时发生错误:', error);
    showOptimizationTip(`更新重排序配置失败: ${error.message}`, 'error');
  } finally {
    // 恢复按钮状态
    const updateBtn = document.querySelector('.reranker-actions .btn-primary');
    updateBtn.textContent = '🔄 更新重排序配置';
    updateBtn.disabled = false;
  }
}

// 刷新重排序状态
async function refreshRerankerStatus() {
  try {
    if (!isIndexBuilt) {
      return;
    }
    
    const response = await fetch('/api/v3/reranker_status');
    const result = await response.json();
    
    if (result.success) {
      displayRerankerStatus(result.reranker_status);
    } else {
      console.error('获取重排序状态失败:', result.message);
    }
    
  } catch (error) {
    console.error('刷新重排序状态时发生错误:', error);
  }
}

// 显示重排序状态
function displayRerankerStatus(status) {
  const statusContent = document.getElementById('rerankerStatusContent');
  
  if (!status.enabled) {
    statusContent.innerHTML = `
      <div class="status-item disabled">
        <span class="status-icon">❌</span>
        <span class="status-text">重排序功能已禁用</span>
      </div>
    `;
    return;
  }
  
  const modelInfo = status.model_info || {};
  const isAvailable = status.available;
  
  let statusHtml = `
    <div class="status-grid">
      <div class="status-item ${isAvailable ? 'success' : 'error'}">
        <span class="status-icon">${isAvailable ? '✅' : '❌'}</span>
        <span class="status-text">${isAvailable ? '重排序器可用' : '重排序器不可用'}</span>
      </div>
      
      <div class="status-item">
        <span class="status-label">模型名称:</span>
        <span class="status-value">${status.model_name || 'N/A'}</span>
      </div>
      
      <div class="status-item">
        <span class="status-label">后端类型:</span>
        <span class="status-value">${status.backend || 'N/A'}</span>
      </div>
      
      <div class="status-item">
        <span class="status-label">候选数量:</span>
        <span class="status-value">${status.top_n || 'N/A'}</span>
      </div>
      
      <div class="status-item">
        <span class="status-label">权重系数:</span>
        <span class="status-value">${status.weight || 'N/A'}</span>
      </div>
    </div>
  `;
  
  if (modelInfo.device) {
    statusHtml += `
      <div class="status-item">
        <span class="status-label">运行设备:</span>
        <span class="status-value">${modelInfo.device}</span>
      </div>
    `;
  }
  
  if (modelInfo.use_fp16 !== undefined) {
    statusHtml += `
      <div class="status-item">
        <span class="status-label">FP16优化:</span>
        <span class="status-value">${modelInfo.use_fp16 ? '启用' : '禁用'}</span>
      </div>
    `;
  }
  
  statusContent.innerHTML = statusHtml;
}

// 测试重排序器
async function testReranker() {
  try {
    if (!isIndexBuilt) {
      showOptimizationTip('请先上传文档构建索引', 'error');
      return;
    }
    
    showOptimizationTip('正在测试重排序器...', 'info');
    
    // 刷新状态以获取最新信息
    await refreshRerankerStatus();
    
    // 显示重排序状态区域
    document.getElementById('rerankerStatus').style.display = 'block';
    
    showOptimizationTip('重排序器测试完成，请查看状态信息', 'success');
    
  } catch (error) {
    console.error('测试重排序器时发生错误:', error);
    showOptimizationTip(`测试重排序器失败: ${error.message}`, 'error');
  }
}

// 重排序配置变化处理
function handleRerankerConfigChange() {
  const useReranker = document.getElementById('use_reranker');
  const rerankerConfigs = document.querySelectorAll('.reranker-config input, .reranker-config select');
  
  // 根据主开关状态启用/禁用其他配置
  rerankerConfigs.forEach(config => {
    if (config !== useReranker) {
      config.disabled = !useReranker.checked;
    }
  });
  
  // 如果启用重排序，显示状态区域
  if (useReranker.checked) {
    document.getElementById('rerankerStatus').style.display = 'block';
    // 延迟刷新状态，避免频繁请求
    setTimeout(() => refreshRerankerStatus(), 1000);
  } else {
    document.getElementById('rerankerStatus').style.display = 'none';
  }
}

// 新增：更新后台重排序配置的函数
async function updateBackendRerankerConfig() {
  try {
    // 获取当前重排序配置
    const use_reranker = document.getElementById('use_reranker').checked;
    const reranker_model_name = document.getElementById('reranker_model_name').value.trim();
    const reranker_top_n = Number(document.getElementById('reranker_top_n').value);
    const reranker_weight = Number(document.getElementById('reranker_weight').value);
    const reranker_backend = document.getElementById('reranker_backend').value;
    
    // 验证配置
    if (use_reranker && !reranker_model_name) {
      console.warn('启用重排序但未指定模型名称，跳过后台更新');
      return;
    }
    
    if (reranker_top_n < 1 || reranker_top_n > 200) {
      console.warn('重排序候选数量无效，跳过后台更新');
      return;
    }
    
    if (reranker_weight < 0 || reranker_weight > 5) {
      console.warn('重排序权重无效，跳过后台更新');
      return;
    }
    
    console.log('正在更新后台重排序配置...');
    
    // 发送更新请求到后台
    const response = await fetch('/api/v3/update_reranker', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        use_reranker,
        reranker_model_name,
        reranker_top_n,
        reranker_weight,
        reranker_backend
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('后台重排序配置更新成功:', result.reranker_status);
      
      // 更新当前V3配置
      if (currentV3Config) {
        currentV3Config.use_reranker = use_reranker;
        currentV3Config.reranker_model_name = reranker_model_name;
        currentV3Config.reranker_top_n = reranker_top_n;
        currentV3Config.reranker_weight = reranker_weight;
        currentV3Config.reranker_backend = reranker_backend;
      }
      
      // 刷新重排序状态显示
      await refreshRerankerStatus();
      
      // 显示成功提示
      showOptimizationTip('配置已保存并同步到后台！', 'success');
      
    } else {
      console.error('后台重排序配置更新失败:', result.message);
      showOptimizationTip(`后台配置更新失败: ${result.message}`, 'warning');
    }
    
  } catch (error) {
    console.error('更新后台重排序配置时发生错误:', error);
    showOptimizationTip(`后台配置更新失败: ${error.message}`, 'warning');
  }
}

// 新增：获取预设配置的函数
function getPresetConfig(presetKey) {
  const presets = {
    balanced: {
      encoder_backend: 'hf',
      hf_model_name: 'BAAI/bge-small-zh-v1.5',
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
      use_hybrid_search: true,
      use_multi_head: true,
      use_length_penalty: true,
      use_stateful_reranking: true,
      precompute_doc_tokens: false,
      enable_amp_if_beneficial: true,
      // 重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 50,
      reranker_weight: 1.5,
      reranker_backend: "auto"
    },
    precision: {
      encoder_backend: 'hf',
      hf_model_name: 'BAAI/bge-small-zh-v1.5',
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
      use_hybrid_search: true,
      use_multi_head: true,
      use_length_penalty: true,
      use_stateful_reranking: true,
      precompute_doc_tokens: false,
      enable_amp_if_beneficial: true,
      // 重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 100,
      reranker_weight: 2.0,
      reranker_backend: "auto"
    },
    speed: {
      encoder_backend: 'hf',
      hf_model_name: 'BAAI/bge-small-zh-v1.5',
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
      use_hybrid_search: true,
      use_multi_head: false,
      use_length_penalty: false,
      use_stateful_reranking: false,
      precompute_doc_tokens: true,
      enable_amp_if_beneficial: true,
      // 重排序配置
      use_reranker: false,
      reranker_model_name: "",
      reranker_top_n: 30,
      reranker_weight: 1.0,
      reranker_backend: "auto"
    },
    conversational: {
      encoder_backend: 'hf',
      hf_model_name: 'BAAI/bge-small-zh-v1.5',
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
      use_hybrid_search: true,
      use_multi_head: true,
      use_length_penalty: true,
      use_stateful_reranking: true,
      precompute_doc_tokens: false,
      enable_amp_if_beneficial: true,
      // 重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 80,
      reranker_weight: 1.8,
      reranker_backend: "auto"
    },
    hf_optimized: {
      encoder_backend: 'hf',
      hf_model_name: 'BAAI/bge-small-zh-v1.5',
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
      use_hybrid_search: true,
      use_multi_head: true,
      use_length_penalty: true,
      use_stateful_reranking: true,
      precompute_doc_tokens: false,
      enable_amp_if_beneficial: true,
      // 重排序配置
      use_reranker: true,
      reranker_model_name: "BAAI/bge-reranker-large",
      reranker_top_n: 60,
      reranker_weight: 1.6,
      reranker_backend: "auto"
    }
  };
  
  return presets[presetKey] || null;
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
  // 加载配置和历史
  loadConfig();
  loadHistory();
  
  // 绑定事件
  elements.uploadForm.addEventListener('submit', uploadFile);
  elements.askBtn.addEventListener('click', askQuestion);
  elements.clearHistoryBtn.addEventListener('click', clearHistory);
  elements.clearConfigBtn.addEventListener('click', clearConfig);
  elements.refreshResultsBtn.addEventListener('click', refreshResultsList);
  elements.batchTestForm.addEventListener('submit', handleBatchTest);
  
  // 绑定重排序配置变化事件
  document.getElementById('use_reranker').addEventListener('change', handleRerankerConfigChange);
  document.getElementById('reranker_model_name').addEventListener('input', handleRerankerConfigChange);
  document.getElementById('reranker_top_n').addEventListener('input', handleRerankerConfigChange);
  document.getElementById('reranker_weight').addEventListener('input', handleRerankerConfigChange);
  document.getElementById('reranker_backend').addEventListener('change', handleRerankerConfigChange);
  
  // 绑定键盘事件
  elements.question.addEventListener('keypress', handleKeyPress);
  
  // 绑定配置变化事件
  document.addEventListener('change', handleConfigChange);
  
  // 初始化重排序配置状态
  handleRerankerConfigChange();
  
  // 加载结果列表
  loadResultsList();
  
  // 显示优化提示
  showOptimizationTip('V3 RAG引擎已就绪，支持动态重排序配置！', 'success');
}); 