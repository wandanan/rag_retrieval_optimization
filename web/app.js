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
  TOP_K_PARENTS: 'top_k_parents',
  
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
  [CONFIG_KEYS.TOP_K_PARENTS]: '4'
};

// DOM 元素
const elements = {
  uploadForm: document.getElementById('uploadForm'),
  fileInput: document.getElementById('fileInput'),
  uploadStatus: document.getElementById('uploadStatus'),
  question: document.getElementById('question'),
  askBtn: document.getElementById('askBtn'),
  attnAnswer: document.getElementById('attnAnswer'),
  vecAnswer: document.getElementById('vecAnswer'),
  attnCtx: document.getElementById('attnCtx'),
  vecCtx: document.getElementById('vecCtx'),
  attnStatus: document.getElementById('attnStatus'),
  vecStatus: document.getElementById('vecStatus'),
  historyList: document.getElementById('historyList'),
  clearConfigBtn: document.getElementById('clearConfigBtn'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn')
};

// 配置管理
function saveConfig() {
  const config = {
    [CONFIG_KEYS.PARENT_CHUNK_SIZE]: document.getElementById('parent_chunk_size').value,
    [CONFIG_KEYS.PARENT_OVERLAP]: document.getElementById('parent_overlap').value,
    [CONFIG_KEYS.SUB_CHUNK_SIZE]: document.getElementById('sub_chunk_size').value,
    [CONFIG_KEYS.SUB_OVERLAP]: document.getElementById('sub_overlap').value,
    [CONFIG_KEYS.BASE_URL]: document.getElementById('base_url').value,
    [CONFIG_KEYS.MODEL]: document.getElementById('model').value,
    [CONFIG_KEYS.API_KEY]: document.getElementById('api_key').value,
    [CONFIG_KEYS.PROMPT]: document.getElementById('prompt').value,
    [CONFIG_KEYS.TOP_K_PARENTS]: document.getElementById('top_k_parents').value
  };
  
  localStorage.setItem('rag_config', JSON.stringify(config));
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
          element.value = config[key];
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
    updateStatus(elements.attnStatus, '就绪', 'success');
    updateStatus(elements.vecStatus, '就绪', 'success');
    
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
    header.textContent = `片段${i + 1} (${ctx.parent_id}) | 向量分: ${ctx.vector_score.toFixed(4)}${ctx.attention_score ? ` | 注意力分: ${ctx.attention_score.toFixed(4)}` : ''}`;
    
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

  // 获取配置
  const base_url = document.getElementById('base_url').value.trim();
  const model = document.getElementById('model').value.trim();
  const api_key = document.getElementById('api_key').value.trim();
  const prompt = document.getElementById('prompt').value.trim();
  const top_k_parents = Number(document.getElementById('top_k_parents').value || '4');

  // 设置加载状态
  setLoading(true);
  updateStatus(elements.attnStatus, '处理中...', 'loading');
  updateStatus(elements.vecStatus, '处理中...', 'loading');
  
  elements.attnAnswer.textContent = '正在生成答案...';
  elements.vecAnswer.textContent = '正在生成答案...';
  elements.attnCtx.innerHTML = '';
  elements.vecCtx.innerHTML = '';

  const payload = {
    question,
    top_k_parents,
    top_k_sub: Math.max(50, top_k_parents * 20),
    prompt,
    llm: { base_url, model, api_key, temperature: 0.2 }
  };

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    if (!res.ok) throw new Error(data.detail || '请求失败');

    const { attention, vector } = data;

    // 更新答案
    elements.attnAnswer.textContent = attention.answer;
    elements.vecAnswer.textContent = vector.answer;

    // 渲染上下文
    renderContexts(elements.attnCtx, attention.contexts || []);
    renderContexts(elements.vecCtx, vector.contexts || []);

    // 更新状态
    updateStatus(elements.attnStatus, '完成', 'success');
    updateStatus(elements.vecStatus, '完成', 'success');

    // 添加到历史记录
    addToHistory(question, attention, vector);

  } catch (err) {
    const errorMsg = `❌ 失败：${err.message}`;
    elements.attnAnswer.textContent = errorMsg;
    elements.vecAnswer.textContent = errorMsg;
    
    updateStatus(elements.attnStatus, '错误', 'error');
    updateStatus(elements.vecStatus, '错误', 'error');
  } finally {
    setLoading(false);
  }
}

// 添加到历史记录
function addToHistory(question, attention, vector) {
  const historyItem = {
    id: Date.now(),
    question,
    attention: attention.answer,
    vector: vector.answer,
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
      <div class="history-answers">
        <div class="history-answer attention">
          <strong>🧠 注意力RAG:</strong><br>
          ${item.attention}
        </div>
        <div class="history-answer vector">
          <strong>🔍 向量RAG:</strong><br>
          ${item.vector}
        </div>
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
  // 加载保存的配置
  loadConfig();
  loadHistory();
  
  // 检查服务状态
  fetch('/api/health')
    .then(res => res.json())
    .then(data => {
      if (data.index_built) {
        isIndexBuilt = true;
        updateStatus(elements.attnStatus, '就绪', 'success');
        updateStatus(elements.vecStatus, '就绪', 'success');
        elements.uploadStatus.textContent = '✅ 索引已就绪';
        elements.uploadStatus.className = 'status success';
      }
    })
    .catch(() => {
      updateStatus(elements.attnStatus, '未连接', 'error');
      updateStatus(elements.vecStatus, '未连接', 'error');
    });
}); 