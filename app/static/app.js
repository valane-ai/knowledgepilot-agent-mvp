let activeTaskId = localStorage.getItem('kp-active-task') || null;
let sessionId = null;
let taskSessions = JSON.parse(localStorage.getItem('kp-task-sessions') || '{}');

const $ = selector => document.querySelector(selector);
const chat = $('#chat');
const taskDialog = $('#task-dialog');

function renderMarkdown(text) {
  let html = escapeHtml(String(text));
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.+)$/gm, '<h5>$1</h5>').replace(/^## (.+)$/gm, '<h4>$1</h4>').replace(/^# (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>').replace(/(<li>[\s\S]*?<\/li>)(?:\n|$)/g, '$1');
  return html.replace(/\n/g, '<br>');
}

function addMessage(text, type, markdown = type === 'assistant') {
  const element = document.createElement('article');
  element.className = `message ${type}`;
  if (markdown) element.innerHTML = renderMarkdown(text);
  else element.textContent = text;
  chat.append(element);
  chat.scrollTop = chat.scrollHeight;
}

function saveTaskSessions() {
  localStorage.setItem('kp-task-sessions', JSON.stringify(taskSessions));
}

async function loadSessionHistory(taskId, storedSessionId) {
  try {
    const messages = await api(`/api/tasks/${taskId}/sessions/${storedSessionId}`);
    if (activeTaskId !== taskId || sessionId !== storedSessionId || !messages.length) return;
    chat.replaceChildren();
    messages.forEach(message => addMessage(message.content, message.role === 'user' ? 'user' : 'assistant'));
  } catch (error) {
    if (activeTaskId === taskId) addMessage(`历史会话加载失败：${error.message}`, 'error', false);
  }
}

function addFaithfulness(report) {
  if (!report) return;
  const label = report.status === 'supported' ? '引用校验通过' : report.status === 'warning' ? '引用校验提示' : '无内部资料引用';
  addMessage(`${label}：${report.message}`, `faithfulness ${report.status}`);
  if (report.invalid_citations?.length) addMessage(`无效引用：${report.invalid_citations.map(item => `[${item}]`).join('、')}`, 'error');
  if (report.uncited_statements?.length) addMessage(`待核对的未引用陈述：\n${report.uncited_statements.join('\n')}`, 'faithfulness warning');
}

function addSourcePreviews(sources) {
  if (!sources?.length) return;
  const card = document.createElement('article');
  card.className = 'source-previews';
  const title = document.createElement('strong');
  title.textContent = '来源原文预览';
  card.append(title);
  sources.forEach(source => {
    const preview = document.createElement('details');
    preview.open = sources.length === 1;
    const summary = document.createElement('summary');
    summary.textContent = `[${source.document}#${source.chunk_id}] · 相关度 ${source.score}`;
    const content = document.createElement('p');
    content.textContent = source.content;
    preview.append(summary, content);
    card.append(preview);
  });
  chat.append(card);
  chat.scrollTop = chat.scrollHeight;
}

function setTaskState(task) {
  activeTaskId = task?.id || null;
  localStorage.setItem('kp-active-task', activeTaskId || '');
  sessionId = task ? taskSessions[task.id] || null : null;
  $('#task-title').textContent = task ? task.name : '选择或创建一个任务';
  $('#message').disabled = !task;
  $('#send').disabled = !task;
  $('#delete-task').disabled = !task;
  $('#upload-label').classList.toggle('disabled', !task);
  $('#route').textContent = task ? '任务资料已隔离' : '等待选择任务';
  chat.replaceChildren();
  if (task) {
    addMessage(`你正在“${task.name}”中提问。上传的资料和答案引用都会限制在此任务内。`, 'assistant');
    if (sessionId) void loadSessionHistory(task.id, sessionId);
  }
  else chat.innerHTML = '<div class="welcome-card"><div class="welcome-icon">✦</div><h3>开始一段有依据的对话</h3><p>先创建或选择一个任务，然后上传资料。</p></div>';
}

async function api(url, options) {
  const response = await fetch(url, options);
  const text = response.status === 204 ? '' : await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = {detail: text}; }
  if (!response.ok) throw new Error(body?.detail || `请求失败（HTTP ${response.status}）`);
  return body;
}

async function streamChat(payload) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const body = await response.text();
    throw new Error(body || `请求失败（HTTP ${response.status}）`);
  }
  const element = document.createElement('article');
  element.className = 'message assistant';
  element.textContent = '正在生成回答…';
  chat.append(element);
  chat.scrollTop = chat.scrollHeight;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = ''; let answer = ''; let completed = null;
  const consume = packet => {
    const event = packet.match(/^event:\s*(.+)$/m)?.[1];
    const dataLine = packet.match(/^data:\s*(.+)$/m)?.[1];
    if (!event || !dataLine) return;
    const data = JSON.parse(dataLine);
    if (event === 'token') { answer += data.content; element.innerHTML = renderMarkdown(answer); chat.scrollTop = chat.scrollHeight; }
    if (event === 'done') completed = data;
  };
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const packets = buffer.split('\n\n'); buffer = packets.pop();
    packets.forEach(consume);
    if (done) break;
  }
  if (buffer.trim()) consume(buffer);
  if (!answer) { element.textContent = '模型未返回可展示的回答。'; }
  if (!completed) throw new Error('流式响应未正常结束。');
  return completed;
}

async function refreshTasks(selectCurrent = true) {
  const taskList = await api('/api/tasks');
  if (!taskList.some(task => task.id === activeTaskId)) activeTaskId = taskList[0]?.id || null;
  const active = taskList.find(task => task.id === activeTaskId);
  $('#tasks').replaceChildren(...taskList.map(task => {
    const button = document.createElement('button');
    button.className = `task-item ${task.id === activeTaskId ? 'active' : ''}`;
    button.innerHTML = `<span class="task-item-name">${escapeHtml(task.name)}</span><small>${task.document_count} 份资料 · ${task.chunk_count} 个片段</small>`;
    button.onclick = async () => { if (task.id !== activeTaskId) { setTaskState(task); await refreshTasks(false); await refreshDocuments(); } };
    return button;
  }));
  if (selectCurrent && active) setTaskState(active);
  if (activeTaskId) await refreshDocuments();
}

function escapeHtml(value) {
  const element = document.createElement('span');
  element.textContent = value;
  return element.innerHTML;
}

async function refreshDocuments() {
  if (!activeTaskId) return;
  const docs = await api(`/api/tasks/${activeTaskId}/documents`);
  $('#document-count').textContent = docs.length;
  $('#documents').replaceChildren(...docs.map(doc => {
    const item = document.createElement('li');
    item.className = 'document-item';
    const state = doc.status === 'processing' ? `正在解析：${doc.progress || 0}%` : doc.status === 'failed' ? `解析失败：${doc.error_detail || '请检查文件内容'}` : `${doc.chunks} 个片段`;
    item.innerHTML = `<span class="file-icon">▤</span><span class="document-meta"><strong title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</strong><small>${escapeHtml(state)}</small></span>`;
    if (doc.status !== 'processing') {
      const reindex = document.createElement('button');
      reindex.className = 'reindex-document';
      reindex.title = `重新解析并建立 ${doc.filename} 的索引`;
      reindex.textContent = '↻';
      reindex.onclick = async () => {
        try {
          await api(`/api/tasks/${activeTaskId}/documents/${doc.id}/reindex`, { method: 'POST' });
          $('#upload-status').textContent = '已开始重新解析并建立索引…';
          await refreshDocuments();
          waitForDocument(doc.id);
        } catch (error) { $('#upload-status').textContent = error.message; }
      };
      item.append(reindex);
    }
    const remove = document.createElement('button');
    remove.className = 'remove-document';
    remove.title = `删除 ${doc.filename}`;
    remove.textContent = '×';
    remove.onclick = async () => {
      if (!confirm(`确定从当前任务删除“${doc.filename}”吗？相关索引和原始文件会一并移除。`)) return;
      try { await api(`/api/tasks/${activeTaskId}/documents/${doc.id}`, { method: 'DELETE' }); await refreshTasks(false); await refreshDocuments(); }
      catch (error) { $('#upload-status').textContent = error.message; }
    };
    item.append(remove);
    return item;
  }));
  if (!docs.length) $('#documents').innerHTML = '<li class="empty-docs">还没有资料</li>';
}

async function waitForDocument(documentId) {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const docs = await api(`/api/tasks/${activeTaskId}/documents`);
    const document = docs.find(item => item.id === documentId);
    await refreshTasks(false); await refreshDocuments();
    if (!document || document.status !== 'processing') {
      $('#upload-status').textContent = document?.status === 'failed' ? `资料解析失败：${document.error_detail || '请检查文件内容。'}` : '资料已解析并建立索引。';
      return;
    }
  }
  $('#upload-status').textContent = '资料仍在后台解析，可稍后刷新查看状态。';
}

$('#new-task').onclick = () => { $('#task-name').value = ''; taskDialog.showModal(); $('#task-name').focus(); };
$('#delete-task').onclick = async () => {
  if (!activeTaskId) return;
  const name = $('#task-title').textContent;
  if (!confirm(`确定删除知识任务“${name}”吗？该任务中的资料、索引和历史对话都会一并删除。`)) return;
  try {
    await api(`/api/tasks/${activeTaskId}`, { method: 'DELETE' });
    delete taskSessions[activeTaskId];
    saveTaskSessions();
    setTaskState(null);
    $('#upload-status').textContent = '知识任务已删除。';
    await refreshTasks();
  } catch (error) { addMessage(error.message, 'error'); }
};
$('#task-form').addEventListener('submit', async event => {
  event.preventDefault();
  const name = $('#task-name').value.trim();
  if (!name) return;
  try {
    const task = await api('/api/tasks', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name}) });
    taskDialog.close(); setTaskState(task); await refreshTasks(false); await refreshDocuments();
  } catch (error) { $('#task-name').setCustomValidity(error.message); $('#task-name').reportValidity(); $('#task-name').setCustomValidity(''); }
});

$('#file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file || !activeTaskId) return;
  const status = $('#upload-status');
  const form = new FormData(); form.append('task_id', activeTaskId); form.append('file', file);
  status.textContent = '正在解析并建立索引…';
  try { const doc = await api('/api/documents', {method: 'POST', body: form}); status.textContent = doc.status === 'processing' ? '资料已入队，正在后台解析…' : `已加入当前任务：${doc.chunks} 个片段`; await refreshTasks(false); await refreshDocuments(); if (doc.status === 'processing') waitForDocument(doc.id); }
  catch (error) { status.textContent = error.message; }
  finally { event.target.value = ''; }
});

$('#form').addEventListener('submit', async event => {
  event.preventDefault();
  const input = $('#message'); const message = input.value.trim();
  if (!message || !activeTaskId) return;
  input.value = ''; addMessage(message, 'user'); $('#send').disabled = true;
  try {
    const data = await streamChat({message, task_id: activeTaskId, session_id: sessionId});
    sessionId = data.session_id;
    taskSessions[activeTaskId] = sessionId;
    saveTaskSessions();
    $('#route').textContent = data.route === 'general_llm' ? '通用大模型回答' : data.route === 'agent_knowledge' ? 'Agent 已检索任务资料' : data.route === 'agent_tool' ? 'Agent 已调用工具' : data.route === 'knowledge_base' ? '已引用任务资料' : data.route;
    addFaithfulness(data.faithfulness);
    addSourcePreviews(data.sources);
    if (data.trace?.length) addMessage(`执行轨迹：\n${data.trace.join('\n')}`, 'trace');
  } catch (error) { addMessage(error.message, 'error'); }
  finally { $('#send').disabled = false; input.focus(); }
});

refreshTasks().catch(error => addMessage(error.message, 'error'));
