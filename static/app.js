const THREAD_KEY = 'OpsGuard_thread_id';
function newThreadId(){ return 'incident-' + crypto.randomUUID(); }
let threadId = localStorage.getItem(THREAD_KEY) || newThreadId();
localStorage.setItem(THREAD_KEY, threadId);

const q = document.getElementById('question');
const send = document.getElementById('sendBtn');
const messages = document.getElementById('messages');
const upload = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const fileLabel = document.getElementById('fileLabel');
const starters = document.getElementById('starterPrompts');
const sessionId = document.getElementById('sessionId');
const newSessionBtn = document.getElementById('newSessionBtn');
function renderSession(){ if(sessionId) sessionId.textContent='MEMORY / '+threadId.slice(-8).toUpperCase(); }
renderSession();

function esc(s=''){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));}
function addMessage(role, html, meta=''){
  const el=document.createElement('div'); el.className=`message ${role}`;
  const avatar=role==='assistant'?'<div class="avatar">S</div>':'';
  const label=role==='assistant'?'OpsGuard':'ON-CALL ENGINEER';
  el.innerHTML=`${avatar}<div class="message-body"><div class="message-label">${label}</div><div class="bubble">${html}</div>${meta}</div>`;
  messages.appendChild(el); messages.scrollTop=messages.scrollHeight; return el;
}
function loadingMarkup(){return '<span class="thinking">Running Self-RAG <i></i><i></i><i></i></span>';}
function pretty(v=''){return String(v).replaceAll('_',' ').replace(/\b\w/g,m=>m.toUpperCase());}

async function ask(){
  const question=q.value.trim(); if(!question) return;
  starters.style.display='none';
  addMessage('user',esc(question)); q.value=''; send.disabled=true;
  const loading=addMessage('assistant',loadingMarkup());
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question, thread_id: threadId})});
    const data=await r.json(); if(!r.ok) throw new Error(data.detail||'Request failed');
    loading.remove();
    let meta=`<div class="meta-card"><div class="verification"><span class="tag">ROUTE · ${esc(data.route)}</span>`;
    if(data.support_status) meta+=`<span class="tag ok">IsSUP · ${esc(pretty(data.support_status))}</span>`;
    if(data.usefulness) meta+=`<span class="tag ok">IsUSE · ${esc(pretty(data.usefulness))}</span>`;
    if(data.used_web_search) meta+=`<span class="tag web">INTERNET SEARCH USED</span>`;
    meta+=`<span class="tag memory">SQLITE MEMORY · ${data.memory_turns||0} TURN${(data.memory_turns||0)===1?'':'S'}</span>`;
    meta+='</div>';
    if(data.sources?.length){
      meta+='<div class="sources">'+data.sources.map(s=>`<div class="source"><span class="source-icon">${s.type==='web'?'⌁':'▱'}</span><span>${esc(s.title||s.source||'Evidence')}${s.page?' · p.'+s.page:''}${s.url?` · <a target="_blank" rel="noopener" href="${esc(s.url)}">open source ↗</a>`:''}</span></div>`).join('')+'</div>';
    }
    if(data.trace?.length) meta+=`<div class="trace"><details><summary>Inspect Self-RAG workflow trace</summary><ol>${data.trace.map(t=>`<li>${esc(t)}</li>`).join('')}</ol></details></div>`;
    meta+='</div>';
    addMessage('assistant',esc(data.answer),meta);
  }catch(e){loading.remove();addMessage('assistant','Request failed: '+esc(e.message));}
  finally{send.disabled=false;q.focus();}
}

send.addEventListener('click',ask);
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask();}});
starters.addEventListener('click',e=>{const b=e.target.closest('button[data-q]');if(!b)return;q.value=b.dataset.q;q.focus();});
const fileListWrap = document.getElementById('fileListWrap');
const fileList = document.getElementById('fileList');
const selectedCount = document.getElementById('selectedCount');
const clearAllBtn = document.getElementById('clearAllBtn');
let selectedFiles = [];

function formatSize(bytes){
  if(!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function getFileExt(name = ''){
  const ext = name.split('.').pop();
  return ext ? ext.toUpperCase() : 'DOC';
}

function renderSelectedFiles(){
  if(!fileList || !fileListWrap) return;
  if(selectedFiles.length === 0){
    fileListWrap.style.display = 'none';
    fileList.innerHTML = '';
    fileLabel.textContent = 'Choose runbook(s)';
    return;
  }
  fileListWrap.style.display = 'flex';
  selectedCount.textContent = `${selectedFiles.length} runbook${selectedFiles.length === 1 ? '' : 's'} selected`;
  fileLabel.textContent = `${selectedFiles.length} runbook${selectedFiles.length === 1 ? '' : 's'} chosen`;

  fileList.innerHTML = selectedFiles.map((f, i) => {
    const ext = getFileExt(f.name);
    const size = formatSize(f.size);
    return `<div class="file-card">
      <div class="file-card-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
      </div>
      <div class="file-card-info">
        <div class="file-card-name" title="${esc(f.name)}">${esc(f.name)}</div>
        <div class="file-card-meta">
          <span class="file-card-tag">${esc(ext)}</span>
          <span>·</span>
          <span>${esc(size)}</span>
        </div>
      </div>
      <button type="button" class="file-card-remove" data-index="${i}" title="Remove ${esc(f.name)}">✕</button>
    </div>`;
  }).join('');
}

fileInput.addEventListener('change',()=>{
  const newFiles = Array.from(fileInput.files || []);
  newFiles.forEach(nf => {
    if(!selectedFiles.some(f => f.name === nf.name && f.size === nf.size && f.lastModified === nf.lastModified)){
      selectedFiles.push(nf);
    }
  });
  uploadStatus.textContent = '';
  renderSelectedFiles();
});

fileList?.addEventListener('click', (e) => {
  const btn = e.target.closest('.file-card-remove');
  if(!btn) return;
  const idx = parseInt(btn.dataset.index, 10);
  if(!isNaN(idx) && idx >= 0 && idx < selectedFiles.length){
    selectedFiles.splice(idx, 1);
    uploadStatus.textContent = '';
    renderSelectedFiles();
  }
});

clearAllBtn?.addEventListener('click', () => {
  selectedFiles = [];
  fileInput.value = '';
  uploadStatus.textContent = '';
  renderSelectedFiles();
});

upload.addEventListener('click',async()=>{
  if(!selectedFiles || selectedFiles.length === 0){
    uploadStatus.textContent = 'Choose runbook(s) first.';
    return;
  }
  upload.disabled = true;
  uploadStatus.textContent = `Indexing ${selectedFiles.length} document${selectedFiles.length === 1 ? '' : 's'} into private operational knowledge…`;
  try{
    const fd = new FormData();
    for(let i = 0; i < selectedFiles.length; i++){
      fd.append('files', selectedFiles[i]);
    }
    const r = await fetch('/api/upload', {method: 'POST', body: fd});
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'Upload failed');
    const totalFiles = d.total_files ?? d.results?.length ?? selectedFiles.length;
    const totalChunks = d.chunks_indexed ?? 0;
    uploadStatus.textContent = `✓ ${totalFiles} file${totalFiles === 1 ? '' : 's'} (${totalChunks} chunks) indexed in ${d.namespace}`;
    selectedFiles = [];
    fileInput.value = '';
    renderSelectedFiles();
  }catch(e){
    uploadStatus.textContent = 'Error: ' + e.message;
  }finally{
    upload.disabled = false;
  }
});


newSessionBtn?.addEventListener('click',()=>{
  threadId = newThreadId();
  localStorage.setItem(THREAD_KEY, threadId);
  renderSession();
  messages.innerHTML = `<div class="message assistant"><div class="avatar">S</div><div class="message-body"><div class="message-label">OpsGuard</div><div class="bubble intro">New incident memory session started. Describe the production issue and I’ll build context across your follow-up questions.</div></div></div>`;
  starters.style.display='flex';
  q.value=''; q.focus();
});