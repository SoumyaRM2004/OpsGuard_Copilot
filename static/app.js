// ==========================================================================
// OpsGuard — Frontend Chat & Upload Logic
// ==========================================================================

// ─── Session / Thread Management ────────────────────────────────────

const THREAD_KEY = 'OpsGuard_thread_id';

function newThreadId() {
  return 'incident-' + crypto.randomUUID();
}

let threadId = localStorage.getItem(THREAD_KEY) || newThreadId();
localStorage.setItem(THREAD_KEY, threadId);


// ─── DOM References ─────────────────────────────────────────────────

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
const fileListWrap = document.getElementById('fileListWrap');
const fileList = document.getElementById('fileList');
const selectedCount = document.getElementById('selectedCount');
const clearAllBtn = document.getElementById('clearAllBtn');

let selectedFiles = [];


// ─── Session Display ────────────────────────────────────────────────

function renderSession() {
  if (sessionId) {
    sessionId.textContent = 'MEMORY / ' + threadId.slice(-8).toUpperCase();
  }
}
renderSession();


// ─── Utility Functions ──────────────────────────────────────────────

/**
 * Escape HTML special characters to prevent XSS attacks.
 * Every user-generated string must pass through this before innerHTML insertion.
 */
function esc(s = '') {
  return String(s).replace(/[&<>"']/g, function (c) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    }[c];
  });
}

/** Format bytes into human-readable size (e.g., "1.5 MB"). */
function formatSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/** Extract and uppercase the file extension (e.g., "report.pdf" → "PDF"). */
function getFileExt(name = '') {
  const ext = name.split('.').pop();
  return ext ? ext.toUpperCase() : 'DOC';
}

/** Convert snake_case to Title Case (e.g., "fully_supported" → "Fully Supported"). */
function pretty(v = '') {
  return String(v)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, function (m) { return m.toUpperCase(); });
}

/** Return the loading animation HTML shown while Self-RAG is running. */
function loadingMarkup() {
  return '<span class="thinking">Running... <i></i><i></i><i></i></span>';
}


// ─── Chat Message Rendering ─────────────────────────────────────────

/**
 * Add a chat message bubble to the messages container.
 * @param {string} role - "user" or "assistant"
 * @param {string} html - HTML content for the bubble (must be pre-escaped if user input)
 * @param {string} meta - Optional HTML for the meta card below the bubble
 */
function addMessage(role, html, meta = '') {
  const el = document.createElement('div');
  el.className = `message ${role}`;

  const avatar = role === 'assistant' ? '<div class="avatar">S</div>' : '';
  const label = role === 'assistant' ? 'OpsGuard' : 'ON-CALL ENGINEER';

  el.innerHTML =
    `${avatar}` +
    `<div class="message-body">` +
      `<div class="message-label">${label}</div>` +
      `<div class="bubble">${html}</div>` +
      `${meta}` +
    `</div>`;

  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}


// ─── Chat: Send Question ────────────────────────────────────────────

async function ask() {
  const question = q.value.trim();
  if (!question) return;

  starters.style.display = 'none';
  addMessage('user', esc(question));
  q.value = '';
  send.disabled = true;

  const loading = addMessage('assistant', loadingMarkup());

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, thread_id: threadId })
    });

    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Request failed');

    loading.remove();

    // Build the meta card with route, support status, sources, and trace
    let meta = `<div class="meta-card"><div class="verification">`;
    meta += `<span class="tag">ROUTE · ${esc(data.route)}</span>`;

    if (data.support_status) {
      meta += `<span class="tag ok">IsSUP · ${esc(pretty(data.support_status))}</span>`;
    }
    if (data.usefulness) {
      meta += `<span class="tag ok">IsUSE · ${esc(pretty(data.usefulness))}</span>`;
    }
    if (data.used_web_search) {
      meta += `<span class="tag web">INTERNET SEARCH USED</span>`;
    }

    const turns = data.memory_turns || 0;
    meta += `<span class="tag memory">SQLITE MEMORY · ${turns} TURN${turns === 1 ? '' : 'S'}</span>`;
    meta += '</div>';

    // Sources section
    if (data.sources && data.sources.length) {
      meta += '<div class="sources">';
      meta += data.sources.map(function (s) {
        const icon = s.type === 'web' ? '⌁' : '▱';
        const title = esc(s.title || s.source || 'Evidence');
        const page = s.page ? ' · p.' + s.page : '';
        const link = s.url
          ? ` · <a target="_blank" rel="noopener" href="${esc(s.url)}">open source ↗</a>`
          : '';
        return `<div class="source"><span class="source-icon">${icon}</span><span>${title}${page}${link}</span></div>`;
      }).join('');
      meta += '</div>';
    }

    // Trace section (collapsible)
    if (data.trace && data.trace.length) {
      meta += '<div class="trace"><details>';
      meta += '<summary>Inspect Self-RAG workflow trace</summary>';
      meta += '<ol>';
      meta += data.trace.map(function (t) { return `<li>${esc(t)}</li>`; }).join('');
      meta += '</ol></details></div>';
    }

    meta += '</div>';
    addMessage('assistant', esc(data.answer), meta);

  } catch (e) {
    loading.remove();
    addMessage('assistant', 'Request failed: ' + esc(e.message));
  } finally {
    send.disabled = false;
    q.focus();
  }
}

// Send on button click or Enter key
send.addEventListener('click', ask);
q.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

// Starter prompt buttons
starters.addEventListener('click', function (e) {
  const b = e.target.closest('button[data-q]');
  if (!b) return;
  q.value = b.dataset.q;
  q.focus();
});


// ─── File Upload: Selection & Display ───────────────────────────────

/** Render the list of selected files in the upload panel. */
function renderSelectedFiles() {
  if (!fileList || !fileListWrap) return;

  if (selectedFiles.length === 0) {
    fileListWrap.style.display = 'none';
    fileList.innerHTML = '';
    fileLabel.textContent = 'Choose runbook(s)';
    return;
  }

  fileListWrap.style.display = 'flex';
  selectedCount.textContent = `${selectedFiles.length} runbook${selectedFiles.length === 1 ? '' : 's'} selected`;
  fileLabel.textContent = `${selectedFiles.length} runbook${selectedFiles.length === 1 ? '' : 's'} chosen`;

  fileList.innerHTML = selectedFiles.map(function (f, i) {
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

// When user picks files via the file input
fileInput.addEventListener('change', function () {
  const newFiles = Array.from(fileInput.files || []);
  newFiles.forEach(function (nf) {
    // Avoid duplicates by checking name + size + lastModified
    const isDuplicate = selectedFiles.some(function (f) {
      return f.name === nf.name && f.size === nf.size && f.lastModified === nf.lastModified;
    });
    if (!isDuplicate) {
      selectedFiles.push(nf);
    }
  });
  uploadStatus.textContent = '';
  renderSelectedFiles();
});

// Remove individual file from selection
if (fileList) {
  fileList.addEventListener('click', function (e) {
    const btn = e.target.closest('.file-card-remove');
    if (!btn) return;
    const idx = parseInt(btn.dataset.index, 10);
    if (!isNaN(idx) && idx >= 0 && idx < selectedFiles.length) {
      selectedFiles.splice(idx, 1);
      uploadStatus.textContent = '';
      renderSelectedFiles();
    }
  });
}

// Clear all selected files
if (clearAllBtn) {
  clearAllBtn.addEventListener('click', function () {
    selectedFiles = [];
    fileInput.value = '';
    uploadStatus.textContent = '';
    renderSelectedFiles();
  });
}


// ─── File Upload: Submit ────────────────────────────────────────────

upload.addEventListener('click', async function () {
  if (!selectedFiles || selectedFiles.length === 0) {
    uploadStatus.textContent = 'Choose runbook(s) first.';
    return;
  }

  upload.disabled = true;
  uploadStatus.textContent =
    `Indexing ${selectedFiles.length} document${selectedFiles.length === 1 ? '' : 's'} into private operational knowledge…`;

  try {
    const fd = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      fd.append('files', selectedFiles[i]);
    }

    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload failed');

    const totalFiles = d.total_files ?? d.results?.length ?? selectedFiles.length;
    const totalChunks = d.chunks_indexed ?? 0;
    uploadStatus.textContent =
      `✓ ${totalFiles} file${totalFiles === 1 ? '' : 's'} (${totalChunks} chunks) indexed in ${d.namespace}`;

    selectedFiles = [];
    fileInput.value = '';
    renderSelectedFiles();

  } catch (e) {
    uploadStatus.textContent = 'Error: ' + e.message;
  } finally {
    upload.disabled = false;
  }
});


// ─── New Session ────────────────────────────────────────────────────

if (newSessionBtn) {
  newSessionBtn.addEventListener('click', function () {
    threadId = newThreadId();
    localStorage.setItem(THREAD_KEY, threadId);
    renderSession();

    messages.innerHTML =
      '<div class="message assistant">' +
        '<div class="avatar">S</div>' +
        '<div class="message-body">' +
          '<div class="message-label">OpsGuard</div>' +
          '<div class="bubble intro">New incident memory session started. ' +
          'Describe the production issue and I\'ll build context across your follow-up questions.</div>' +
        '</div>' +
      '</div>';

    starters.style.display = 'flex';
    q.value = '';
    q.focus();
  });
}


// ─── Multi-View Navigation ──────────────────────────────────────────

const navItems = document.querySelectorAll('.nav-item');
const viewPanes = document.querySelectorAll('.view-pane');

function switchView(viewName) {
  navItems.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-view') === viewName);
  });
  viewPanes.forEach(pane => {
    pane.classList.toggle('active', pane.id === ('view' + viewName.charAt(0).toUpperCase() + viewName.slice(1)));
  });

  if (viewName === 'runbooks') {
    loadRunbooks();
  } else if (viewName === 'audits') {
    loadAudits();
  }
}

navItems.forEach(btn => {
  btn.addEventListener('click', function () {
    const view = this.getAttribute('data-view');
    if (view) switchView(view);
  });
});


// ─── Runbook Vault Catalog ──────────────────────────────────────────

let runbooksCache = [];

async function loadRunbooks() {
  const tbody = document.getElementById('runbookTableBody');
  const countMetric = document.getElementById('vaultCountMetric');
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="5" class="loading-cell">Loading documents catalog…</td></tr>';

  try {
    const res = await fetch('/api/runbooks');
    const data = await res.json();
    runbooksCache = data.runbooks || [];

    if (countMetric) {
      countMetric.textContent = `${runbooksCache.length} Document${runbooksCache.length === 1 ? '' : 's'}`;
    }

    renderRunbookTable(runbooksCache);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">Error loading runbooks: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function renderRunbookTable(docs) {
  const tbody = document.getElementById('runbookTableBody');
  if (!tbody) return;

  if (!docs || docs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No documents found. Upload SOPs or runbooks to populate.</td></tr>';
    return;
  }

  tbody.innerHTML = docs.map(d => {
    const sizeKB = (d.size_bytes / 1024).toFixed(1) + ' KB';
    return `
      <tr>
        <td><strong>${escapeHtml(d.name)}</strong></td>
        <td><span class="badge">${escapeHtml(d.category)}</span></td>
        <td><span class="badge badge-format">${escapeHtml(d.type)}</span></td>
        <td>${sizeKB}</td>
        <td>
          <button class="action-btn ask-doc-btn" data-doc="${escapeHtml(d.name)}">Ask Copilot</button>
        </td>
      </tr>
    `;
  }).join('');

  tbody.querySelectorAll('.ask-doc-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      const docName = this.getAttribute('data-doc');
      switchView('copilot');
      q.value = `What are the key operational procedures and response steps outlined in ${docName}?`;
      q.focus();
    });
  });
}

const runbookSearchInput = document.getElementById('runbookSearchInput');
if (runbookSearchInput) {
  runbookSearchInput.addEventListener('input', function () {
    const term = this.value.trim().toLowerCase();
    const filtered = runbooksCache.filter(d =>
      d.name.toLowerCase().includes(term) || d.type.toLowerCase().includes(term) || d.category.toLowerCase().includes(term)
    );
    renderRunbookTable(filtered);
  });
}

const vaultUploadBtn = document.getElementById('vaultUploadBtn');
if (vaultUploadBtn && fileInput) {
  vaultUploadBtn.addEventListener('click', () => fileInput.click());
}


// ─── Audit Trail ────────────────────────────────────────────────────

let auditsCache = [];

async function loadAudits() {
  const tbody = document.getElementById('auditTableBody');
  const refreshBtn = document.getElementById('refreshAuditsBtn');
  if (refreshBtn) {
    refreshBtn.disabled = true;
    refreshBtn.innerHTML = '<span class="spin-icon">↻</span> Refreshing…';
  }

  if (tbody && (!auditsCache || auditsCache.length === 0)) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading-cell">Loading audit trail records…</td></tr>';
  }

  try {
    const res = await fetch('/api/audits?limit=50&_t=' + Date.now(), {
      cache: 'no-store',
      headers: { 'Pragma': 'no-cache', 'Cache-Control': 'no-cache' }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch audit records`);
    const data = await res.json();
    auditsCache = data || [];

    // Calculate metrics
    const total = auditsCache.length;
    const internal = auditsCache.filter(a => a.route === 'Private Runbooks').length;
    const web = auditsCache.filter(a => a.used_web || a.route === 'Internet Search').length;
    const supported = auditsCache.filter(a => !a.support_status || a.support_status === 'fully_supported').length;
    const verifiedPct = total > 0 ? Math.round((supported / total) * 100) : 100;

    const totalEl = document.getElementById('auditTotalMetric');
    const runbooksEl = document.getElementById('auditRunbooksMetric');
    const webEl = document.getElementById('auditWebMetric');
    const verifiedEl = document.getElementById('auditVerifiedMetric');

    if (totalEl) totalEl.textContent = `${total} Queries`;
    if (runbooksEl) runbooksEl.textContent = `${internal} Runbooks`;
    if (webEl) webEl.textContent = `${web} Fallbacks`;
    if (verifiedEl) verifiedEl.textContent = `${verifiedPct}%`;

    filterAndRenderAudits();

    if (refreshBtn) {
      refreshBtn.innerHTML = `✓ Refreshed (${total})`;
      setTimeout(() => {
        refreshBtn.innerHTML = '↻ Refresh Logs';
        refreshBtn.disabled = false;
      }, 1200);
    }
  } catch (err) {
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-cell">Error loading audit records: ${escapeHtml(err.message)}</td></tr>`;
    }
    if (refreshBtn) {
      refreshBtn.innerHTML = '⚠ Refresh Failed';
      setTimeout(() => {
        refreshBtn.innerHTML = '↻ Refresh Logs';
        refreshBtn.disabled = false;
      }, 2000);
    }
  }
}

function filterAndRenderAudits() {
  const tbody = document.getElementById('auditTableBody');
  const searchInput = document.getElementById('auditSearchInput');
  const routeSelect = document.getElementById('auditRouteFilter');

  const term = (searchInput ? searchInput.value : '').trim().toLowerCase();
  const routeFilter = routeSelect ? routeSelect.value : 'all';

  const filtered = auditsCache.filter(a => {
    const matchesRoute = routeFilter === 'all' || a.route === routeFilter;
    const matchesSearch = !term ||
      (a.question && a.question.toLowerCase().includes(term)) ||
      (a.answer && a.answer.toLowerCase().includes(term)) ||
      (a.sources_json && a.sources_json.toLowerCase().includes(term));
    return matchesRoute && matchesSearch;
  });

  if (!filtered || filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-cell">No matching audit records found.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered.map(a => {
    const timeStr = a.created_at ? new Date(a.created_at).toLocaleString() : 'N/A';
    let routeBadgeClass = 'badge-route-runbooks';
    if (a.route === 'Internet Search') routeBadgeClass = 'badge-route-web';
    else if (a.route === 'General Knowledge') routeBadgeClass = 'badge-route-direct';

    let sources = [];
    try {
      sources = JSON.parse(a.sources_json || '[]');
    } catch (_) {}

    const sourceText = sources.length > 0
      ? sources.map(s => s.title || s.source || 'Doc').slice(0, 2).join(', ') + (sources.length > 2 ? ` (+${sources.length - 2})` : '')
      : 'None';

    return `
      <tr>
        <td style="color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 11px;">${escapeHtml(timeStr)}</td>
        <td><strong>${escapeHtml(a.question.slice(0, 75))}${a.question.length > 75 ? '…' : ''}</strong></td>
        <td><span class="badge ${routeBadgeClass}">${escapeHtml(a.route || 'Runbooks')}</span></td>
        <td><span class="badge badge-format">${escapeHtml(pretty(a.support_status || 'fully_supported'))}</span></td>
        <td style="color: var(--muted);">${escapeHtml(sourceText)}</td>
        <td>
          <button class="action-btn inspect-audit-btn" data-id="${a.id}">Inspect</button>
        </td>
      </tr>
    `;
  }).join('');

  tbody.querySelectorAll('.inspect-audit-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      const id = parseInt(this.getAttribute('data-id'), 10);
      const record = auditsCache.find(r => r.id === id);
      if (record) openAuditModal(record);
    });
  });
}

const auditSearchInput = document.getElementById('auditSearchInput');
if (auditSearchInput) {
  auditSearchInput.addEventListener('input', filterAndRenderAudits);
}

const auditRouteFilter = document.getElementById('auditRouteFilter');
if (auditRouteFilter) {
  auditRouteFilter.addEventListener('change', filterAndRenderAudits);
}

const refreshAuditsBtn = document.getElementById('refreshAuditsBtn');
if (refreshAuditsBtn) {
  refreshAuditsBtn.addEventListener('click', loadAudits);
}


// ─── Audit Modal Inspector ──────────────────────────────────────────

const auditDetailModal = document.getElementById('auditDetailModal');
const closeAuditModalBtn = document.getElementById('closeAuditModalBtn');
const modalContent = document.getElementById('modalContent');

function openAuditModal(record) {
  if (!auditDetailModal || !modalContent) return;

  let trace = [];
  try {
    trace = JSON.parse(record.trace_json || '[]');
  } catch (_) {}

  let sources = [];
  try {
    sources = JSON.parse(record.sources_json || '[]');
  } catch (_) {}

  modalContent.innerHTML = `
    <h4>Incident Question</h4>
    <div style="font-weight: 600; font-size: 14px; margin-bottom: 14px;">${escapeHtml(record.question)}</div>

    <h4>Self-RAG Routing & Evidence</h4>
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;">
      <span class="badge badge-route-runbooks">Route: ${escapeHtml(record.route || 'Runbooks')}</span>
      <span class="badge badge-format">Support: ${escapeHtml(pretty(record.support_status || 'verified'))}</span>
      <span class="badge badge-format">Usefulness: ${escapeHtml(pretty(record.usefulness || 'useful'))}</span>
    </div>

    <h4>Copilot Answer</h4>
    <pre>${escapeHtml(record.answer)}</pre>

    ${sources.length > 0 ? `
      <h4>Sources Cited (${sources.length})</h4>
      <div style="margin-bottom: 14px;">
        ${sources.map(s => `
          <div style="padding: 6px 10px; background: #0c1212; border: 1px solid var(--line); border-radius: 6px; margin-bottom: 6px; font-size: 11px;">
            <strong>${escapeHtml(s.title || s.source || 'Doc')}</strong>
            ${s.page ? ` · Page ${s.page}` : ''}
            ${s.url ? ` · <a href="${escapeHtml(s.url)}" target="_blank" style="color: var(--cyan); text-decoration: none;">Link</a>` : ''}
          </div>
        `).join('')}
      </div>
    ` : ''}

    ${trace.length > 0 ? `
      <h4>Execution Trace Steps (${trace.length})</h4>
      <div>
        ${trace.map(t => `<div class="trace-step-item">→ ${escapeHtml(t)}</div>`).join('')}
      </div>
    ` : ''}
  `;

  auditDetailModal.style.display = 'flex';
}

if (closeAuditModalBtn && auditDetailModal) {
  closeAuditModalBtn.addEventListener('click', () => {
    auditDetailModal.style.display = 'none';
  });

  auditDetailModal.addEventListener('click', (e) => {
    if (e.target === auditDetailModal) {
      auditDetailModal.style.display = 'none';
    }
  });
}