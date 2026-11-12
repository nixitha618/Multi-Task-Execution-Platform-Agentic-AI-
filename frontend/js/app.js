const API_BASE = '/api/v1';

// Centralized Indian Standard Time (IST, UTC+5:30) Formatter
function formatIST(dateInput, includeSeconds = true) {
    if (!dateInput) return 'N/A';
    try {
        const d = new Date(dateInput);
        if (isNaN(d.getTime())) return String(dateInput);
        const options = {
            timeZone: 'Asia/Kolkata',
            year: 'numeric',
            month: 'short',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        };
        if (includeSeconds) {
            options.second = '2-digit';
        }
        return d.toLocaleString('en-IN', options) + ' IST';
    } catch (e) {
        return String(dateInput);
    }
}

function formatISTTimeOnly(dateInput) {
    if (!dateInput) return 'N/A';
    try {
        const d = new Date(dateInput);
        if (isNaN(d.getTime())) return String(dateInput);
        return d.toLocaleTimeString('en-IN', {
            timeZone: 'Asia/Kolkata',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        }) + ' IST';
    } catch (e) {
        return String(dateInput);
    }
}

function updateLiveISTClock() {
    const el = document.getElementById('liveIstClock');
    if (!el) return;
    const now = new Date();
    el.innerText = now.toLocaleTimeString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });
}

// Global Theme Management (Light / Dark)
function initThemeSwitcher() {
    const savedTheme = localStorage.getItem('agentic_ai_theme') || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('agentic_ai_theme', theme);
    document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
        if (theme === 'light') {
            btn.innerHTML = '🌙';
            btn.setAttribute('title', 'Switch to Dark Mode');
            btn.setAttribute('aria-label', 'Switch to Dark Mode');
        } else {
            btn.innerHTML = '☀️';
            btn.setAttribute('title', 'Switch to Light Mode');
            btn.setAttribute('aria-label', 'Switch to Light Mode');
        }
    });
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(nextTheme);
}

// DOM Elements
let activeTaskId = null;
let eventSource = null;

document.addEventListener('DOMContentLoaded', () => {
    initThemeSwitcher();
    initTabs();
    initQuickPrompts();
    checkSystemStatus();
    updateAuthUI();
    loadTasks();
    loadMemories();
    loadSchedules();
    loadRagStatus();
    updateSessionUI();
});

// Session & Conversation History Management
function getOrCreateSessionId() {
    let sid = localStorage.getItem('agentic_ai_session_id');
    if (!sid) {
        sid = 'sess_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now().toString(36);
        localStorage.setItem('agentic_ai_session_id', sid);
    }
    return sid;
}

function updateSessionUI() {
    const badge = document.getElementById('sessionBadge');
    if (badge) {
        const sid = getOrCreateSessionId();
        badge.innerText = `Session: ${sid.substring(0, 16)}...`;
        badge.title = `Active Session ID: ${sid}`;
    }
}

function startNewSession() {
    const newSid = 'sess_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now().toString(36);
    localStorage.setItem('agentic_ai_session_id', newSid);
    updateSessionUI();
    appendConsole("System", `Started new conversation session [${newSid}]. Context reset for this session.`, "info");
    const resultsCard = document.getElementById('taskResultsCard');
    if (resultsCard) resultsCard.style.display = 'none';
}

async function toggleChatHistoryModal() {
    const modal = document.getElementById('chatHistoryModal');
    if (!modal) return;
    if (modal.style.display === 'flex') {
        modal.style.display = 'none';
    } else {
        modal.style.display = 'flex';
        await loadSessionChatHistory();
    }
}

async function loadSessionChatHistory() {
    const listEl = document.getElementById('chatHistoryModalList');
    if (!listEl) return;
    const sid = getOrCreateSessionId();
    listEl.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem;">Loading session history...</p>';

    try {
        const res = await fetch(`${API_BASE}/chat/history/${sid}`);
        if (!res.ok) throw new Error("Failed to fetch session history");
        const data = await res.json();
        const msgs = data.messages || [];

        if (msgs.length === 0) {
            listEl.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem;">No conversation messages in this session yet.</p>';
            return;
        }

        listEl.innerHTML = msgs.map((m, idx) => {
            const isUser = m.role === 'user';
            const time = formatIST(m.timestamp);
            return `
                <div style="background:${isUser ? 'rgba(59,130,246,0.1)' : 'rgba(255,255,255,0.03)'}; border:1px solid ${isUser ? 'rgba(59,130,246,0.25)' : 'var(--border-color)'}; border-radius:8px; padding:0.75rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem; font-size:0.75rem; color:var(--text-muted);">
                        <strong style="color:${isUser ? 'var(--accent-blue)' : 'var(--accent-emerald)'};">${isUser ? '👤 User' : '🤖 Assistant'}</strong>
                        <span>${time}</span>
                    </div>
                    <div style="font-size:0.85rem; line-height:1.45; white-space:pre-wrap; word-break:break-word;">${escapeHtml(m.content)}</div>
                </div>
            `;
        }).join('');
    } catch (err) {
        listEl.innerHTML = `<p style="color:var(--accent-rose); font-size:0.85rem;">Error loading history: ${escapeHtml(err.message)}</p>`;
    }
}

async function clearActiveSessionHistory() {
    const sid = getOrCreateSessionId();
    if (!confirm(`Are you sure you want to clear conversation history for session ${sid}?`)) return;

    try {
        const res = await fetch(`${API_BASE}/chat/history/${sid}`, { method: 'DELETE' });
        if (res.ok) {
            appendConsole("System", `Session conversation history cleared.`, "info");
            await loadSessionChatHistory();
        }
    } catch (err) {
        alert("Failed to clear session history: " + err.message);
    }
}

function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

// Tab Navigation
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');

            if (target === 'tab-tasks') loadTasks();
            if (target === 'tab-rag') loadRagStatus();
            if (target === 'tab-memory') loadMemories();
            if (target === 'tab-schedule') loadSchedules();
            if (target === 'tab-account') updateAuthUI();
        });
    });
}

// Quick Preset Prompts
function initQuickPrompts() {
    const tags = document.querySelectorAll('.quick-tag');
    const promptInput = document.getElementById('taskPrompt');
    tags.forEach(tag => {
        tag.addEventListener('click', (e) => {
            if (tag.id === 'presetRagTag' || tag.classList.contains('quick-tag-rag')) {
                // RAG Document Query activates the conversational RAG chatbot
                activateRagChatbotPreset(e);
                return;
            }
            promptInput.value = tag.getAttribute('data-prompt');
            promptInput.focus();
        });
    });
}

// =========================================================================
// CONVERSATIONAL RAG DOCUMENT CHATBOT CLIENT
// =========================================================================

let isRagSending = false;
let ragVoiceRecognition = null;

function getOrCreateRagChatSessionId() {
    let sid = localStorage.getItem('agentic_ai_rag_session_id');
    if (!sid) {
        sid = 'rag_sess_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now().toString(36);
        localStorage.setItem('agentic_ai_rag_session_id', sid);
    }
    return sid;
}

async function activateRagChatbotPreset(e, initialQuestion = null) {
    if (e && typeof e.preventDefault === 'function') {
        e.preventDefault();
        e.stopPropagation();
    }

    const card = document.getElementById('ragChatbotCard');
    const ragTag = document.getElementById('presetRagTag');
    if (!card) return;

    card.style.display = 'block';
    if (ragTag) {
        ragTag.classList.add('active-rag-preset');
    }

    // Illuminate Research Agent & Memory Agent to reflect live conversational RAG
    setAgentActive("ResearchAgent");
    const memNode = document.getElementById('node-memory');
    if (memNode) {
        memNode.classList.add('active');
    }

    // Populate scope dropdown and chunk count badge
    await populateRagChatScopeDropdown();

    // Auto-select attached file or mentioned document in Scope dropdown if available
    const select = document.getElementById('ragChatScopeSelect');
    if (select && select.options.length > 1) {
        let targetDoc = "";
        if (attachedDatasetPath) {
            targetDoc = attachedDatasetPath.split(/[/\\]/).pop();
        } else {
            const promptVal = (document.getElementById('taskPrompt')?.value || '');
            const match = promptVal.match(/[\w\-.]+\.(?:pdf|docx|txt|csv|xlsx)/i);
            if (match) targetDoc = match[0];
        }

        if (targetDoc) {
            for (let i = 0; i < select.options.length; i++) {
                if (select.options[i].value.toLowerCase().includes(targetDoc.toLowerCase()) || 
                    targetDoc.toLowerCase().includes(select.options[i].value.toLowerCase())) {
                    select.selectedIndex = i;
                    break;
                }
            }
        }
    }

    // If a prior answer was already generated on the dashboard, seamlessly import into the chatbot thread
    const resultsContent = document.getElementById('resultsContent');
    const promptInput = document.getElementById('taskPrompt');
    const thread = document.getElementById('ragChatThread');
    if (resultsContent && resultsContent.innerText.trim() && thread) {
        const priorQ = promptInput?.value.trim() || "Document Query";
        const priorA = resultsContent.innerText.trim();
        const existingMsgs = thread.querySelectorAll('.rag-chat-msg');
        // Only seed once (when thread only has the initial welcome message)
        if (existingMsgs.length <= 1) {
            appendRagMessageToThread('user', priorQ);
            appendRagMessageToThread('assistant', priorA);
        }
    }

    // Scroll into view
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Focus chat input
    const input = document.getElementById('ragChatInput');
    if (input) {
        if (initialQuestion) {
            input.value = initialQuestion;
            autoGrowRagTextarea(input);
        } else {
            input.value = "";
            input.placeholder = "Ask a follow-up question about your document... (Press Enter to send, Shift+Enter for newline)";
            autoGrowRagTextarea(input);
        }
        input.focus();
    }
}

async function openChatbotFromOutput() {
    const followupInput = document.getElementById('outputFollowupInput');
    const initialQ = followupInput ? followupInput.value.trim() : '';
    if (followupInput) followupInput.value = '';
    await activateRagChatbotPreset(null, initialQ);
    if (initialQ) {
        await sendRagChatMessage();
    }
}

function submitOutputFollowup() {
    const followupInput = document.getElementById('outputFollowupInput');
    const promptInput = document.getElementById('taskPrompt');
    let followQ = followupInput ? followupInput.value.trim() : '';

    // If output follow-up input was empty, fall back to main taskPrompt box
    if (!followQ && promptInput) {
        followQ = promptInput.value.trim();
    }

    if (!followQ) return alert("Please enter your follow-up question in the input box.");

    // Clear follow-up input
    if (followupInput) followupInput.value = '';

    // Set into main prompt box and submit task
    if (promptInput) {
        promptInput.value = followQ;
    }

    // Scroll up smoothly to the 7 Agent Visualizer & SSE Console so the user actively sees the agents running
    const visualizer = document.querySelector('.agent-visualizer') || document.getElementById('liveConsole');
    if (visualizer) {
        visualizer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    submitTask();
}

function exitRagChatbotMode() {
    const card = document.getElementById('ragChatbotCard');
    const ragTag = document.getElementById('presetRagTag');
    if (card) {
        card.style.display = 'none';
    }
    if (ragTag) {
        ragTag.classList.remove('active-rag-preset');
    }
    resetAgentVisualizer();
}

async function populateRagChatScopeDropdown() {
    const select = document.getElementById('ragChatScopeSelect');
    const chunkBadge = document.getElementById('ragChatChunkBadge');
    if (!select) return;

    try {
        const res = await fetch(`${API_BASE}/documents/status`);
        if (res.ok) {
            const data = await res.json();
            const currentVal = select.value;
            select.innerHTML = '<option value="">🌐 All Indexed Documents</option>';
            
            const docs = data.documents || [];
            docs.forEach(doc => {
                const opt = document.createElement('option');
                opt.value = doc.filename || doc.source;
                opt.innerText = `📄 ${doc.filename || doc.source} (${doc.chunk_count} chunks)`;
                select.appendChild(opt);
            });

            if (currentVal) {
                select.value = currentVal;
            }

            if (chunkBadge) {
                const total = data.total_chunks_indexed || 0;
                chunkBadge.innerText = `📚 ${total} RAG chunks (${docs.length} docs)`;
                chunkBadge.title = `Connected to ${total} vector chunks across ${docs.length} indexed documents`;
            }
        }
    } catch (e) {
        console.warn("Could not load RAG chat scope documents:", e);
    }
}

function onRagChatScopeChanged() {
    const select = document.getElementById('ragChatScopeSelect');
    const val = select ? select.value : '';
    const label = val ? `Filtered to '${val}'` : 'Global across all documents';
    appendConsole("ResearchAgent", `RAG Chatbot search scope updated: ${label}`, "info");
}

function autoGrowRagTextarea(el) {
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

function handleRagInputKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendRagChatMessage();
    }
}

async function askRagSuggested(btn) {
    if (!btn) return;
    const text = btn.innerText.trim();
    const input = document.getElementById('ragChatInput');
    if (input) {
        input.value = text;
        autoGrowRagTextarea(input);
    }
    await sendRagChatMessage();
}

async function sendRagChatMessage() {
    if (isRagSending) return;

    const input = document.getElementById('ragChatInput');
    const prompt = input ? input.value.trim() : '';
    if (!prompt) return;

    const sendBtn = document.getElementById('btnRagSend');
    const scopeSelect = document.getElementById('ragChatScopeSelect');
    const scope = scopeSelect ? scopeSelect.value.trim() : '';
    const sid = getOrCreateRagChatSessionId();

    isRagSending = true;
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<span>⏳ Thinking...</span>';
    }
    if (input) {
        input.disabled = true;
    }

    // Render User Message into thread
    appendRagMessageToThread('user', prompt, [], [], formatIST(new Date()));

    // Render Live Typing Indicator Bubble
    appendRagTypingIndicator();

    // Illuminate Research Agent & Memory Agent
    setAgentActive("ResearchAgent");
    const memNode = document.getElementById('node-memory');
    if (memNode) memNode.classList.add('active');

    try {
        const res = await fetch(`${API_BASE}/documents/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: prompt,
                session_id: sid,
                user_id: getActiveUserId(),
                document_scope: scope || null,
                top_k: 5
            })
        });

        removeRagTypingIndicator();

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "RAG chatbot query failed");
        }

        const data = await res.json();
        
        // Append Assistant Message into thread
        appendRagMessageToThread(
            'assistant',
            data.answer,
            data.citations || [],
            data.sources || [],
            formatIST(data.timestamp || new Date())
        );

        appendConsole("ResearchAgent", `Retrieved ${data.citations?.length || 0} grounded citations for RAG conversation query.`, "tool_output");

    } catch (err) {
        removeRagTypingIndicator();
        appendRagMessageToThread(
            'assistant',
            `⚠️ **Error communicating with RAG Knowledge Engine**: ${err.message}\n\nPlease verify that documents are uploaded in the Knowledge Base tab or try asking again.`,
            [],
            [],
            formatIST(new Date())
        );
        appendConsole("System", `RAG Chatbot error: ${err.message}`, "error");
    } finally {
        isRagSending = false;
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.innerHTML = '<span>💬 Ask Question</span>';
        }
        if (input) {
            input.disabled = false;
            input.value = '';
            input.style.height = 'auto';
            input.focus();
        }
    }
}

function appendRagMessageToThread(role, content, citations = [], sources = [], timeString = '') {
    const thread = document.getElementById('ragChatThread');
    if (!thread) return;

    const isUser = role === 'user';
    const msgEl = document.createElement('div');
    msgEl.className = `rag-chat-msg ${isUser ? 'user' : 'assistant'}`;

    const avatar = isUser ? '👤' : '🤖';
    const author = isUser ? 'You' : 'RAG Knowledge Assistant';

    // Format Markdown content safely
    let formattedText = escapeHtml(content);
    if (!isUser && typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        try {
            formattedText = marked.parse(content);
        } catch(e) {
            formattedText = escapeHtml(content).replace(/\n/g, '<br>');
        }
    } else if (!isUser) {
        // Fallback markdown styling (bold, bullets, linebreaks)
        formattedText = escapeHtml(content)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^-\s+(.*)$/gm, '<li>$1</li>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
    } else {
        formattedText = escapeHtml(content).replace(/\n/g, '<br>');
    }

    let citationsHtml = '';
    if (citations && citations.length > 0) {
        const citeBadges = citations.map(c => `
            <span class="rag-citation-tag" title="Verified ChromaDB Vector Match">
                📄 ${escapeHtml(c)}
            </span>
        `).join('');
        citationsHtml = `
            <div class="rag-citations-wrapper">
                <span style="font-size:0.7rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Grounded Sources:</span>
                <div class="rag-citations-list">${citeBadges}</div>
            </div>
        `;
    }

    msgEl.innerHTML = `
        <div class="rag-msg-avatar">${avatar}</div>
        <div class="rag-msg-body">
            <div class="rag-msg-header">
                <span class="rag-msg-author">${author}</span>
                <span class="rag-msg-time">${timeString || 'Just now'}</span>
                ${!isUser ? `<button type="button" class="rag-btn-copy" onclick="copyRagMsgText(this)" title="Copy answer text">📋</button>` : ''}
            </div>
            <div class="rag-msg-text">${formattedText}</div>
            ${citationsHtml}
        </div>
    `;

    thread.appendChild(msgEl);
    thread.scrollTop = thread.scrollHeight;
}

function appendRagTypingIndicator() {
    const thread = document.getElementById('ragChatThread');
    if (!thread) return;

    removeRagTypingIndicator();

    const ind = document.createElement('div');
    ind.id = 'ragTypingBubble';
    ind.className = 'rag-chat-msg assistant typing';
    ind.innerHTML = `
        <div class="rag-msg-avatar">🤖</div>
        <div class="rag-msg-body">
            <div class="rag-typing-dots">
                <span></span><span></span><span></span>
            </div>
            <span style="font-size:0.75rem; color:var(--text-muted); margin-left:0.5rem;">
                Searching ChromaDB vector knowledge & synthesizing grounded answer...
            </span>
        </div>
    `;
    thread.appendChild(ind);
    thread.scrollTop = thread.scrollHeight;
}

function removeRagTypingIndicator() {
    const ind = document.getElementById('ragTypingBubble');
    if (ind) ind.remove();
}

function copyRagMsgText(btn) {
    const textEl = btn.closest('.rag-msg-body')?.querySelector('.rag-msg-text');
    if (!textEl) return;
    const raw = textEl.innerText;
    navigator.clipboard.writeText(raw).then(() => {
        const prev = btn.innerText;
        btn.innerText = '✓';
        setTimeout(() => btn.innerText = prev, 1500);
    }).catch(() => {});
}

async function clearRagChatSession() {
    const sid = getOrCreateRagChatSessionId();
    try {
        await fetch(`${API_BASE}/documents/chat/history/${sid}`, { method: 'DELETE' }).catch(() => {});
    } catch (e) {}

    // Reset session ID
    const newSid = 'rag_sess_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now().toString(36);
    localStorage.setItem('agentic_ai_rag_session_id', newSid);

    const thread = document.getElementById('ragChatThread');
    if (thread) {
        thread.innerHTML = `
            <div class="rag-chat-msg assistant">
                <div class="rag-msg-avatar">🤖</div>
                <div class="rag-msg-body">
                    <div class="rag-msg-header">
                        <span class="rag-msg-author">RAG Knowledge Assistant</span>
                        <span class="rag-msg-time">Just now</span>
                    </div>
                    <div class="rag-msg-text">
                        🔄 <strong>New Conversation Session Started!</strong><br><br>
                        Prior conversation context has been reset. What question would you like to ask about your documents?
                    </div>
                </div>
            </div>
        `;
    }

    const input = document.getElementById('ragChatInput');
    if (input) {
        input.value = '';
        input.focus();
    }
    appendConsole("System", `RAG Chatbot session reset to [${newSid}]. Context cleared.`, "info");
}

function toggleRagVoiceInput() {
    const btn = document.getElementById('btnRagVoice');
    const input = document.getElementById('ragChatInput');
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert("Voice speech recognition is not supported in this browser. Please use Chrome or Edge.");
        return;
    }

    if (ragVoiceRecognition) {
        ragVoiceRecognition.stop();
        ragVoiceRecognition = null;
        if (btn) btn.classList.remove('listening');
        return;
    }

    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    ragVoiceRecognition = new SpeechRec();
    ragVoiceRecognition.continuous = false;
    ragVoiceRecognition.interimResults = false;
    ragVoiceRecognition.lang = 'en-US';

    ragVoiceRecognition.onstart = () => {
        if (btn) {
            btn.classList.add('listening');
            btn.title = "Listening... Speak now!";
        }
    };

    ragVoiceRecognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (input) {
            input.value = (input.value ? input.value + ' ' : '') + transcript;
            autoGrowRagTextarea(input);
            input.focus();
        }
    };

    ragVoiceRecognition.onerror = () => {
        if (btn) btn.classList.remove('listening');
        ragVoiceRecognition = null;
    };

    ragVoiceRecognition.onend = () => {
        if (btn) btn.classList.remove('listening');
        ragVoiceRecognition = null;
    };

    ragVoiceRecognition.start();
}


// Check Backend Status
async function checkSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/documents/status`);
        if (res.ok) {
            const data = await res.json();
            document.getElementById('ragCountBadge').innerText = `${data.total_chunks_indexed} RAG chunks`;
        }
    } catch (e) {
        console.warn("Backend status check offline:", e);
    }
}

let attachedDatasetPath = null;
let attachedDatasetName = null;

async function handleDatasetUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', getActiveUserId());

    const isDoc = /\.(pdf|docx|txt|csv|xlsx|xls)$/i.test(file.name);
    appendConsole("System", `Uploading ${isDoc ? 'document' : 'dataset'} '${file.name}' to server...`, "info");

    try {
        const res = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok && data.success) {
            attachedDatasetPath = data.file_path;
            attachedDatasetName = file.name;
            
            const badge = document.getElementById('attachedDatasetBadge');
            const removeBtn = document.getElementById('btnRemoveDataset');
            if (badge) {
                badge.style.display = 'inline-block';
                badge.innerText = `${isDoc ? '📄' : '📊'} Attached: ${file.name}`;
            }
            if (removeBtn) {
                removeBtn.style.display = 'inline-block';
            }

            appendConsole("System", `${isDoc ? 'Document' : 'Dataset'} '${file.name}' uploaded successfully (${data.chunks_indexed} chunks indexed into RAG) and attached to workflow.`, "info");
            
            const promptInput = document.getElementById('taskPrompt');
            if (!promptInput.value.trim() || promptInput.value.includes("Analyze sales dataset") || promptInput.value.includes("Search knowledge base")) {
                if (isDoc) {
                    promptInput.value = `Search knowledge base for information on ${file.name} and summarize key concepts.`;
                } else {
                    promptInput.value = `Analyze ${file.name} dataset, compute statistical summaries, and generate a visualization chart.`;
                }
            }
            checkSystemStatus();
        } else {
            alert("Upload failed: " + (data.message || data.detail));
        }
    } catch (err) {
        console.error(err);
        alert("Error uploading dataset file: " + err.message);
    }
}

function clearAttachedDataset() {
    attachedDatasetPath = null;
    attachedDatasetName = null;
    const fileInput = document.getElementById('datasetFileInput');
    if (fileInput) fileInput.value = '';
    const badge = document.getElementById('attachedDatasetBadge');
    if (badge) badge.style.display = 'none';
    const removeBtn = document.getElementById('btnRemoveDataset');
    if (removeBtn) removeBtn.style.display = 'none';
    appendConsole("System", "Attached dataset removed.", "info");
}

// Submit Task
async function submitTask() {
    const promptInput = document.getElementById('taskPrompt');
    const prompt = promptInput.value.trim();
    if (!prompt) return alert("Please enter a task prompt.");

    const runBtn = document.getElementById('btnRunTask');
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerText = "Executing Workflow...";
    }

    const followupBtn = document.querySelector('#outputFollowupBar button.btn-primary');
    if (followupBtn) {
        followupBtn.disabled = true;
        followupBtn.innerText = "⏳ Running Agents...";
    }

    // Reset Agent Visualizer
    resetAgentVisualizer();
    clearConsole();
    appendConsole("System", "Submitting task to Agentic Orchestrator...", "info");

    const payload = {
        prompt: prompt,
        user_id: getActiveUserId(),
        session_id: getOrCreateSessionId()
    };
    if (attachedDatasetPath) {
        payload.attached_files = [attachedDatasetPath];
    } else {
        const badge = document.getElementById('attachedDatasetBadge');
        if (badge && badge.innerText.includes('Attached:')) {
            const fname = badge.innerText.replace(/.*Attached:\s*/, '').trim();
            if (fname) payload.attached_files = [fname];
        }
    }

    try {
        const res = await fetch(`${API_BASE}/tasks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Failed to initialize task");
        const task = await res.json();
        activeTaskId = task.id;
        appendConsole("System", `Task initialized [ID: ${task.id}]. Opening real-time SSE stream...`, "info");

        // Immediately illuminate Memory Agent (1st agent in DAG)
        setAgentActive("MemoryAgent");

        // Open live SSE stream
        listenToStream(task.id);
    } catch (err) {
        appendConsole("System", `Error: ${err.message}`, "error");
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerText = "🚀 Run Autonomous Task";
        }
        if (followupBtn) {
            followupBtn.disabled = false;
            followupBtn.innerText = "🚀 Ask Follow-up";
        }
    }
}

// Server-Sent Events (SSE) Stream with Polling Fallback
let taskPollInterval = null;

function listenToStream(taskId) {
    if (eventSource) eventSource.close();
    if (taskPollInterval) clearInterval(taskPollInterval);

    eventSource = new EventSource(`${API_BASE}/tasks/${taskId}/stream`);

    eventSource.onmessage = (e) => {
        try {
            const event = JSON.parse(e.data);
            handleStreamEvent(event);
        } catch (err) {
            // Heartbeat or parse error
        }
    };

    eventSource.onerror = () => {
        // SSE transient error, polling fallback below will handle completion if SSE closes
    };

    // Polling fallback to guarantee completion and render even if SSE drops
    taskPollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/tasks/${taskId}`);
            if (res.ok) {
                const task = await res.json();
                if (task.status === "COMPLETED" || task.status === "FAILED") {
                    clearInterval(taskPollInterval);
                    taskPollInterval = null;
                    if (eventSource) eventSource.close();
                    onTaskFinished(taskId, {
                        final_output: task.final_output,
                        artifacts: task.structured_data?.artifacts || []
                    });
                }
            }
        } catch (e) {
            // Transient fetch error
        }
    }, 1500);
}

function handleStreamEvent(event) {
    const { event_type, agent_name, message, data } = event;

    // Highlight agent node
    if (agent_name) {
        setAgentActive(agent_name);
    }

    appendConsole(agent_name || "System", message, event_type);

    if (event_type === "DONE" || event_type === "ERROR") {
        if (eventSource) eventSource.close();
        if (taskPollInterval) {
            clearInterval(taskPollInterval);
            taskPollInterval = null;
        }
        onTaskFinished(activeTaskId, data);
    }
}

async function onTaskFinished(taskId, eventData) {
    if (taskPollInterval) {
        clearInterval(taskPollInterval);
        taskPollInterval = null;
    }
    if (eventSource) eventSource.close();

    const runBtn = document.getElementById('btnRunTask');
    if (runBtn) {
        runBtn.disabled = false;
        runBtn.innerText = "🚀 Run Autonomous Task";
    }

    const followupBtn = document.querySelector('#outputFollowupBar button.btn-primary');
    if (followupBtn) {
        followupBtn.disabled = false;
        followupBtn.innerText = "🚀 Ask Follow-up";
    }

    // Mark all prior nodes completed
    document.querySelectorAll('.agent-node').forEach(node => {
        node.classList.remove('active');
        node.classList.add('completed');
    });

    // If event payload already contains final output, render immediately!
    if (eventData && eventData.final_output) {
        renderTaskResults({
            final_output: eventData.final_output,
            structured_data: { artifacts: eventData.artifacts || [] }
        });
    }

    // Refresh task details from DB & history
    try {
        const res = await fetch(`${API_BASE}/tasks/${taskId}`);
        if (res.ok) {
            const task = await res.json();
            if (task.final_output) {
                renderTaskResults(task);
            }
            loadTasks();
        }
    } catch (e) {
        console.error("Error loading final task:", e);
    }
}

function renderTaskResults(task) {
    const container = document.getElementById('taskResultsCard');
    const content = document.getElementById('resultsContent');
    const chartContainer = document.getElementById('chartPreview');
    const artifactsContainer = document.getElementById('artifactsList');

    container.style.display = 'block';
    content.innerHTML = formatMarkdown(task.final_output || "Workflow finished without text output.");

    // Check for artifacts
    chartContainer.innerHTML = '';
    artifactsContainer.innerHTML = '';

    // Scroll to results so user immediately sees the generated response
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const artifacts = task.structured_data?.artifacts || [];
    if (artifacts.length > 0) {
        artifacts.forEach(path => {
            const filename = path.split('\\').pop().split('/').pop();
            const url = `/outputs/${filename}`;

            if (filename.endsWith('.png') || filename.endsWith('.jpg')) {
                chartContainer.innerHTML += `
                    <div class="chart-preview-container">
                        <p style="color: var(--accent-cyan); font-weight: 600; margin-bottom: 0.5rem;">Generated Visualization: ${filename}</p>
                        <a href="${url}" target="_blank"><img src="${url}" alt="Generated Chart" /></a>
                    </div>
                `;
            } else {
                artifactsContainer.innerHTML += `
                    <div style="margin-top: 0.5rem;">
                        <a href="${url}" download class="btn-secondary" style="display:inline-block;">📄 Download ${filename}</a>
                    </div>
                `;
            }
        });
    }

    container.scrollIntoView({ behavior: 'smooth' });
}

// Visualizer State Helpers
function resetAgentVisualizer() {
    document.querySelectorAll('.agent-node').forEach(node => {
        node.classList.remove('active', 'completed');
    });
}

function setAgentActive(agentName) {
    const mapping = {
        'MemoryAgent': 'node-memory',
        'TaskUnderstandingAgent': 'node-understand',
        'PlannerAgent': 'node-planner',
        'ExecutionAgent': 'node-exec',
        'ResearchAgent': 'node-research',
        'EvaluatorAgent': 'node-evaluator',
        'ResponseAgent': 'node-response'
    };

    const targetId = mapping[agentName];
    if (targetId) {
        document.querySelectorAll('.agent-node').forEach(node => {
            if (node.classList.contains('active')) {
                node.classList.remove('active');
                node.classList.add('completed');
            }
        });
        const target = document.getElementById(targetId);
        if (target) {
            target.classList.remove('completed');
            target.classList.add('active');
        }
    }
}

// Console Logging
function clearConsole() {
    document.getElementById('liveConsole').innerHTML = '';
}

function appendConsole(agent, msg, type = "info") {
    const consoleBox = document.getElementById('liveConsole');
    const time = formatISTTimeOnly(new Date());
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    let colorClass = 'log-msg';
    if (type === 'tool_call') colorClass = 'log-tool';
    if (type === 'error' || type === 'ERROR') colorClass = 'log-rose';

    entry.innerHTML = `
        <span class="log-time">[${time}]</span>
        <span class="log-agent">&lt;${agent}&gt;</span>
        <span class="${colorClass}">${escapeHtml(msg)}</span>
    `;
    consoleBox.appendChild(entry);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

// Markdown formatter helper
function formatMarkdown(text) {
    if (!text) return "";

    // 1. Process table blocks
    const lines = text.split(/\r?\n/);
    const outputLines = [];
    let inTable = false;
    let tableRows = [];

    function flushTable() {
        if (tableRows.length === 0) return "";
        let html = '<div style="overflow-x:auto; margin: 1rem 0;"><table class="data-table">';
        let hasHeader = false;
        
        for (let i = 0; i < tableRows.length; i++) {
            const row = tableRows[i].trim();
            if (/^\|[\s\-:|]+\|$/.test(row)) {
                hasHeader = true;
                continue;
            }
            const rawCells = row.split('|');
            const cells = rawCells.slice(1, rawCells.length - 1).map(c => c.trim());
            if (!hasHeader && i === 0 && tableRows.length > 1 && /^\|[\s\-:|]+\|$/.test(tableRows[1].trim())) {
                html += '<thead><tr>' + cells.map(c => `<th>${formatInline(c)}</th>`).join('') + '</tr></thead><tbody>';
            } else {
                html += '<tr>' + cells.map(c => `<td>${formatInline(c)}</td>`).join('') + '</tr>';
            }
        }
        if (hasHeader) html += '</tbody>';
        html += '</table></div>';
        tableRows = [];
        return html;
    }

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
            inTable = true;
            tableRows.push(line);
        } else {
            if (inTable) {
                outputLines.push(flushTable());
                inTable = false;
            }
            outputLines.push(line);
        }
    }
    if (inTable) {
        outputLines.push(flushTable());
    }

    let processed = outputLines.join('\n');

    // 2. Format blocks and lines
    processed = processed
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        .replace(/^\s*[-*]\s+(.*$)/gim, '<li style="margin-left: 1.25rem;">$1</li>')
        .replace(/^\s*(\d+)\.\s+(.*$)/gim, '<li style="margin-left: 1.25rem; list-style-type: decimal;">$2</li>')
        .replace(/\$\$(.*?)\$\$/gims, '<div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 6px; padding: 0.5rem 1rem; margin: 0.5rem 0; font-family: monospace; color: var(--accent-cyan); text-align: center;">$1</div>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        .replace(/`([^`]+)`/gim, '<code>$1</code>')
        .replace(/^---\s*$/gim, '<hr style="border:none; border-top:1px solid var(--border-color); margin: 1rem 0;"/>')
        .replace(/^\> (.*$)/gim, '<blockquote style="border-left: 3px solid var(--primary); padding-left: 0.75rem; color: var(--text-muted); margin: 0.5rem 0;">$1</blockquote>')
        .replace(/\n\n/gim, '<br/><br/>')
        .replace(/\n/gim, '<br/>');

    return processed;
}

function formatInline(cellText) {
    return (cellText || "")
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function escapeHtml(str) {
    return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Task History
async function loadTasks() {
    try {
        const res = await fetch(`${API_BASE}/tasks?limit=15`);
        if (!res.ok) return;
        const tasks = await res.json();
        const tbody = document.getElementById('tasksTableBody');
        tbody.innerHTML = '';

        tasks.forEach(t => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${t.id.slice(0, 8)}</code></td>
                <td><strong>${escapeHtml(t.prompt)}</strong></td>
                <td><span class="badge-status badge-${t.status}">${t.status}</span></td>
                <td>${t.task_type || 'General'}</td>
                <td>${formatIST(t.created_at)}</td>
                <td><button class="btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="viewTaskDetails('${t.id}')">Inspect</button></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Error loading tasks:", e);
    }
}

let currentInspectedTask = null;

async function viewTaskDetails(taskId) {
    try {
        const res = await fetch(`${API_BASE}/tasks/${taskId}`);
        if (!res.ok) {
            alert("Failed to load task details.");
            return;
        }
        const task = await res.json();
        currentInspectedTask = task;

        // Populate Modal Header & Badges
        const idBadge = document.getElementById('inspectTaskIdBadge');
        if (idBadge) {
            idBadge.innerText = task.id.slice(0, 8);
            idBadge.title = `Full ID: ${task.id}`;
        }
        
        const statusBadge = document.getElementById('inspectTaskStatusBadge');
        if (statusBadge) {
            statusBadge.className = `badge-status badge-${task.status}`;
            statusBadge.innerText = task.status;
        }

        // Populate Prompt & Meta
        const promptEl = document.getElementById('inspectTaskPrompt');
        if (promptEl) {
            promptEl.innerText = task.prompt || 'No prompt recorded.';
        }
        
        const metaEl = document.getElementById('inspectTaskMeta');
        if (metaEl) {
            const createdDate = formatIST(task.created_at);
            const durationSec = task.completed_at ? ((new Date(task.completed_at) - new Date(task.created_at)) / 1000).toFixed(1) + 's' : 'N/A';
            metaEl.innerHTML = `
                <span><strong>Type:</strong> ${escapeHtml(task.task_type || 'General')}</span>
                <span><strong>Created:</strong> ${createdDate}</span>
                <span><strong>Duration:</strong> ${durationSec}</span>
                <span><strong>Session:</strong> <code>${task.session_id ? task.session_id.slice(0, 10) + '...' : 'N/A'}</code></span>
            `;
        }

        // Populate Execution Steps
        const stepsContainer = document.getElementById('inspectTaskStepsContainer');
        const stepsList = document.getElementById('inspectTaskStepsList');
        if (stepsContainer && stepsList) {
            if (task.steps && task.steps.length > 0) {
                stepsContainer.style.display = 'block';
                let stepsHtml = '';
                task.steps.forEach((s, idx) => {
                    const stepStatusColor = s.status === 'COMPLETED' ? 'var(--accent-emerald)' : (s.status === 'FAILED' ? 'var(--accent-rose)' : 'var(--accent-cyan)');
                    stepsHtml += `
                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; padding:0.25rem 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                            <div>
                                <span style="color:${stepStatusColor}; font-weight:600;">Step ${s.step_index || idx+1}:</span>
                                <span style="color:#e2e8f0; margin-left:0.3rem;">${escapeHtml(s.description || s.agent_name)}</span>
                                ${s.tool_name ? `<code style="font-size:0.72rem; margin-left:0.4rem; background:rgba(255,255,255,0.06); padding:1px 4px; border-radius:3px;">[${escapeHtml(s.tool_name)}]</code>` : ''}
                            </div>
                            <span style="color:var(--text-muted); font-size:0.72rem;">${(s.duration_seconds || 0).toFixed(2)}s</span>
                        </div>
                    `;
                });
                stepsList.innerHTML = stepsHtml;
            } else {
                stepsContainer.style.display = 'none';
            }
        }

        // Populate Final Output
        const outputEl = document.getElementById('inspectTaskOutputContent');
        if (outputEl) {
            outputEl.innerHTML = formatMarkdown(task.final_output || "Workflow finished without text output.");
        }

        // Populate Artifacts & Documents
        const artifactsContainer = document.getElementById('inspectTaskArtifactsContainer');
        const artifactsList = document.getElementById('inspectTaskArtifactsList');
        if (artifactsContainer && artifactsList) {
            const artifacts = task.structured_data?.artifacts || [];
            if (artifacts.length > 0) {
                artifactsContainer.style.display = 'block';
                let artHtml = '';
                artifacts.forEach(path => {
                    const filename = path.split('\\').pop().split('/').pop();
                    const url = `/outputs/${filename}`;
                    if (filename.endsWith('.png') || filename.endsWith('.jpg')) {
                        artHtml += `
                            <div style="margin-top:0.4rem;">
                                <a href="${url}" target="_blank" style="font-size:0.8rem; color:var(--accent-cyan); display:inline-flex; align-items:center; gap:0.4rem;">
                                    📊 View Visualization: <strong>${escapeHtml(filename)}</strong>
                                </a>
                            </div>
                        `;
                    } else {
                        artHtml += `
                            <a href="${url}" download class="btn-secondary" style="font-size:0.8rem; padding:0.3rem 0.7rem; display:inline-flex; align-items:center; gap:0.4rem;">
                                📄 Download ${escapeHtml(filename)}
                            </a>
                        `;
                    }
                });
                artifactsList.innerHTML = artHtml;
            } else {
                artifactsContainer.style.display = 'none';
            }
        }

        // Open Modal in-place (STAY ON TASK HISTORY TAB, NO REDIRECTION!)
        const modal = document.getElementById('taskInspectionModal');
        if (modal) {
            modal.style.display = 'flex';
        }
    } catch (e) {
        console.error("Error inspecting task:", e);
        alert(`Failed to inspect task: ${e.message}`);
    }
}

function closeTaskInspectionModal() {
    const modal = document.getElementById('taskInspectionModal');
    if (modal) {
        modal.style.display = 'none';
    }
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        const btn = document.getElementById('btnSpeakInspectOutput');
        if (btn) btn.innerText = "🔊 Read Aloud";
    }
}

function copyInspectTaskOutput() {
    if (!currentInspectedTask || !currentInspectedTask.final_output) {
        return alert("No output text to copy.");
    }
    navigator.clipboard.writeText(currentInspectedTask.final_output).then(() => {
        alert("Task output copied to clipboard!");
    }).catch(err => {
        console.error("Copy failed:", err);
    });
}

function speakInspectOutput() {
    if (!currentInspectedTask || !currentInspectedTask.final_output) return;
    const btn = document.getElementById('btnSpeakInspectOutput');
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        if (btn) btn.innerText = "🔊 Read Aloud";
        return;
    }
    const cleanText = currentInspectedTask.final_output.replace(/[#*_`>\[\]]/g, '');
    const utter = new SpeechSynthesisUtterance(cleanText);
    utter.onend = () => { if (btn) btn.innerText = "🔊 Read Aloud"; };
    utter.onerror = () => { if (btn) btn.innerText = "🔊 Read Aloud"; };
    window.speechSynthesis.speak(utter);
    if (btn) btn.innerText = "⏹️ Stop";
}

function openCurrentTaskInStudio() {
    if (!currentInspectedTask) return;
    closeTaskInspectionModal();
    // Switch to studio tab only if user explicitly clicks this button
    document.querySelector('[data-tab="tab-studio"]').click();
    renderTaskResults(currentInspectedTask);
}

// Add Escape key listener to close modals
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeTaskInspectionModal();
        const chatModal = document.getElementById('chatHistoryModal');
        if (chatModal && chatModal.style.display === 'flex') {
            chatModal.style.display = 'none';
        }
    }
});

// RAG Upload & Status
async function uploadDocument() {
    const fileInput = document.getElementById('ragFileInput');
    if (!fileInput.files || fileInput.files.length === 0) {
        return alert("Please select a file to upload.");
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('user_id', getActiveUserId());

    const msgEl = document.getElementById('ragUploadMsg');
    msgEl.innerText = "Uploading, chunking, and generating vector embeddings...";

    try {
        const res = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            msgEl.innerText = `✅ ${data.message} Active search scope set to this document.`;
            attachedDatasetPath = data.file_path;
            attachedDatasetName = data.filename || (fileInput.files[0] ? fileInput.files[0].name : 'document');
            
            const badge = document.getElementById('attachedDatasetBadge');
            const removeBtn = document.getElementById('btnRemoveDataset');
            const isDoc = /\.(pdf|docx|txt)$/i.test(attachedDatasetName || '');
            if (badge) {
                badge.style.display = 'inline-block';
                badge.innerText = `${isDoc ? '📄' : '📊'} Attached: ${attachedDatasetName}`;
            }
            if (removeBtn) {
                removeBtn.style.display = 'inline-block';
            }

            const promptInput = document.getElementById('taskPrompt');
            if (promptInput && (!promptInput.value.trim() || promptInput.value.includes("Search knowledge base") || promptInput.value.includes("Analyze sales"))) {
                promptInput.value = `Generate a markdown summary report of ${attachedDatasetName}`;
            }

            appendConsole("System", `Document '${attachedDatasetName}' indexed into RAG and attached to active workflow.`, "info");
            
            // Reload status and auto-select this newly uploaded document in the playground
            await loadRagStatus(attachedDatasetName);
            fileInput.value = '';
        } else {
            msgEl.innerText = `❌ ${data.message}`;
        }
    } catch (e) {
        msgEl.innerText = `❌ Upload failed: ${e.message}`;
    }
}

async function loadRagStatus(preferredDoc = null) {
    try {
        const res = await fetch(`${API_BASE}/documents/status`);
        if (res.ok) {
            const data = await res.json();
            const docs = data.documents || [];
            const statusEl = document.getElementById('ragStatusText');
            if (statusEl) {
                statusEl.innerText = 
                    `ChromaDB Knowledge Base: ${data.total_chunks_indexed} chunks indexed across ${docs.length} documents | Model: ${data.embedding_model}`;
            }
            const countBadge = document.getElementById('ragCountBadge');
            if (countBadge) {
                countBadge.innerText = `${data.total_chunks_indexed} RAG chunks`;
            }

            // Populate Scope Dropdown
            const scopeSelect = document.getElementById('ragDocumentScopeSelect');
            if (scopeSelect) {
                const currentVal = preferredDoc || scopeSelect.value || '';
                let optionsHtml = '<option value="">🌐 All Indexed Documents (Global Search)</option>';
                docs.forEach(doc => {
                    const sel = (doc.filename === currentVal) ? 'selected' : '';
                    optionsHtml += `<option value="${escapeHtml(doc.filename)}" ${sel}>📄 ${escapeHtml(doc.filename)} (${doc.chunks_count} chunks)</option>`;
                });
                scopeSelect.innerHTML = optionsHtml;
                onRagScopeChanged();
            }

            // Populate Document List in Upload Card
            const listEl = document.getElementById('ragIndexedDocsList');
            if (listEl) {
                if (docs.length === 0) {
                    listEl.innerHTML = '<p style="font-size:0.8rem; color:var(--text-muted); margin:0.25rem 0;">No documents indexed yet.</p>';
                } else {
                    let listHtml = '';
                    docs.forEach(doc => {
                        listHtml += `
                            <div style="display:flex; justify-content:space-between; align-items:center; padding:0.35rem 0.5rem; border-bottom:1px solid rgba(255,255,255,0.06);">
                                <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:65%;">
                                    <span style="font-size:0.82rem; font-weight:600; color:#e2e8f0;" title="${escapeHtml(doc.filename)}">📄 ${escapeHtml(doc.filename)}</span>
                                    <span style="font-size:0.75rem; color:var(--accent-cyan); margin-left:0.4rem;">(${doc.chunks_count} chunks)</span>
                                </div>
                                <div style="display:flex; gap:0.3rem;">
                                    <button class="btn-secondary" style="font-size:0.72rem; padding:0.2rem 0.45rem; color:var(--accent-cyan);" onclick="setRagSearchScope('${escapeHtml(doc.filename)}')" title="Scope playground search to this document">🎯 Target</button>
                                    <button class="btn-secondary" style="font-size:0.72rem; padding:0.2rem 0.45rem; color:var(--accent-rose); border-color:var(--accent-rose);" onclick="deleteDocumentFromRag('${escapeHtml(doc.filename)}')" title="Delete all chunks of this file">🗑️</button>
                                </div>
                            </div>
                        `;
                    });
                    listEl.innerHTML = listHtml;
                }
            }
        }
    } catch (e) {
        console.error("Error loading RAG status:", e);
    }
}

function onRagScopeChanged() {
    const scopeSelect = document.getElementById('ragDocumentScopeSelect');
    const badge = document.getElementById('activeDocScopeBadge');
    if (!scopeSelect || !badge) return;
    const val = scopeSelect.value;
    if (val) {
        badge.innerText = `🎯 Scoped: ${val}`;
        badge.style.background = 'rgba(16, 185, 129, 0.2)';
        badge.style.color = 'var(--accent-emerald)';
    } else {
        badge.innerText = '🌐 All Documents';
        badge.style.background = 'rgba(56, 189, 248, 0.15)';
        badge.style.color = 'var(--accent-cyan)';
    }
}

function setRagSearchScope(filename) {
    const scopeSelect = document.getElementById('ragDocumentScopeSelect');
    if (scopeSelect) {
        scopeSelect.value = filename;
        onRagScopeChanged();
        const playgroundInput = document.getElementById('ragQueryInput');
        if (playgroundInput) {
            playgroundInput.focus();
        }
    }
}

async function deleteDocumentFromRag(filename) {
    if (!confirm(`Are you sure you want to remove all indexed chunks for '${filename}' from the Knowledge Base?`)) {
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/documents/source/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (res.ok) {
            appendConsole("System", `Removed '${filename}' from Knowledge Base (${data.deleted_chunks || 0} chunks).`, "info");
            await loadRagStatus();
        } else {
            alert(`Failed to delete document: ${data.detail || data.message}`);
        }
    } catch (e) {
        alert(`Error deleting document: ${e.message}`);
    }
}

async function clearEntireKnowledgeBase() {
    if (!confirm("Are you sure you want to delete ALL indexed documents and chunks from the ChromaDB Knowledge Base? This cannot be undone.")) {
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/documents/clear`, {
            method: 'DELETE'
        });
        if (res.ok) {
            appendConsole("System", "Entire ChromaDB Knowledge Base cleared successfully.", "warning");
            await loadRagStatus();
        } else {
            const data = await res.json();
            alert(`Failed to clear knowledge base: ${data.detail || data.message}`);
        }
    } catch (e) {
        alert(`Error clearing knowledge base: ${e.message}`);
    }
}

async function testRagQuery(forceAll = false) {
    const q = document.getElementById('ragQueryInput').value.trim();
    if (!q) return;
    const resContainer = document.getElementById('ragQueryResults');
    
    const scopeSelect = document.getElementById('ragDocumentScopeSelect');
    const selectedSource = forceAll ? '' : (scopeSelect ? scopeSelect.value : '');
    
    const scopeLabel = selectedSource ? `strictly within <strong>${escapeHtml(selectedSource)}</strong>` : 'across <strong>All Documents</strong>';
    resContainer.innerHTML = `<p style="font-size:0.85rem; color:var(--text-muted);">Retrieving semantic matches ${scopeLabel}...</p>`;

    try {
        let url = `${API_BASE}/documents/query?query=${encodeURIComponent(q)}&top_k=5`;
        if (selectedSource) {
            url += `&source=${encodeURIComponent(selectedSource)}`;
        }
        const res = await fetch(url, { method: 'POST' });
        const data = await res.json();
        resContainer.innerHTML = '';
        
        // Scope banner
        const bannerHtml = `
            <div style="font-size:0.78rem; padding:0.35rem 0.6rem; border-radius:5px; margin-bottom:0.6rem; background:rgba(56, 189, 248, 0.12); color:var(--accent-cyan); display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(56, 189, 248, 0.25);">
                <span>${selectedSource ? `🎯 Results filtered strictly to: <strong>${escapeHtml(selectedSource)}</strong>` : '🌐 Global search across all indexed documents'}</span>
                <span style="font-weight:600;">${data.matches ? data.matches.length : 0} match${(data.matches && data.matches.length === 1) ? '' : 'es'}</span>
            </div>
        `;
        resContainer.innerHTML = bannerHtml;

        if (!data.matches || data.matches.length === 0) {
            resContainer.innerHTML += `<p style="color: #94a3b8; font-size: 0.85rem; padding:0.5rem 0;">No matching chunks found ${selectedSource ? `in '${escapeHtml(selectedSource)}'` : 'above similarity threshold'}.</p>`;
            return;
        }
        
        data.matches.forEach((m, idx) => {
            const pageStr = m.page ? ` — Page ${m.page}` : '';
            const scoreStr = m.score ? ` (Score: ${(m.score * 100).toFixed(1)}%)` : '';
            resContainer.innerHTML += `
                <div style="background: var(--bg-input); padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid var(--accent-cyan);">
                    <strong>#${idx+1} [Source: ${escapeHtml(m.source)}${pageStr}]${scoreStr}</strong>
                    <p style="margin-top: 0.3rem; font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">${escapeHtml(m.content)}</p>
                </div>
            `;
        });
    } catch (e) {
        resContainer.innerText = `Query failed: ${e.message}`;
    }
}

// Memory System
function setMemPreset(key, val, cat) {
    const keyInput = document.getElementById('memKey');
    const valInput = document.getElementById('memValue');
    const catInput = document.getElementById('memCategory');
    if (keyInput) keyInput.value = key;
    if (valInput) valInput.value = val;
    if (catInput && cat) catInput.value = cat;
    if (valInput) valInput.focus();
}

async function loadMemories() {
    try {
        const res = await fetch(`${API_BASE}/memory?user_id=${getActiveUserId()}`);
        if (!res.ok) return;
        const mems = await res.json();
        const list = document.getElementById('memoryList');
        list.innerHTML = '';
        if (mems.length === 0) {
            list.innerHTML = '<p style="color:var(--text-muted); padding:1rem; text-align:center;">No custom memories or preferences stored yet.</p>';
            return;
        }

        mems.forEach(m => {
            let catColor = '#06b6d4'; // cyan
            let catBg = 'rgba(6,182,212,0.12)';
            if (m.category === 'fact') {
                catColor = '#38bdf8'; // sky
                catBg = 'rgba(56,189,248,0.12)';
            } else if (m.category === 'rule') {
                catColor = '#f59e0b'; // amber
                catBg = 'rgba(245,158,11,0.12)';
            } else if (m.category === 'task_history') {
                catColor = '#94a3b8'; // slate
                catBg = 'rgba(148,163,184,0.12)';
            }

            list.innerHTML += `
                <div style="background: var(--bg-input); padding: 0.85rem; border-radius: 8px; margin-bottom: 0.65rem; border: 1px solid rgba(255,255,255,0.06); display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem;">
                    <div style="flex: 1; min-width: 0;">
                        <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.35rem; flex-wrap: wrap;">
                            <span style="font-size: 0.7rem; font-weight: 700; color: ${catColor}; background: ${catBg}; padding: 0.15rem 0.45rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em;">
                                ${escapeHtml(m.category)}
                            </span>
                            <span style="color: #f8fafc; font-weight: 600; font-size: 0.9rem;">
                                ${escapeHtml(m.key)}
                            </span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.45; word-break: break-word; white-space: pre-wrap;">${escapeHtml(m.value)}</div>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.35rem; align-items: flex-end;">
                        <button class="btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem; white-space: nowrap;" onclick="setMemPreset('${escapeHtml(m.key).replace(/'/g, "\\'")}', '${escapeHtml(m.value).replace(/'/g, "\\'")}', '${escapeHtml(m.category)}')">✏️ Edit</button>
                        <button class="btn-secondary" style="color: var(--accent-rose); border-color: rgba(244,63,94,0.4); padding: 0.2rem 0.5rem; font-size: 0.75rem; white-space: nowrap;" onclick="deleteMemory('${m.id}')">🗑️ Delete</button>
                    </div>
                </div>
            `;
        });
    } catch (e) {
        console.error("Error loading memories:", e);
    }
}

async function addMemory() {
    const key = document.getElementById('memKey').value.trim();
    const value = document.getElementById('memValue').value.trim();
    const category = document.getElementById('memCategory').value;
    if (!key || !value) return alert("Please enter both a key and a value.");

    await fetch(`${API_BASE}/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: getActiveUserId(), key, value, category })
    });

    document.getElementById('memKey').value = '';
    document.getElementById('memValue').value = '';
    loadMemories();
}

async function deleteMemory(id) {
    await fetch(`${API_BASE}/memory/${id}`, { method: 'DELETE' });
    loadMemories();
}

// Scheduler
async function loadSchedules() {
    try {
        const res = await fetch(`${API_BASE}/schedule`);
        if (!res.ok) return;
        const jobs = await res.json();
        const tbody = document.getElementById('scheduleTableBody');
        tbody.innerHTML = '';
        if (jobs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No scheduled workflows configured.</td></tr>';
            return;
        }
        jobs.forEach(j => {
            const statusBadge = j.is_active 
                ? '<span style="color:#10b981; font-weight:600; display:inline-flex; align-items:center; gap:0.25rem;">✅ Active</span>' 
                : '<span style="color:#f59e0b; font-weight:600; display:inline-flex; align-items:center; gap:0.25rem;">⏸️ Paused</span>';

            const toggleBtnText = j.is_active ? '⏸️ Pause' : '▶️ Resume';
            const toggleBtnTitle = j.is_active ? 'Pause this job from running' : 'Resume scheduled execution';

            const rawPrompt = (j.prompt || '').trim();
            // Clean outer quotes and replace internal newlines with space for readable prose
            const cleanPrompt = rawPrompt.replace(/^["']|["']$/g, '').replace(/[\r\n]+/g, ' ').trim();

            tbody.innerHTML += `
                <tr>
                    <td class="sched-col-name">${escapeHtml(j.name)}</td>
                    <td><code class="sched-col-cron">${escapeHtml(j.cron_expression)}</code></td>
                    <td class="sched-col-prompt">${escapeHtml(cleanPrompt)}</td>
                    <td class="sched-col-status">${statusBadge}</td>
                    <td class="sched-col-actions">
                        <div style="display:inline-flex; gap:0.4rem; align-items:center;">
                            <button class="btn-secondary" style="padding:0.25rem 0.55rem; font-size:0.75rem; white-space:nowrap;" onclick="toggleSchedule('${j.id}')" title="${toggleBtnTitle}">${toggleBtnText}</button>
                            <button class="btn-secondary" style="padding:0.25rem 0.55rem; font-size:0.75rem; color:#ef4444; border-color:rgba(239,68,68,0.35); white-space:nowrap;" onclick="deleteSchedule('${j.id}', '${escapeHtml(j.name)}')" title="Stop and permanently remove this scheduled job">🗑️ Delete</button>
                            <button class="btn-secondary" style="padding:0.25rem 0.55rem; font-size:0.75rem; white-space:nowrap;" onclick="runScheduleNow('${j.id}', '${escapeHtml(j.name)}')" title="Run this workflow once right now">⚡ Run Now</button>
                        </div>
                    </td>
                </tr>
            `;
        });
    } catch (e) {
        console.error("Error loading schedules:", e);
    }
}

async function createSchedule() {
    const name = document.getElementById('schedName').value.trim();
    const cron = document.getElementById('schedCron').value.trim();
    const promptInput = document.getElementById('schedPrompt').value.trim();
    if (!name || !cron || !promptInput) return alert("Please fill in Job Name, Cron Expression, and Prompt.");

    // Clean outer quotes and normalize internal newlines into single spaces
    const prompt = promptInput.replace(/^["']|["']$/g, '').replace(/[\r\n]+/g, ' ').trim();

    try {
        const res = await fetch(`${API_BASE}/schedule`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: getActiveUserId(), name, cron_expression: cron, prompt })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Failed to create job' }));
            return alert(`Failed to schedule task: ${err.detail || res.statusText}`);
        }
        document.getElementById('schedName').value = '';
        document.getElementById('schedCron').value = '';
        document.getElementById('schedPrompt').value = '';
        await loadSchedules();
    } catch (e) {
        alert("Error scheduling task: " + e.message);
    }
}

async function toggleSchedule(jobId) {
    try {
        const res = await fetch(`${API_BASE}/schedule/${jobId}/toggle`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (res.ok) {
            await loadSchedules();
        } else {
            const err = await res.json().catch(() => ({ detail: 'Failed to toggle job' }));
            alert(`Error: ${err.detail || res.statusText}`);
        }
    } catch (e) {
        alert("Error toggling scheduled job: " + e.message);
    }
}

async function deleteSchedule(jobId, jobName) {
    if (!confirm(`Are you sure you want to stop and delete the scheduled job "${jobName}"?`)) {
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/schedule/${jobId}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            await loadSchedules();
        } else {
            const err = await res.json().catch(() => ({ detail: 'Failed to delete job' }));
            alert(`Error: ${err.detail || res.statusText}`);
        }
    } catch (e) {
        alert("Error deleting scheduled job: " + e.message);
    }
}

async function runScheduleNow(jobId, jobName) {
    try {
        const payload = {};
        if (attachedDatasetPath) {
            payload.attached_files = [attachedDatasetPath];
        }
        const res = await fetch(`${API_BASE}/schedule/${jobId}/trigger`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            alert(`⚡ Workflow "${jobName}" triggered successfully in background${attachedDatasetName ? ` with attached file "${attachedDatasetName}"` : ''}!`);
        } else {
            alert("Failed to trigger workflow run.");
        }
    } catch (e) {
        alert("Error triggering workflow: " + e.message);
    }
}

// Multimodal Voice Support (Web Speech API)
let recognition = null;
let isRecording = false;

function toggleVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        return alert("Speech Recognition is not supported by your browser. Please try Chrome or Edge.");
    }

    const btn = document.getElementById('btnVoiceInput');
    const promptInput = document.getElementById('taskPrompt');

    if (!recognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            promptInput.value = (promptInput.value ? promptInput.value + " " : "") + transcript;
            appendConsole("Voice", `Captured speech: "${transcript}"`, "info");
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            appendConsole("Voice", `Microphone error: ${event.error}`, "error");
            isRecording = false;
            btn.innerText = "🎙️ Voice Input";
            btn.style.borderColor = "var(--border-color)";
        };

        recognition.onend = () => {
            isRecording = false;
            btn.innerText = "🎙️ Voice Input";
            btn.style.borderColor = "var(--border-color)";
        };
    }

    if (!isRecording) {
        recognition.start();
        isRecording = true;
        btn.innerText = "🔴 Listening...";
        btn.style.borderColor = "var(--accent-rose)";
    } else {
        recognition.stop();
        isRecording = false;
        btn.innerText = "🎙️ Voice Input";
        btn.style.borderColor = "var(--border-color)";
    }
}

function speakOutput() {
    if (!window.speechSynthesis) {
        return alert("Text-to-Speech is not supported by your browser.");
    }

    const content = document.getElementById('resultsContent').innerText;
    if (!content) return;

    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        document.getElementById('btnSpeakOutput').innerText = "🔊 Read Aloud (TTS)";
        return;
    }

    const utterance = new SpeechSynthesisUtterance(content);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.onend = () => {
        document.getElementById('btnSpeakOutput').innerText = "🔊 Read Aloud (TTS)";
    };

    document.getElementById('btnSpeakOutput').innerText = "⏹️ Stop Reading";
    window.speechSynthesis.speak(utterance);
}

// --- Authentication & Session State Management ---

function getActiveUser() {
    const raw = localStorage.getItem('agentic_ai_user');
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch (e) {
        return null;
    }
}

function getActiveUserId() {
    const user = getActiveUser();
    return user && user.id ? user.id : 'default_user';
}

function getAuthToken() {
    return localStorage.getItem('agentic_ai_token') || null;
}

function updateAuthUI() {
    const user = getActiveUser();
    const token = getAuthToken();
    const loggedOutEl = document.getElementById('authLoggedOut');
    const loggedInEl = document.getElementById('authLoggedIn');
    const nameEl = document.getElementById('userDisplayName');
    const emailEl = document.getElementById('userDisplayEmail');
    const avatarEl = document.getElementById('userAvatar');

    if (user && user.id) {
        if (loggedOutEl) loggedOutEl.style.display = 'none';
        if (loggedInEl) loggedInEl.style.display = 'flex';
        
        const displayName = user.full_name || user.username || (user.email ? user.email.split('@')[0] : 'Account');
        if (nameEl) nameEl.innerText = displayName;
        if (emailEl) emailEl.innerText = user.email || '';
        if (avatarEl) {
            avatarEl.innerText = displayName.charAt(0).toUpperCase();
        }

        const btnAccount = document.getElementById('btnAccountMenu');
        if (btnAccount) {
            btnAccount.setAttribute('title', `Account: ${displayName} (${user.email || ''})`);
            btnAccount.setAttribute('aria-label', `Account Menu: ${displayName}`);
        }

        const createdIST = user.created_at ? formatIST(user.created_at) : 'N/A';
        const lastLoginIST = user.last_login ? formatIST(user.last_login) : (user.created_at ? formatIST(user.created_at) : 'Just now');

        // Populate Account Corner Dropdown Elements
        const accDropAvatar = document.getElementById('accDropdownAvatar');
        if (accDropAvatar) accDropAvatar.innerText = displayName.charAt(0).toUpperCase();
        const accDropName = document.getElementById('accDropdownName');
        if (accDropName) accDropName.innerText = displayName;
        const accDropEmail = document.getElementById('accDropdownEmail');
        if (accDropEmail) accDropEmail.innerText = user.email || 'N/A';
        const accDropId = document.getElementById('accDropdownId');
        if (accDropId) accDropId.innerText = user.id || 'N/A';
        const accDropCreated = document.getElementById('accDropdownCreated');
        if (accDropCreated) accDropCreated.innerText = createdIST;
        const accDropLastLogin = document.getElementById('accDropdownLastLogin');
        if (accDropLastLogin) accDropLastLogin.innerText = lastLoginIST;

        // Populate Account Tab Elements
        const tabAvatar = document.getElementById('accTabAvatar');
        if (tabAvatar) tabAvatar.innerText = displayName.charAt(0).toUpperCase();
        const tabName = document.getElementById('accTabName');
        if (tabName) tabName.innerText = displayName;
        const tabEmail = document.getElementById('accTabEmail');
        if (tabEmail) tabEmail.innerText = user.email || 'N/A';
        const tabUsername = document.getElementById('accTabUsername');
        if (tabUsername) tabUsername.innerText = user.username || user.id || 'N/A';
        const tabCreated = document.getElementById('accTabCreated');
        if (tabCreated) tabCreated.innerText = createdIST;
        const tabLastLogin = document.getElementById('accTabLastLogin');
        if (tabLastLogin) tabLastLogin.innerText = lastLoginIST;
        const tabToken = document.getElementById('accTabToken');
        if (tabToken) {
            const masked = token ? (token.length > 20 ? token.substring(0, 8) + '...' + token.substring(token.length - 8) : token) : 'Active Session';
            tabToken.innerText = masked;
        }
    } else {
        if (loggedOutEl) loggedOutEl.style.display = 'flex';
        if (loggedInEl) loggedInEl.style.display = 'none';
    }
}

function toggleAccountDropdown(event) {
    if (event) event.stopPropagation();
    const dropdown = document.getElementById('accountDropdownPanel');
    if (!dropdown) return;
    const isShowing = dropdown.style.display === 'block';
    dropdown.style.display = isShowing ? 'none' : 'block';
}

function closeAccountDropdown() {
    const dropdown = document.getElementById('accountDropdownPanel');
    if (dropdown) dropdown.style.display = 'none';
}

function switchToAccountTab() {
    closeAccountDropdown();
    const tabBtn = document.querySelector('.tab-btn[data-tab="tab-account"]');
    if (tabBtn) {
        tabBtn.click();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function openAuthModal(mode = 'signin') {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.style.display = 'flex';
    clearAuthAlert();
    switchAuthTab(mode);
}

function closeAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.style.display = 'none';
    clearAuthAlert();
}

function switchAuthTab(tab) {
    const btnIn = document.getElementById('tabBtnSignIn');
    const btnUp = document.getElementById('tabBtnSignUp');
    const formIn = document.getElementById('signInForm');
    const formUp = document.getElementById('signUpForm');
    const title = document.getElementById('authModalTitle');
    const subtitle = document.getElementById('authModalSubtitle');

    clearAuthAlert();

    if (tab === 'signin') {
        if (btnIn) btnIn.classList.add('active');
        if (btnUp) btnUp.classList.remove('active');
        if (formIn) formIn.style.display = 'block';
        if (formUp) formUp.style.display = 'none';
        if (title) title.innerText = 'Sign In to Platform';
        if (subtitle) subtitle.innerText = 'Access autonomous workflows, RAG knowledge, and personalization';
        setTimeout(() => document.getElementById('loginIdentifier')?.focus(), 100);
    } else {
        if (btnUp) btnUp.classList.add('active');
        if (btnIn) btnIn.classList.remove('active');
        if (formUp) formUp.style.display = 'block';
        if (formIn) formIn.style.display = 'none';
        if (title) title.innerText = 'Create an Account';
        if (subtitle) subtitle.innerText = 'Secure registration with salted password hashing';
        setTimeout(() => document.getElementById('signupEmail')?.focus(), 100);
    }
}

function showAuthAlert(message, type = 'error') {
    const banner = document.getElementById('authAlertBanner');
    if (!banner) return;
    banner.className = `auth-alert ${type}`;
    banner.innerText = message;
    banner.style.display = 'block';
}

function clearAuthAlert() {
    const banner = document.getElementById('authAlertBanner');
    if (!banner) return;
    banner.style.display = 'none';
    banner.innerText = '';
}

function togglePasswordVisibility(inputId, btnEl) {
    const input = document.getElementById(inputId);
    if (!input) return;
    if (input.type === 'password') {
        input.type = 'text';
        if (btnEl) btnEl.innerText = '🙈';
    } else {
        input.type = 'password';
        if (btnEl) btnEl.innerText = '👁️';
    }
}

// Live Validation Functions
function validateEmailFormat(email) {
    const re = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;
    return re.test(String(email).trim());
}

function validateSignupEmailLive() {
    const emailInput = document.getElementById('signupEmail');
    const hint = document.getElementById('signupEmailValidationHint');
    if (!emailInput || !hint) return false;

    const email = emailInput.value.trim();
    if (!email) {
        hint.innerText = '';
        emailInput.classList.remove('input-valid', 'input-invalid');
        return false;
    }

    if (validateEmailFormat(email)) {
        hint.innerText = '✓ Valid email format';
        hint.className = 'field-hint valid';
        emailInput.classList.add('input-valid');
        emailInput.classList.remove('input-invalid');
        return true;
    } else {
        hint.innerText = '✗ Please enter a valid email address (e.g. user@example.com)';
        hint.className = 'field-hint invalid';
        emailInput.classList.add('input-invalid');
        emailInput.classList.remove('input-valid');
        return false;
    }
}

function validateSignupPasswordLive() {
    const pwdInput = document.getElementById('signupPassword');
    const strengthBar = document.getElementById('strengthBarFill');
    const strengthText = document.getElementById('strengthText');
    if (!pwdInput) return false;

    const pwd = pwdInput.value;

    const hasLength = pwd.length >= 8;
    const hasUpper = /[A-Z]/.test(pwd);
    const hasLower = /[a-z]/.test(pwd);
    const hasDigit = /[0-9]/.test(pwd);
    const hasSpecial = /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?~`\\/]/.test(pwd);

    // Update Checklist DOM
    const updateCriterion = (id, met) => {
        const el = document.getElementById(id);
        if (el) {
            if (met) {
                el.classList.add('met');
                el.querySelector('.crit-icon').innerText = '✅';
            } else {
                el.classList.remove('met');
                el.querySelector('.crit-icon').innerText = '⚪';
            }
        }
    };

    updateCriterion('critLength', hasLength);
    updateCriterion('critUpper', hasUpper);
    updateCriterion('critLower', hasLower);
    updateCriterion('critDigit', hasDigit);
    updateCriterion('critSpecial', hasSpecial);

    // Strength Scoring
    const score = [hasLength, hasUpper, hasLower, hasDigit, hasSpecial].filter(Boolean).length;

    if (strengthBar && strengthText) {
        if (!pwd) {
            strengthBar.style.width = '0%';
            strengthBar.style.backgroundColor = 'transparent';
            strengthText.innerText = 'Enter password';
            strengthText.style.color = 'var(--text-muted)';
        } else if (score <= 2) {
            strengthBar.style.width = '33%';
            strengthBar.style.backgroundColor = 'var(--accent-rose)';
            strengthText.innerText = 'Weak';
            strengthText.style.color = 'var(--accent-rose)';
        } else if (score < 5) {
            strengthBar.style.width = '66%';
            strengthBar.style.backgroundColor = 'var(--accent-amber)';
            strengthText.innerText = 'Medium';
            strengthText.style.color = 'var(--accent-amber)';
        } else {
            strengthBar.style.width = '100%';
            strengthBar.style.backgroundColor = 'var(--accent-green)';
            strengthText.innerText = 'Strong';
            strengthText.style.color = 'var(--accent-green)';
        }
    }

    if (score === 5) {
        pwdInput.classList.add('input-valid');
        pwdInput.classList.remove('input-invalid');
    } else if (pwd.length > 0) {
        pwdInput.classList.add('input-invalid');
        pwdInput.classList.remove('input-valid');
    } else {
        pwdInput.classList.remove('input-valid', 'input-invalid');
    }

    // Also revalidate confirm password if it has value
    validateSignupConfirmLive();

    return score === 5;
}

function validateSignupConfirmLive() {
    const pwdInput = document.getElementById('signupPassword');
    const confirmInput = document.getElementById('signupConfirmPassword');
    const hint = document.getElementById('signupMatchHint');
    if (!confirmInput || !pwdInput || !hint) return false;

    const pwd = pwdInput.value;
    const confirm = confirmInput.value;

    if (!confirm) {
        hint.innerText = '';
        confirmInput.classList.remove('input-valid', 'input-invalid');
        return false;
    }

    if (pwd === confirm) {
        hint.innerText = '✓ Passwords match';
        hint.className = 'field-hint valid';
        confirmInput.classList.add('input-valid');
        confirmInput.classList.remove('input-invalid');
        return true;
    } else {
        hint.innerText = '✗ Passwords do not match';
        hint.className = 'field-hint invalid';
        confirmInput.classList.add('input-invalid');
        confirmInput.classList.remove('input-valid');
        return false;
    }
}

async function handleSignInSubmit(e) {
    e.preventDefault();
    clearAuthAlert();

    const identifier = document.getElementById('loginIdentifier')?.value.trim();
    const password = document.getElementById('loginPassword')?.value;
    const btn = document.getElementById('btnSignInSubmit');

    if (!identifier || !password) {
        showAuthAlert("Please provide both email/username and password.", "error");
        return;
    }

    btn.disabled = true;
    btn.querySelector('.btn-text').innerText = 'Signing in...';

    try {
        const res = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email_or_username: identifier,
                password: password
            })
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Authentication failed. Please verify credentials.");
        }

        // Store user and token
        localStorage.setItem('agentic_ai_user', JSON.stringify(data.user));
        if (data.token) {
            localStorage.setItem('agentic_ai_token', data.token);
        }

        updateAuthUI();
        closeAuthModal();
        appendConsole("Auth", `Welcome back, ${data.user.full_name || data.user.username || data.user.email}! Logged in successfully.`, "success");

        // Reload memories and tasks for this user
        loadMemories();
        loadTasks();
    } catch (err) {
        showAuthAlert(err.message, "error");
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').innerText = 'Sign In';
    }
}

async function handleSignUpSubmit(e) {
    e.preventDefault();
    clearAuthAlert();

    const email = document.getElementById('signupEmail')?.value.trim();
    const password = document.getElementById('signupPassword')?.value;
    const confirmPassword = document.getElementById('signupConfirmPassword')?.value;
    const fullName = document.getElementById('signupFullName')?.value.trim() || null;
    const username = document.getElementById('signupUsername')?.value.trim() || null;
    const btn = document.getElementById('btnSignUpSubmit');

    // Frontend pre-flight constraint check
    if (!validateEmailFormat(email)) {
        showAuthAlert("Please provide a valid email address.", "error");
        return;
    }
    if (!validateSignupPasswordLive()) {
        showAuthAlert("Password does not meet all security criteria (8+ chars, upper, lower, digit, special).", "error");
        return;
    }
    if (password !== confirmPassword) {
        showAuthAlert("Password confirmation does not match password.", "error");
        return;
    }

    btn.disabled = true;
    btn.querySelector('.btn-text').innerText = 'Creating account...';

    try {
        const res = await fetch(`${API_BASE}/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: email,
                password: password,
                confirm_password: confirmPassword,
                full_name: fullName,
                username: username
            })
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Sign up failed. Please check your information.");
        }

        // Auto login on successful signup
        localStorage.setItem('agentic_ai_user', JSON.stringify(data.user));
        if (data.token) {
            localStorage.setItem('agentic_ai_token', data.token);
        }

        updateAuthUI();
        closeAuthModal();
        appendConsole("Auth", `Account created and verified in SQLite database! Welcome, ${data.user.full_name || data.user.username || data.user.email}.`, "success");

        // Reload memories and tasks for this user
        loadMemories();
        loadTasks();
    } catch (err) {
        showAuthAlert(err.message, "error");
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').innerText = 'Create Account';
    }
}

async function logoutUser() {
    const user = getActiveUser();
    const name = user ? (user.full_name || user.username || user.email) : 'User';

    try {
        await fetch(`${API_BASE}/auth/logout`, { method: 'POST' });
    } catch (e) {
        // Continue logout locally
    }

    localStorage.removeItem('agentic_ai_user');
    localStorage.removeItem('agentic_ai_token');
    window.location.href = '/home';
}

// Global modal and dropdown esc key handler
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAccountDropdown();
        closeAuthModal();
        if (typeof closeTaskInspectionModal === 'function') {
            closeTaskInspectionModal();
        }
        if (typeof toggleChatHistoryModal === 'function') {
            const histModal = document.getElementById('chatHistoryModal');
            if (histModal && histModal.style.display === 'flex') {
                histModal.style.display = 'none';
            }
        }
    }
});

// Click outside account dropdown to close
window.addEventListener('click', (e) => {
    const dropdown = document.getElementById('accountDropdownPanel');
    const btn = document.getElementById('btnAccountMenu');
    if (dropdown && dropdown.style.display === 'block') {
        if (!dropdown.contains(e.target) && (!btn || !btn.contains(e.target))) {
            dropdown.style.display = 'none';
        }
    }
});



