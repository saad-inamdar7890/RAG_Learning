// ── Tab switching ──────────────────────────────────────────────────────────
function showTab(tab) {
    document.getElementById('tab-chat').classList.toggle('hidden', tab !== 'chat');
    document.getElementById('tab-eval').classList.toggle('hidden', tab !== 'eval');
    document.getElementById('tab-metrics').classList.toggle('hidden', tab !== 'metrics');
    document.getElementById('nav-chat').classList.toggle('active', tab === 'chat');
    document.getElementById('nav-eval').classList.toggle('active', tab === 'eval');
    document.getElementById('nav-metrics').classList.toggle('active', tab === 'metrics');
    document.getElementById('history-panel').classList.toggle('hidden', tab !== 'chat');
    if (tab === 'metrics') loadMetrics();
}

async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();

        document.getElementById('m-total').textContent = data.total_requests || 0;
        document.getElementById('m-p50').textContent = data.latency?.p50_s != null ? data.latency.p50_s + 's' : '—';
        document.getElementById('m-p95').textContent = data.latency?.p95_s != null ? data.latency.p95_s + 's' : '—';
        document.getElementById('m-tokens').textContent = data.avg_tokens_per_request || '—';

        const stepLabels = { embed_s: 'Embed', faiss_s: 'FAISS Search', bm25_hybrid_s: 'BM25 Hybrid', rerank_s: 'Rerank', generate_s: 'LLM Generate' };
        const stepsEl = document.getElementById('metrics-steps');
        const steps = data.step_averages_s || {};
        if (Object.keys(steps).length === 0) {
            stepsEl.innerHTML = '<p style="color:#aaa;font-size:0.82rem">No data yet — ask a question in the Chat tab first.</p>';
        } else {
            const maxVal = Math.max(...Object.values(steps));
            stepsEl.innerHTML = Object.entries(steps).map(([k, v]) => {
                const pct = Math.round((v / maxVal) * 100);
                return `<div class="criteria-item">
                    <div class="criteria-label">${stepLabels[k] || k}</div>
                    <div class="criteria-bar-wrap"><div class="criteria-bar-fill" style="width:${pct}%"></div></div>
                    <div class="criteria-count">${v}s</div>
                </div>`;
            }).join('');
        }

        const recentEl = document.getElementById('metrics-recent');
        const recent = (data.recent_requests || []).slice().reverse();
        recentEl.innerHTML = recent.map((r, i) =>
            `<tr><td>${i + 1}</td><td>${r.total_s}s</td><td>${r.tokens || '—'}</td></tr>`
        ).join('') || '<tr><td colspan="3" style="color:#aaa">No requests yet.</td></tr>';

    } catch (e) {
        console.error('Failed to load metrics:', e);
    }
}

// ── Strategy card radio sync ──────────────────────────────────────────────
document.querySelectorAll('.strategy-card').forEach(card => {
    card.addEventListener('click', () => {
        card.querySelector('input[type=radio]').checked = true;
    });
});

// ── CHAT ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('query-form');
    const input = document.getElementById('query-input');
    const chatWindow = document.getElementById('chat-window');
    const sendButton = document.getElementById('send-button');
    const historyList = document.getElementById('history');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query) return;

        input.value = '';
        input.disabled = true;
        sendButton.disabled = true;

        addMessage(query, 'user');
        addHistoryItem(query);
        const typingId = addTypingIndicator();

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            const data = await response.json();
            removeElement(typingId);
            addMessage(data.answer, 'ai', data.sources);
        } catch (error) {
            removeElement(typingId);
            addMessage('An error occurred. Make sure the backend and Ollama are running.', 'system');
        } finally {
            input.disabled = false;
            sendButton.disabled = false;
            input.focus();
        }
    });

    function addMessage(text, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        const avatarText = sender === 'user' ? 'U' : 'AI';
        const avatarClass = sender === 'user' ? 'user-avatar' : 'ai-avatar';
        let contentHTML = `<p>${formatText(text)}</p>`;

        if (sources && sources.length > 0) {
            let citationsHTML = '<div class="citations-block">';
            sources.forEach(src => {
                citationsHTML += `
                    <div class="citation-pill">
                        <span class="citation-meta">[${src.citation_number}] Doc: ${src.doc_id} | Page: ${src.page_number}</span>
                        <span class="citation-text">"${src.text}"</span>
                    </div>`;
            });
            citationsHTML += '</div>';
            contentHTML += citationsHTML;
        }

        messageDiv.innerHTML = `
            <div class="avatar ${avatarClass}">${avatarText}</div>
            <div class="content">${contentHTML}</div>`;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message ai-message';
        div.id = id;
        div.innerHTML = `
            <div class="avatar ai-avatar">AI</div>
            <div class="content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
        chatWindow.appendChild(div);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function addHistoryItem(query) {
        const emptyState = historyList.querySelector('.empty-state');
        if (emptyState) emptyState.remove();
        const li = document.createElement('li');
        li.textContent = query;
        li.title = query;
        historyList.prepend(li);
    }

    function formatText(text) {
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }
});

// ── EVALUATION ────────────────────────────────────────────────────────────

let evalQueries = [];

async function loadEvalQueries() {
    try {
        const res = await fetch('/api/eval-queries');
        const data = await res.json();
        evalQueries = data.queries || [];
        renderEvalList();
    } catch (e) {
        document.getElementById('eval-questions-list').innerHTML = `<p style="color:red;text-align:center">Failed to load queries.</p>`;
    }
}

function renderEvalList() {
    const list = document.getElementById('eval-questions-list');
    list.innerHTML = '';
    
    evalQueries.forEach((qObj, index) => {
        const card = document.createElement('div');
        card.className = 'q-card';
        card.dataset.index = index;
        
        card.innerHTML = `
            <div class="q-card-header" onclick="toggleCard(${index})">
                <div class="q-card-title">${index + 1}. ${qObj.query}</div>
                <div class="q-card-toggle">▼</div>
            </div>
            <div class="q-card-body" id="q-body-${index}">
                <div class="q-action-bar">
                    <button class="q-btn q-btn-generate" id="btn-gen-${index}" onclick="generateAnswer(${index})">
                        <span>Generate Answer</span>
                    </button>
                    <button class="q-btn q-btn-judge hidden" id="btn-judge-${index}" onclick="judgeAnswer(${index})">
                        <span>Evaluate Answer</span>
                    </button>
                </div>
                <div class="q-answer-box empty" id="q-answer-${index}">Answer will appear here...</div>
                <div class="q-judge-results hidden" id="q-results-${index}">
                    <div class="q-judge-item"><span class="q-judge-label">Latency</span> <span class="q-judge-value" id="res-lat-${index}"></span></div>
                    <div class="q-judge-item"><span class="q-judge-label">Citations</span> <span class="q-judge-value" id="res-cit-${index}"></span></div>
                    <div class="q-judge-item"><span class="q-judge-label">Faithfulness</span> <span class="q-judge-value" id="res-fai-${index}"></span></div>
                    <div class="q-judge-item"><span class="q-judge-label">Topicality</span> <span class="q-judge-value" id="res-top-${index}"></span></div>
                </div>
            </div>
        `;
        list.appendChild(card);
    });
}

function toggleCard(index) {
    const cards = document.querySelectorAll('.q-card');
    cards.forEach((c, i) => {
        if (i === index) {
            c.classList.toggle('expanded');
        } else {
            c.classList.remove('expanded');
        }
    });
}

// Store intermediate state for each question
const evalState = {};

async function generateAnswer(index) {
    const strategyInput = document.querySelector('input[name="strategy"]:checked');
    if (!strategyInput) return;
    const strategy = strategyInput.value;
    const qObj = evalQueries[index];
    
    const btnGen = document.getElementById(`btn-gen-${index}`);
    const btnJudge = document.getElementById(`btn-judge-${index}`);
    const answerBox = document.getElementById(`q-answer-${index}`);
    const resultsBox = document.getElementById(`q-results-${index}`);
    
    btnGen.disabled = true;
    btnGen.innerHTML = `<span class="spinner"></span> Generating...`;
    btnJudge.classList.add('hidden');
    resultsBox.classList.add('hidden');
    answerBox.className = 'q-answer-box';
    answerBox.innerHTML = '<i>Generating...</i>';

    try {
        const res = await fetch('/api/eval-generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy, query: qObj.query })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Server error');

        answerBox.textContent = data.answer;
        evalState[index] = data; // Save retrieved_context, answer, latency_s
        
        btnGen.innerHTML = `Regenerate`;
        btnJudge.classList.remove('hidden');
    } catch (e) {
        answerBox.innerHTML = `<span style="color:red">Error: ${e.message}</span>`;
        btnGen.innerHTML = `Generate Answer`;
    } finally {
        btnGen.disabled = false;
    }
}

async function judgeAnswer(index) {
    const state = evalState[index];
    if (!state) return;
    const qObj = evalQueries[index];

    const btnJudge = document.getElementById(`btn-judge-${index}`);
    const resultsBox = document.getElementById(`q-results-${index}`);
    
    btnJudge.disabled = true;
    btnJudge.innerHTML = `<span class="spinner"></span> Evaluating...`;
    
    // reset labels
    ['lat', 'cit', 'fai', 'top'].forEach(k => {
        document.getElementById(`res-${k}-${index}`).innerHTML = '<span class="badge-pending">...</span>';
    });
    resultsBox.classList.remove('hidden');

    try {
        const res = await fetch('/api/eval-judge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: qObj.query,
                answer: state.answer,
                expected_topic: state.expected_topic,
                retrieved_context: state.retrieved_context,
                generate_latency_s: state.latency_s
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Server error');

        const c = data.criteria;
        const setBadge = (id, ok, detail) => {
            const el = document.getElementById(`res-${id}-${index}`);
            el.innerHTML = `<span class="${ok ? 'badge-pass' : 'badge-fail'}">${ok ? 'PASS' : 'FAIL'}</span> <span style="font-size:0.75rem;color:#777;margin-left:6px">${detail}</span>`;
        };

        setBadge('lat', c.latency.passed, c.latency.detail);
        setBadge('cit', c.citation.passed, c.citation.detail);
        setBadge('fai', c.faithfulness.passed, c.faithfulness.detail);
        setBadge('top', c.topical.passed, c.topical.detail);

        btnJudge.innerHTML = data.passed ? `✅ All Passed` : `❌ Failed`;
    } catch (e) {
        alert("Judge error: " + e.message);
        btnJudge.innerHTML = `Evaluate Answer`;
    } finally {
        btnJudge.disabled = false;
    }
}

// Initial load
loadEvalQueries();
