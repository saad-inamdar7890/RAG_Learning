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
async function runEvaluation() {
    const strategyInput = document.querySelector('input[name="strategy"]:checked');
    if (!strategyInput) return;
    const strategy = strategyInput.value;

    const btn = document.getElementById('run-eval-btn');
    const runLabel = document.getElementById('run-label');
    const runSpinner = document.getElementById('run-spinner');
    const progressDiv = document.getElementById('eval-progress');
    const progFill = document.getElementById('prog-fill');
    const progLabel = document.getElementById('prog-label');
    const scoreSummary = document.getElementById('score-summary');
    const resultsTable = document.getElementById('results-table');
    const resultsBody = document.getElementById('results-body');

    // Disable UI
    btn.disabled = true;
    runLabel.classList.add('hidden');
    runSpinner.classList.remove('hidden');
    scoreSummary.classList.add('hidden');
    resultsTable.classList.add('hidden');

    // Show progress bar
    progressDiv.classList.remove('hidden');
    let fakeProgress = 0;
    const progressInterval = setInterval(() => {
        if (fakeProgress < 85) {
            fakeProgress += Math.random() * 3;
            progFill.style.width = fakeProgress + '%';
            const steps = [
                'Chunking documents…',
                'Building FAISS vector index…',
                'Building BM25 lexical index…',
                'Running benchmark queries…',
                'Evaluating with LLM-as-a-judge…',
            ];
            const stepIdx = Math.min(Math.floor(fakeProgress / 20), steps.length - 1);
            progLabel.textContent = steps[stepIdx];
        }
    }, 800);

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy })
        });

        clearInterval(progressInterval);
        progFill.style.width = '100%';
        progLabel.textContent = 'Complete!';

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Unknown server error');
        }

        const data = await response.json();
        setTimeout(() => progressDiv.classList.add('hidden'), 800);

        // Fill score summary
        document.getElementById('score-val').textContent = `${data.passed} / ${data.total}`;
        document.getElementById('score-pct').textContent = `${data.score_pct}%`;
        document.getElementById('score-latency').textContent = `${data.avg_latency_s}s`;
        const strategyLabels = {
            normal: 'Normal',
            semantic: 'Semantic',
            parent_child: 'Parent-Child'
        };
        document.getElementById('score-strategy').textContent = strategyLabels[data.strategy] || data.strategy;
        scoreSummary.classList.remove('hidden');

        // Criteria breakdown bar
        const criteriaGrid = document.getElementById('criteria-grid');
        const criteriaLabels = { latency: 'Latency', citation: 'Citations', faithfulness: 'Faithfulness', topical: 'Topical' };
        criteriaGrid.innerHTML = '';
        Object.entries(data.criteria_pass_counts).forEach(([key, count]) => {
            const pct = Math.round(count / data.total * 100);
            criteriaGrid.innerHTML += `
                <div class="criteria-item">
                    <div class="criteria-label">${criteriaLabels[key] || key}</div>
                    <div class="criteria-bar-wrap">
                        <div class="criteria-bar-fill" style="width:${pct}%"></div>
                    </div>
                    <div class="criteria-count">${count}/${data.total} (${pct}%)</div>
                </div>`;
        });
        document.getElementById('criteria-breakdown').classList.remove('hidden');

        // Fill results table
        resultsBody.innerHTML = '';
        data.results.forEach((r, i) => {
            const c = r.criteria || {};
            const cell = (crit) => {
                const ok = c[crit]?.passed;
                return `<td title="${c[crit]?.detail || ''}"><span class="${ok ? 'badge-pass' : 'badge-fail'}">${ok ? '\u2713' : '\u2717'}</span></td>`;
            };
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${i + 1}</td>
                <td>${r.query}</td>
                ${cell('latency')}
                ${cell('citation')}
                ${cell('faithfulness')}
                ${cell('topical')}
                <td><span class="${r.passed ? 'badge-pass' : 'badge-fail'}">${r.passed ? 'PASS' : 'FAIL'}</span></td>`;
            resultsBody.appendChild(row);
        });
        resultsTable.classList.remove('hidden');

    } catch (err) {
        clearInterval(progressInterval);
        progFill.style.width = '0%';
        progLabel.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
        runLabel.classList.remove('hidden');
        runSpinner.classList.add('hidden');
    }
}
