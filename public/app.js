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

        // Clear input
        input.value = '';
        input.disabled = true;
        sendButton.disabled = true;

        // Add user message to UI
        addMessage(query, 'user');
        addHistoryItem(query);

        // Show typing indicator
        const typingId = addTypingIndicator();

        try {
            // Call FastAPI Backend
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Remove typing indicator
            removeElement(typingId);
            
            // Render AI response with citations
            addMessage(data.answer, 'ai', data.sources);

        } catch (error) {
            console.error('Error:', error);
            removeElement(typingId);
            addMessage('An error occurred while fetching the answer. Make sure the backend and Ollama are running.', 'system');
        } finally {
            input.disabled = false;
            sendButton.disabled = false;
            input.focus();
        }
    });

    function addMessage(text, sender, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        let avatarText = sender === 'user' ? 'U' : 'AI';
        let avatarClass = sender === 'user' ? 'user-avatar' : 'ai-avatar';
        if (sender === 'system') {
            avatarText = '!';
            avatarClass = 'user-avatar';
        }

        let contentHTML = `<p>${formatText(text)}</p>`;
        
        if (sources && sources.length > 0) {
            let citationsHTML = '<div class="citations-block">';
            sources.forEach(src => {
                citationsHTML += `
                    <div class="citation-pill">
                        <span class="citation-meta">[${src.citation_number}] Doc: ${src.doc_id} | Page: ${src.page_number}</span>
                        <span class="citation-text">"${src.text}"</span>
                    </div>
                `;
            });
            citationsHTML += '</div>';
            contentHTML += citationsHTML;
        }

        messageDiv.innerHTML = `
            <div class="avatar ${avatarClass}">${avatarText}</div>
            <div class="content">
                ${contentHTML}
            </div>
        `;

        chatWindow.appendChild(messageDiv);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ai-message`;
        messageDiv.id = id;
        
        messageDiv.innerHTML = `
            <div class="avatar ai-avatar">AI</div>
            <div class="content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        
        chatWindow.appendChild(messageDiv);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function addHistoryItem(query) {
        // Remove empty state if present
        const emptyState = historyList.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const li = document.createElement('li');
        li.textContent = query;
        li.title = query;
        historyList.prepend(li);
    }

    // Basic markdown formatting for the answer
    function formatText(text) {
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }
});
