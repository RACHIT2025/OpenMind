/**
 * OpenMind Chat Application
 *
 * Full-featured chat UI with:
 * - Streaming API integration (SSE)
 * - Conversation management (create, switch, delete)
 * - Local storage persistence
 * - Dark/light theme toggle
 * - Settings panel with model parameters
 * - Markdown rendering for code blocks
 * - Auto-resizing textarea
 */

// ─── State ───────────────────────────────────────────────
const state = {
    conversations: [],
    currentConversationId: null,
    isGenerating: false,
    abortController: null,
    theme: localStorage.getItem('openmind-theme') || 'dark',
    settings: {
        model: 'openmind-125m',
        temperature: 0.7,
        maxTokens: 512,
        topP: 0.9,
        systemPrompt: 'You are OpenMind, a helpful, harmless, and honest AI assistant.',
        apiEndpoint: 'http://localhost:8000',
    },
};

// ─── DOM Elements ────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elements = {
    sidebar: $('#sidebar'),
    conversationList: $('#conversationList'),
    welcomeScreen: $('#welcomeScreen'),
    chatMessages: $('#chatMessages'),
    messageInput: $('#messageInput'),
    sendBtn: $('#sendBtn'),
    stopBtn: $('#stopBtn'),
    newChatBtn: $('#newChatBtn'),
    searchInput: $('#searchInput'),
    settingsBtn: $('#settingsBtn'),
    settingsOverlay: $('#settingsOverlay'),
    closeSettings: $('#closeSettings'),
    themeToggle: $('#themeToggle'),
    mobileMenuBtn: $('#mobileMenuBtn'),
    modelSelect: $('#modelSelect'),
    tempSlider: $('#tempSlider'),
    tempValue: $('#tempValue'),
    maxTokensSlider: $('#maxTokensSlider'),
    maxTokensValue: $('#maxTokensValue'),
    topPSlider: $('#topPSlider'),
    topPValue: $('#topPValue'),
    systemPrompt: $('#systemPrompt'),
    apiEndpoint: $('#apiEndpoint'),
};

// ─── Initialization ──────────────────────────────────────
function init() {
    loadState();
    applyTheme();
    setupEventListeners();
    renderConversationList();
    updateChatView();
}

// ─── State Management ────────────────────────────────────
function loadState() {
    try {
        const saved = localStorage.getItem('openmind-conversations');
        if (saved) {
            state.conversations = JSON.parse(saved);
        }
        const savedSettings = localStorage.getItem('openmind-settings');
        if (savedSettings) {
            Object.assign(state.settings, JSON.parse(savedSettings));
        }
        const savedCurrentId = localStorage.getItem('openmind-current-id');
        if (savedCurrentId && state.conversations.find(c => c.id === savedCurrentId)) {
            state.currentConversationId = savedCurrentId;
        }
    } catch (e) {
        console.error('Failed to load state:', e);
    }
}

function saveState() {
    try {
        localStorage.setItem('openmind-conversations', JSON.stringify(state.conversations));
        localStorage.setItem('openmind-settings', JSON.stringify(state.settings));
        if (state.currentConversationId) {
            localStorage.setItem('openmind-current-id', state.currentConversationId);
        }
    } catch (e) {
        console.error('Failed to save state:', e);
    }
}

// ─── Conversation Management ─────────────────────────────
function createConversation() {
    const conv = {
        id: 'conv-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8),
        title: 'New Chat',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
    };
    state.conversations.unshift(conv);
    state.currentConversationId = conv.id;
    saveState();
    renderConversationList();
    updateChatView();
    elements.messageInput.focus();
    return conv;
}

function getCurrentConversation() {
    return state.conversations.find(c => c.id === state.currentConversationId);
}

function switchConversation(id) {
    state.currentConversationId = id;
    saveState();
    renderConversationList();
    updateChatView();

    // Close mobile sidebar
    elements.sidebar.classList.remove('open');
}

function deleteConversation(id) {
    state.conversations = state.conversations.filter(c => c.id !== id);
    if (state.currentConversationId === id) {
        state.currentConversationId = state.conversations[0]?.id || null;
    }
    saveState();
    renderConversationList();
    updateChatView();
}

function updateConversationTitle(conv) {
    if (conv.messages.length > 0 && conv.title === 'New Chat') {
        const firstMsg = conv.messages[0].content;
        conv.title = firstMsg.slice(0, 40) + (firstMsg.length > 40 ? '...' : '');
    }
}

// ─── Rendering ───────────────────────────────────────────
function renderConversationList() {
    const search = elements.searchInput.value.toLowerCase();
    const filtered = state.conversations.filter(c =>
        !search || c.title.toLowerCase().includes(search)
    );

    elements.conversationList.innerHTML = filtered.map(conv => `
        <button class="conversation-item ${conv.id === state.currentConversationId ? 'active' : ''}"
                onclick="switchConversation('${conv.id}')">
            <span class="conv-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
            </span>
            <span class="conv-title">${escapeHtml(conv.title)}</span>
            <span class="conv-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')" title="Delete">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 6h18M8 6V4h8v2M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>
                </svg>
            </span>
        </button>
    `).join('');
}

function updateChatView() {
    const conv = getCurrentConversation();

    if (!conv || conv.messages.length === 0) {
        elements.welcomeScreen.classList.remove('hidden');
        elements.chatMessages.classList.remove('active');
    } else {
        elements.welcomeScreen.classList.add('hidden');
        elements.chatMessages.classList.add('active');
        renderMessages(conv.messages);
    }
}

function renderMessages(messages) {
    elements.chatMessages.innerHTML = messages.map((msg, i) => `
        <div class="message ${msg.role}" id="msg-${i}">
            <div class="message-avatar">
                ${msg.role === 'user' ? 'U' : '✦'}
            </div>
            <div class="message-content">
                <div class="message-role">${msg.role === 'user' ? 'You' : 'OpenMind'}</div>
                <div class="message-text">${formatMessage(msg.content)}</div>
            </div>
        </div>
    `).join('');

    scrollToBottom();
}

function appendMessage(role, content) {
    const conv = getCurrentConversation();
    if (!conv) return;

    conv.messages.push({ role, content });
    updateConversationTitle(conv);
    conv.updatedAt = new Date().toISOString();
    saveState();

    elements.welcomeScreen.classList.add('hidden');
    elements.chatMessages.classList.add('active');

    const msgIdx = conv.messages.length - 1;
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.id = `msg-${msgIdx}`;
    msgDiv.innerHTML = `
        <div class="message-avatar">${role === 'user' ? 'U' : '✦'}</div>
        <div class="message-content">
            <div class="message-role">${role === 'user' ? 'You' : 'OpenMind'}</div>
            <div class="message-text">${formatMessage(content)}</div>
        </div>
    `;

    elements.chatMessages.appendChild(msgDiv);
    scrollToBottom();
    renderConversationList();

    return msgDiv;
}

function appendStreamingMessage() {
    const conv = getCurrentConversation();
    if (!conv) return null;

    conv.messages.push({ role: 'assistant', content: '' });
    const msgIdx = conv.messages.length - 1;

    elements.welcomeScreen.classList.add('hidden');
    elements.chatMessages.classList.add('active');

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';
    msgDiv.id = `msg-${msgIdx}`;
    msgDiv.innerHTML = `
        <div class="message-avatar">✦</div>
        <div class="message-content">
            <div class="message-role">OpenMind</div>
            <div class="message-text">
                <div class="typing-indicator">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        </div>
    `;

    elements.chatMessages.appendChild(msgDiv);
    scrollToBottom();

    return { element: msgDiv, msgIdx };
}

function updateStreamingMessage(msgData, text) {
    const conv = getCurrentConversation();
    if (!conv || !msgData) return;

    conv.messages[msgData.msgIdx].content = text;
    const textEl = msgData.element.querySelector('.message-text');
    textEl.innerHTML = formatMessage(text);
    scrollToBottom();
}

// ─── API Integration ─────────────────────────────────────
async function sendMessage(userText) {
    if (!userText.trim() || state.isGenerating) return;

    // Ensure we have a conversation
    let conv = getCurrentConversation();
    if (!conv) {
        conv = createConversation();
    }

    // Add user message
    appendMessage('user', userText.trim());

    // Clear input
    elements.messageInput.value = '';
    autoResizeTextarea();

    // Update UI state
    state.isGenerating = true;
    elements.sendBtn.disabled = true;
    elements.stopBtn.classList.remove('hidden');

    // Start streaming response
    const msgData = appendStreamingMessage();
    let fullResponse = '';

    try {
        state.abortController = new AbortController();

        const messages = conv.messages
            .filter(m => m.content) // Skip empty messages
            .slice(0, -1)          // Exclude the empty assistant message we just added
            .map(m => ({ role: m.role, content: m.content }));

        // Add system prompt
        if (state.settings.systemPrompt) {
            messages.unshift({ role: 'system', content: state.settings.systemPrompt });
        }

        const response = await fetch(`${state.settings.apiEndpoint}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: state.settings.model,
                messages: messages,
                temperature: state.settings.temperature,
                top_p: state.settings.topP,
                max_tokens: state.settings.maxTokens,
                stream: true,
            }),
            signal: state.abortController.signal,
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6).trim();
                    if (data === '[DONE]') break;

                    try {
                        const parsed = JSON.parse(data);
                        const delta = parsed.choices?.[0]?.delta;
                        if (delta?.content) {
                            fullResponse += delta.content;
                            updateStreamingMessage(msgData, fullResponse);
                        }
                    } catch (e) {
                        // Skip malformed chunks
                    }
                }
            }
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            fullResponse += '\n\n*[Generation stopped]*';
        } else {
            console.error('API Error:', error);
            // Show a friendly demo response when API is not available
            fullResponse = generateDemoResponse(userText);
        }
    }

    // Finalize
    if (msgData) {
        updateStreamingMessage(msgData, fullResponse || 'I apologize, but I encountered an error generating a response.');
    }

    state.isGenerating = false;
    state.abortController = null;
    elements.sendBtn.disabled = false;
    elements.stopBtn.classList.add('hidden');
    saveState();
    renderConversationList();
}

function stopGenerating() {
    if (state.abortController) {
        state.abortController.abort();
    }
}

/**
 * Generate a demo response when the API server is not running.
 * This allows the UI to be tested standalone.
 */
function generateDemoResponse(userText) {
    const query = userText.toLowerCase();

    if (query.includes('hello') || query.includes('hi')) {
        return "Hello! 👋 I'm **OpenMind**, your open-source AI assistant. I'm currently running in demo mode since the API server isn't connected. Once you train the model and start the server, I'll provide real AI-generated responses!\n\nHere's what you can do:\n1. Train the model using the provided scripts\n2. Start the API server with `python src/inference/api_server.py --model your_model_path`\n3. Come back here and chat with your very own AI!";
    }

    if (query.includes('code') || query.includes('python') || query.includes('function')) {
        return "Here's an example of what I can do when connected to the model:\n\n```python\ndef fibonacci(n):\n    \"\"\"Generate Fibonacci sequence up to n terms.\"\"\"\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib\n\n# Example usage\nprint(fibonacci(10))\n# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n```\n\n*Note: This is a demo response. Connect the API server for real AI-generated code!*";
    }

    if (query.includes('quantum')) {
        return "## Quantum Computing in Simple Terms 🔬\n\nImagine you have a coin:\n\n- **Classical computing**: The coin is either heads (0) or tails (1)\n- **Quantum computing**: The coin is spinning in the air — it's *both* heads AND tails at the same time! This is called **superposition**.\n\n### Key Concepts:\n\n1. **Qubits** - Quantum bits that can be 0, 1, or both simultaneously\n2. **Superposition** - Being in multiple states at once\n3. **Entanglement** - Two qubits linked so that measuring one instantly affects the other\n4. **Quantum Gates** - Operations that manipulate qubits\n\n> \"If you think you understand quantum mechanics, you don't understand quantum mechanics.\" — Richard Feynman\n\n*This is a demo response. Start the API server for real AI answers!*";
    }

    return `Thanks for your message! 🚀\n\nI'm **OpenMind** running in **demo mode**. The API server isn't connected yet, but here's what I understood from your message:\n\n> "${userText.slice(0, 100)}${userText.length > 100 ? '...' : ''}"\n\n### To get real AI responses:\n\n1. **Train** the 125M parameter model using Google Colab (see the training notebook)\n2. **Start the server**: \`python src/inference/api_server.py --model models/checkpoints/openmind-125m\`\n3. **Update the API endpoint** in Settings to point to your server\n4. **Chat away!** 🎉\n\nThe UI is fully functional — conversations are saved locally, you can create multiple chats, toggle dark/light mode, and adjust generation parameters.\n\n*Demo mode • Model not connected*`;
}

// ─── Message Formatting ──────────────────────────────────
function formatMessage(text) {
    if (!text) return '';

    // Escape HTML first
    let html = escapeHtml(text);

    // Code blocks (```lang\ncode\n```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code (`code`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold (**text**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic (*text*)
    html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');

    // Headers (## text)
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Blockquotes (> text)
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered lists (- item)
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

    // Ordered lists (1. item)
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Line breaks
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph
    if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
    }

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Utilities ───────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    });
}

function autoResizeTextarea() {
    const textarea = elements.messageInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function applyTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    const btn = elements.themeToggle;
    if (btn) {
        btn.querySelector('span').textContent = state.theme === 'dark' ? 'Light Mode' : 'Dark Mode';
    }
}

function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('openmind-theme', state.theme);
    applyTheme();
}

// ─── Suggestion Handler ──────────────────────────────────
function sendSuggestion(text) {
    elements.messageInput.value = text;
    sendMessage(text);
}

// Make it globally accessible
window.sendSuggestion = sendSuggestion;
window.switchConversation = switchConversation;
window.deleteConversation = deleteConversation;

// ─── Event Listeners ─────────────────────────────────────
function setupEventListeners() {
    // Send message
    elements.sendBtn.addEventListener('click', () => {
        sendMessage(elements.messageInput.value);
    });

    // Enter to send (Shift+Enter for new line)
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(elements.messageInput.value);
        }
    });

    // Auto-resize textarea
    elements.messageInput.addEventListener('input', autoResizeTextarea);

    // Stop generating
    elements.stopBtn.addEventListener('click', stopGenerating);

    // New chat
    elements.newChatBtn.addEventListener('click', createConversation);

    // Search conversations
    elements.searchInput.addEventListener('input', renderConversationList);

    // Settings
    elements.settingsBtn.addEventListener('click', () => {
        elements.settingsOverlay.classList.remove('hidden');
        // Sync settings to UI
        elements.modelSelect.value = state.settings.model;
        elements.tempSlider.value = state.settings.temperature;
        elements.tempValue.textContent = state.settings.temperature;
        elements.maxTokensSlider.value = state.settings.maxTokens;
        elements.maxTokensValue.textContent = state.settings.maxTokens;
        elements.topPSlider.value = state.settings.topP;
        elements.topPValue.textContent = state.settings.topP;
        elements.systemPrompt.value = state.settings.systemPrompt;
        elements.apiEndpoint.value = state.settings.apiEndpoint;
    });

    elements.closeSettings.addEventListener('click', () => {
        elements.settingsOverlay.classList.add('hidden');
    });

    elements.settingsOverlay.addEventListener('click', (e) => {
        if (e.target === elements.settingsOverlay) {
            elements.settingsOverlay.classList.add('hidden');
        }
    });

    // Settings controls
    elements.modelSelect.addEventListener('change', (e) => {
        state.settings.model = e.target.value;
        saveState();
    });

    elements.tempSlider.addEventListener('input', (e) => {
        state.settings.temperature = parseFloat(e.target.value);
        elements.tempValue.textContent = state.settings.temperature;
        saveState();
    });

    elements.maxTokensSlider.addEventListener('input', (e) => {
        state.settings.maxTokens = parseInt(e.target.value);
        elements.maxTokensValue.textContent = state.settings.maxTokens;
        saveState();
    });

    elements.topPSlider.addEventListener('input', (e) => {
        state.settings.topP = parseFloat(e.target.value);
        elements.topPValue.textContent = state.settings.topP;
        saveState();
    });

    elements.systemPrompt.addEventListener('change', (e) => {
        state.settings.systemPrompt = e.target.value;
        saveState();
    });

    elements.apiEndpoint.addEventListener('change', (e) => {
        state.settings.apiEndpoint = e.target.value;
        saveState();
    });

    // Theme toggle
    elements.themeToggle.addEventListener('click', toggleTheme);

    // Mobile menu
    elements.mobileMenuBtn.addEventListener('click', () => {
        elements.sidebar.classList.toggle('open');
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 &&
            elements.sidebar.classList.contains('open') &&
            !elements.sidebar.contains(e.target) &&
            !elements.mobileMenuBtn.contains(e.target)) {
            elements.sidebar.classList.remove('open');
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+N: New chat
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            createConversation();
        }
        // Escape: Close settings / Stop generating
        if (e.key === 'Escape') {
            if (!elements.settingsOverlay.classList.contains('hidden')) {
                elements.settingsOverlay.classList.add('hidden');
            } else if (state.isGenerating) {
                stopGenerating();
            }
        }
    });
}

// ─── Boot ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
