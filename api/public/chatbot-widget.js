// ============================================================
//  BMC Control-M Automation Chatbot Widget
//  Isolated Plug-and-Play Frontend Component
// ============================================================

class BmcChatbotWidget extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.isOpen = false;
    this.apiUrl = this.getAttribute('api-url') || '/api/chat';

    // Manage unique session ID per browser tab session
    if (!sessionStorage.getItem('chatbot_session_id')) {
      const uniqueId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      sessionStorage.setItem('chatbot_session_id', uniqueId);
    }
    this.sessionId = sessionStorage.getItem('chatbot_session_id');
  }

  static get observedAttributes() {
    return ['api-url'];
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === 'api-url' && newValue) {
      this.apiUrl = newValue;
    }
  }

  connectedCallback() {
    this.render();
    this.setupEventListeners();
  }

  openChat() {
    this.isOpen = true;
    this.updateChatState();
    this.shadowRoot.getElementById('chatInput').focus();
  }

  closeChat() {
    this.isOpen = false;
    this.updateChatState();
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
    this.updateChatState();
    if (this.isOpen) {
      this.shadowRoot.getElementById('chatInput').focus();
    }
  }

  updateChatState() {
    const chatBody = this.shadowRoot.getElementById('chatBody');
    const chatToggle = this.shadowRoot.getElementById('chatHeader');

    if (this.isOpen) {
      chatBody.classList.add('open');
      chatToggle.classList.add('open');
      this.classList.add('widget-open');
    } else {
      chatBody.classList.remove('open');
      chatToggle.classList.remove('open');
      this.classList.remove('widget-open');
    }
  }

  setupEventListeners() {
    const chatHeader = this.shadowRoot.getElementById('chatHeader');
    const chatForm = this.shadowRoot.getElementById('chatForm');

    chatHeader.addEventListener('click', () => this.toggleChat());

    chatForm.addEventListener('submit', (event) => {
      event.preventDefault();
      this.handleFormSubmit();
    });
  }

  async handleFormSubmit() {
    const inputEl = this.shadowRoot.getElementById('chatInput');
    const submitBtn = this.shadowRoot.querySelector('#chatForm button');
    const text = inputEl.value.trim();
    if (!text) return;

    // Append User Message
    this.appendMessage(text, 'user');
    inputEl.value = '';

    // Disable input and button to prevent concurrent submissions
    inputEl.disabled = true;
    if (submitBtn) submitBtn.disabled = true;

    // Show Typing Indicator
    this.showTypingIndicator(true);

    try {
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          message: text,
          session_id: this.sessionId
        })
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const hint = data.response || data.error || `HTTP ${response.status}`;
        throw new Error(hint);
      }

      this.showTypingIndicator(false);
      const reply = data.response || data.message || data.text || JSON.stringify(data);
      this.appendMessage(reply, 'bot');

    } catch (error) {
      console.error('Chatbot API error:', error);
      this.showTypingIndicator(false);
      const detail = error.message || String(error);
      this.appendMessage(
        `Could not reach the chat backend.\n\n` +
        `• Start the UI: npm run start:api (http://localhost:3000)\n` +
        `• Start the agent: npm run start:agent (port 5001)\n` +
        `• Open the site at http://localhost:3000 (not a local file)\n\n` +
        `Detail: ${detail}`,
        'bot error'
      );
    } finally {
      // Re-enable input and button
      inputEl.disabled = false;
      if (submitBtn) submitBtn.disabled = false;
      inputEl.focus();
    }
  }

  appendMessage(text, type) {
    const chatMessages = this.shadowRoot.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;

    // Format response (basic markdown: bullet points, bold, bold italic)
    messageEl.innerHTML = this.formatMessageText(text);

    chatMessages.appendChild(messageEl);
    this.scrollToBottom();
  }

  showTypingIndicator(show) {
    const indicator = this.shadowRoot.getElementById('typingIndicator');
    if (show) {
      indicator.classList.add('visible');
      this.scrollToBottom();
    } else {
      indicator.classList.remove('visible');
    }
  }

  scrollToBottom() {
    const chatMessages = this.shadowRoot.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  formatMessageText(text) {
    if (!text) return '';
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Format bold: **text** -> <strong>text</strong>
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Make URLs clickable
    html = html.replace(
      /(https?:\/\/[^\s<]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" style="color: #f5851f; text-decoration: underline;">$1</a>'
    );

    // Helper to render HTML table with smart cell wrapping
    const buildTableHtml = (headers, rows) => {
      let tableHtml = '<div class="table-container"><table class="chat-table">';
      tableHtml += '<thead><tr>';
      headers.forEach(h => {
        const headerLower = h.toLowerCase().trim();
        let cellClass = '';
        if (headerLower === 'description' || h.length > 30) {
          cellClass = ' class="wrap-text"';
        }
        tableHtml += `<th${cellClass}>${h}</th>`;
      });
      tableHtml += '</tr></thead><tbody>';

      rows.forEach(row => {
        tableHtml += '<tr>';
        for (let c = 0; c < headers.length; c++) {
          const cellValue = row[c] || '';
          const headerLower = (headers[c] || '').toLowerCase().trim();
          let cellClass = '';
          if (headerLower === 'description' || cellValue.length > 50 || (cellValue.includes(' ') && cellValue.length > 25)) {
            cellClass = ' class="wrap-text"';
          }
          tableHtml += `<td${cellClass}>${cellValue}</td>`;
        }
        tableHtml += '</tr>';
      });
      tableHtml += '</tbody></table></div>';
      return tableHtml;
    };

    // Parse Markdown tables
    const lines = html.split('\n');
    const processedLines = [];
    let inTable = false;
    let tableHeaders = [];
    let tableRows = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const isTableRow = line.startsWith('|') && line.endsWith('|') && line.length > 2;

      if (isTableRow) {
        const cells = line.split('|').map(c => c.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);

        if (!inTable) {
          const nextLine = (lines[i + 1] || '').trim();
          const isSeparator = nextLine.startsWith('|') && nextLine.endsWith('|') && nextLine.replace(/[|:\-\s]/g, '') === '';

          if (isSeparator) {
            inTable = true;
            tableHeaders = cells;
            tableRows = [];
            i++; // Skip separator line
            continue;
          }
        } else {
          tableRows.push(cells);
          continue;
        }
      }

      if (inTable && !isTableRow) {
        processedLines.push(buildTableHtml(tableHeaders, tableRows));
        inTable = false;
      }

      if (!inTable) {
        processedLines.push(lines[i]);
      }
    }

    if (inTable) {
      processedLines.push(buildTableHtml(tableHeaders, tableRows));
    }

    let inList = false;
    const finalLines = processedLines.map(line => {
      const trimmed = line.trim();

      if (trimmed.startsWith('<div') || trimmed.startsWith('<table')) {
        let suffix = '';
        if (inList) {
          inList = false;
          suffix = '</ul>';
        }
        return suffix + line;
      }

      if (trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')) {
        if (trimmed.startsWith('-') && trimmed.replace(/[\-\s]/g, '') === '') {
          let suffix = '';
          if (inList) {
            inList = false;
            suffix = '</ul>';
          }
          return suffix + '<hr class="chat-divider">';
        }
        const itemContent = trimmed.substring(1).trim();
        let prefix = '';
        if (!inList) {
          inList = true;
          prefix = '<ul class="chat-list">';
        }
        return `${prefix}<li>${itemContent}</li>`;
      } else {
        let suffix = '';
        if (inList) {
          inList = false;
          suffix = '</ul>';
        }
        return suffix + (trimmed ? `<p>${line}</p>` : '<br>');
      }
    });

    if (inList) {
      finalLines.push('</ul>');
    }

    return finalLines.join('');
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          position: fixed;
          right: 24px;
          bottom: 24px;
          width: min(100vw - 48px, 280px); /* Compact width when closed */
          z-index: 99999;
          font-family: 'IBM Plex Sans', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          font-size: 0.95rem;
          color: #102a43;
          transition: width 220ms cubic-bezier(0.4, 0, 0.2, 1);
        }

        :host(.widget-open) {
          width: min(100vw - 48px, 580px); /* Wide width when open to fit tables */
        }

        .chat-widget {
          display: flex;
          flex-direction: column;
          width: 100%;
        }

        .chat-toggle {
          width: 100%;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border: 1px solid rgba(16, 42, 67, 0.15); /* Soft dark border */
          padding: 16px 20px;
          border-radius: 24px;
          background: rgba(255, 255, 255, 0.25); /* More opaque glass background */
          color: #102a43; /* Dark blue text */
          cursor: pointer;
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          box-shadow: 0 24px 50px rgba(0, 0, 0, 0.12);
          transition: background-color 200ms ease, box-shadow 200ms ease, color 200ms ease;
        }

        .chat-toggle:hover {
          background: rgba(255, 255, 255, 0.45);
          box-shadow: 0 30px 60px rgba(16, 42, 67, 0.18);
        }

        .chat-left {
          display: inline-flex;
          gap: 10px;
          align-items: center;
        }

        .chat-dot {
          width: 8px;
          height: 8px;
          background-color: #22c55e;
          border-radius: 50%;
          display: inline-block;
          box-shadow: 0 0 8px #22c55e;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
          100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }

        .chat-label {
          font-weight: 600;
          letter-spacing: 0.02em;
        }

        .chat-status {
          font-size: 0.8rem;
          opacity: 0.8;
        }

        .chat-arrow {
          font-size: 0.95rem;
          opacity: 0.78;
          transition: transform 180ms ease;
        }

        .chat-toggle.open {
          background: rgba(255, 255, 255, 0.14);
          color: #ffffff; /* White text when open to contrast with dark body */
          border-bottom-left-radius: 0;
          border-bottom-right-radius: 0;
          border-color: rgba(255, 255, 255, 0.08);
        }

        .chat-toggle.open .chat-arrow {
          transform: rotate(180deg);
        }

        .chat-body {
          display: flex;
          flex-direction: column;
          background: rgba(15, 23, 42, 0.96);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-top: none;
          border-bottom-left-radius: 24px;
          border-bottom-right-radius: 24px;
          overflow: hidden;
          transform: scaleY(0);
          transform-origin: top;
          transition: transform 220ms cubic-bezier(0.4, 0, 0.2, 1), max-height 220ms ease;
          max-height: 0;
          box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3);
        }

        .chat-body.open {
          transform: scaleY(1);
          max-height: 600px;
        }

        .chat-messages {
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 14px;
          height: 420px;
          overflow-y: auto;
          scrollbar-width: thin;
          scrollbar-color: rgba(255,255,255,0.1) transparent;
        }

        /* Markdown Table Styles */
        .table-container {
          width: 100%;
          overflow-x: auto;
          margin: 10px 0;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .chat-table {
          width: 100%;
          min-width: max-content;
          border-collapse: collapse;
          font-size: 0.85rem;
          color: #f8fafc;
          text-align: left;
        }

        .chat-table th {
          background: rgba(255, 255, 255, 0.12);
          padding: 8px 12px;
          font-weight: 600;
          border-bottom: 1px solid rgba(255, 255, 255, 0.15);
          white-space: nowrap;
        }

        .chat-table td {
          padding: 8px 12px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          white-space: nowrap;
        }

        .chat-table th.wrap-text,
        .chat-table td.wrap-text {
          white-space: normal;
          min-width: 180px;
          max-width: 280px;
          overflow-wrap: break-word;
          word-break: break-word;
        }

        .chat-table tr:last-child td {
          border-bottom: none;
        }

        .chat-table tr:nth-child(even) {
          background: rgba(255, 255, 255, 0.03);
        }

        .chat-divider {
          border: 0;
          height: 1px;
          background: rgba(255, 255, 255, 0.1);
          margin: 12px 0;
        }

        .chat-messages::-webkit-scrollbar {
          width: 6px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
          background-color: rgba(255,255,255,0.1);
          border-radius: 99px;
        }

        .message {
          padding: 12px 16px;
          border-radius: 18px;
          line-height: 1.5;
          max-width: 85%;
          word-break: break-word;
          animation: messageFadeIn 250ms ease-out forwards;
        }

        @keyframes messageFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .message p {
          margin: 0 0 8px 0;
        }

        .message p:last-child {
          margin-bottom: 0;
        }

        .message.bot {
          background: rgba(255, 255, 255, 0.08);
          color: #f8fafc;
          align-self: flex-start;
          border-bottom-left-radius: 4px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .message.user {
          background: rgba(245, 133, 31, 0.15);
          color: #ffe7d1;
          align-self: flex-end;
          border-bottom-right-radius: 4px;
          border: 1px solid rgba(245, 133, 31, 0.25);
        }

        .message.error {
          background: rgba(239, 68, 68, 0.15);
          color: #fca5a5;
          border: 1px solid rgba(239, 68, 68, 0.25);
        }

        .chat-list {
          margin: 4px 0 8px 0;
          padding-left: 20px;
        }

        .chat-list li {
          margin-bottom: 4px;
        }

        .chat-list li:last-child {
          margin-bottom: 0;
        }

        /* Typing indicator styling */
        .typing-indicator {
          display: none;
          align-self: flex-start;
          background: rgba(255, 255, 255, 0.08);
          padding: 14px 18px;
          border-radius: 18px;
          border-bottom-left-radius: 4px;
          border: 1px solid rgba(255, 255, 255, 0.05);
          margin-bottom: 4px;
        }

        .typing-indicator.visible {
          display: flex;
          gap: 6px;
          align-items: center;
        }

        .typing-dot {
          width: 6px;
          height: 6px;
          background-color: #a1a1aa;
          border-radius: 50%;
          animation: bounce 1.4s infinite ease-in-out both;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }

        .chat-form {
          display: flex;
          gap: 10px;
          padding: 16px;
          background: rgba(255, 255, 255, 0.03);
          border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        .chat-form input {
          flex: 1;
          min-width: 0;
          padding: 12px 16px;
          border-radius: 16px;
          border: 1px solid rgba(255, 255, 255, 0.12);
          background: rgba(255, 255, 255, 0.06);
          color: #ffffff;
          outline: none;
          font-size: 0.95rem;
          transition: border-color 150ms ease, background-color 150ms ease;
        }

        .chat-form input:focus {
          border-color: #ffc080;
          background: rgba(255, 255, 255, 0.09);
        }

        .chat-form input::placeholder {
          color: rgba(255, 255, 255, 0.4);
        }

        .chat-form button {
          border: none;
          padding: 0 18px;
          border-radius: 16px;
          background: #ffc080;
          color: #0c1c2c;
          font-weight: 700;
          cursor: pointer;
          transition: transform 150ms ease, background-color 150ms ease;
        }

        .chat-form button:hover:not(:disabled) {
          background: #ffd8b3;
          transform: scale(1.02);
        }

        .chat-form button:active:not(:disabled) {
          transform: scale(0.98);
        }

        .chat-form button:disabled {
          background: rgba(255, 255, 255, 0.08);
          color: rgba(255, 255, 255, 0.3);
          cursor: not-allowed;
          transform: none;
          pointer-events: none;
        }

        .chat-form input:disabled {
          background: rgba(255, 255, 255, 0.02);
          color: rgba(255, 255, 255, 0.3);
          border-color: rgba(255, 255, 255, 0.05);
          cursor: not-allowed;
        }
      </style>

      <div class="chat-widget">
        <button class="chat-toggle" id="chatHeader">
          <span class="chat-left">
            <span class="chat-dot"></span>
            <span class="chat-label">Automation Agent</span>
            <span class="chat-status">(Online)</span>
          </span>
          <span class="chat-arrow">▼</span>
        </button>
        <div class="chat-body" id="chatBody">
          <div class="chat-messages" id="chatMessages">
            <div class="message bot">
              <p>Hi there! I am your Control-M automation agent. Type <strong>hello</strong> or <strong>help</strong> to begin.</p>
            </div>
          </div>
          
          <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
          </div>

          <form class="chat-form" id="chatForm">
            <input type="text" id="chatInput" placeholder="Type a message..." autocomplete="off" />
            <button type="submit">Send</button>
          </form>
        </div>
      </div>
    `;
  }
}

customElements.define('bmc-chatbot-widget', BmcChatbotWidget);

// Auto-initialize if a regular div with id "bmc-chatbot" is present on the page
document.addEventListener('DOMContentLoaded', () => {
  const targetDiv = document.getElementById('bmc-chatbot');
  if (targetDiv) {
    if (!targetDiv.querySelector('bmc-chatbot-widget')) {
      const widget = document.createElement('bmc-chatbot-widget');
      const apiUrl = targetDiv.getAttribute('data-api-url');
      if (apiUrl) {
        widget.setAttribute('api-url', apiUrl);
      }
      targetDiv.appendChild(widget);
    }
  }
});
