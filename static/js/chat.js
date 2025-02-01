class ChatManager {
    constructor() {
        this.messageContainer = document.getElementById('messages');
        this.chatForm = document.getElementById('chat-form');
        this.messageInput = document.getElementById('message-input');

        this.connectWebSocket();
        this.initialize();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/`;

        console.log('Connecting to WebSocket:', wsUrl);
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.addSystemMessage('Connected to chat');
        };

        this.socket.onclose = () => {
            console.log('WebSocket disconnected');
            this.addSystemMessage('Disconnected from chat. Trying to reconnect...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.addSystemMessage('Error connecting to chat');
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.error) {
                    this.addSystemMessage(`Error: ${data.error}`);
                } else {
                    this.addMessage(data.message, 'assistant');
                }
            } catch (e) {
                console.error('Error parsing message:', e);
                this.addSystemMessage('Error processing message');
            }
        };
    }

    initialize() {
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const message = this.messageInput.value.trim();
            if (message) {
                this.sendMessage(message);
                this.messageInput.value = '';
            }
        });
    }

    sendMessage(message) {
        if (this.socket.readyState === WebSocket.OPEN) {
            this.addMessage(message, 'user');
            this.socket.send(JSON.stringify({
                'message': message
            }));
        } else {
            this.addSystemMessage('Not connected to chat. Please wait...');
        }
    }

    addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role} mb-4 ${
            role === 'user' ? 'text-right' : 'text-left'
        }`;

        const bubble = document.createElement('div');
        bubble.className = `inline-block p-3 rounded-lg ${
            role === 'user'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-800'
        }`;

        bubble.textContent = content;
        messageDiv.appendChild(bubble);
        this.messageContainer.appendChild(messageDiv);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    addSystemMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system mb-4 text-center';

        const bubble = document.createElement('div');
        bubble.className = 'inline-block p-2 rounded-lg bg-yellow-100 text-yellow-800 text-sm';

        bubble.textContent = message;
        messageDiv.appendChild(bubble);
        this.messageContainer.appendChild(messageDiv);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
});

document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const statusDiv = document.getElementById('upload-status');

    statusDiv.innerHTML = '<div class="text-blue-600">Uploading documents...</div>';

    fetch('/', {  // Changed from '/upload/' to '/'
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="text-green-600">${data.message}</div>
                ${data.errors ? `<div class="text-yellow-600 mt-2">Warnings: ${data.errors.join(', ')}</div>` : ''}
            `;
        } else {
            statusDiv.innerHTML = `
                <div class="text-red-600">Error: ${data.message}</div>
                ${data.errors ? `<div class="text-red-600 mt-2">${data.errors.join(', ')}</div>` : ''}
            `;
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="text-red-600">Error: ${error.message}</div>`;
    });
});