document.addEventListener('DOMContentLoaded', function() {
    console.log('Main script loaded');
});

# static/js/chat.js
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
            const data = JSON.parse(event.data);
            this.addMessage(data.message, 'assistant');
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

document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
});