/* filepath: c:\Users\kushr\OneDrive\Desktop\Chatbot_Django\Chatbot\static\js\chat.js */
document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const uploadBtn = document.getElementById("upload-btn");
    const pdfUploadInput = document.getElementById("pdf-upload-input");

    let source = null; // For EventSource

    // Auto-scroll to the bottom of the chat
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
        updateSendButtonState();
    });

    function updateSendButtonState() {
        sendBtn.disabled = userInput.value.trim() === "";
    }
    updateSendButtonState();


    function addMessage(role, content) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", role);

        const iconClass = role === 'user' ? 'fa-user' : 'fa-robot';
        const iconBg = role === 'user' ? '#6a4ea7' : '#10a37f';

        messageDiv.innerHTML = `
            <div class="message-icon" style="background-color: ${iconBg};">
                <i class="fa-solid ${iconClass}"></i>
            </div>
            <div class="message-content">${content}</div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        addMessage("user", message);
        userInput.value = "";
        userInput.style.height = 'auto';
        updateSendButtonState();

        if (source) source.close();

        const currentUrl = window.location.pathname;
        source = new EventSource(`/stream${currentUrl}?message=${encodeURIComponent(message)}`);

        const botMessageDiv = addMessage("assistant", '<i class="fa-solid fa-spinner fa-spin"></i>');
        const botContentDiv = botMessageDiv.querySelector('.message-content');
        botContentDiv.dataset.streaming = "true";
        let fullResponse = "";

        source.onmessage = function(event) {
            if (event.data === "[DONE]") {
                source.close();
                botContentDiv.dataset.streaming = "false";
                return;
            }
            fullResponse += event.data;
            botContentDiv.innerHTML = fullResponse.replace(/\n/g, '<br>');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };

        source.onerror = function() {
            botContentDiv.innerHTML += "<br><br><span style='color: #ff5555;'>Error: Connection lost.</span>";
            source.close();
        };
    }

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // PDF Upload
    uploadBtn.addEventListener("click", () => pdfUploadInput.click());
    pdfUploadInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("pdf", file);

        const statusMessage = addMessage('assistant', `Uploading and processing ${file.name}... <i class="fa-solid fa-spinner fa-spin"></i>`);

        fetch("/upload_pdf/", {
            method: "POST",
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const contentDiv = statusMessage.querySelector('.message-content');
            if (data.status) {
                contentDiv.innerHTML = `<i class="fa-solid fa-check-circle"></i> ${data.status}`;
            } else {
                contentDiv.innerHTML = `<i class="fa-solid fa-exclamation-triangle"></i> Error: ${data.error}`;
            }
        })
        .catch(error => {
            const contentDiv = statusMessage.querySelector('.message-content');
            contentDiv.innerHTML = `<i class="fa-solid fa-exclamation-triangle"></i> Error: ${error.message}`;
        });
    });

    // Confirm before deleting a thread
    document.querySelectorAll('.delete-thread-form').forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!confirm('Are you sure you want to delete this chat?')) {
                e.preventDefault();
            }
        });
    });
});