
document.addEventListener('DOMContentLoaded', function() {
    // 1. 获取DOM元素
    const aiDialog = document.getElementById('aiDialog');
    const aiCloseBtn = document.getElementById('aiCloseBtn');
    const aiMessages = document.getElementById('aiMessages');
    const aiInput = document.getElementById('aiInput');
    const aiSendBtn = document.getElementById('aiSendBtn');
    const socket = io.connect('http://localhost:5000');

    // 监听后端推送的摔倒警报事件
    socket.on('fall_alert', (data) => {
        const alertContent = `【紧急警报】${data.message}`;
        addMessage(alertContent, false); // 第二个参数false表示非用户消息（用AI样式）

        // 额外给告警消息添加alert类（如果需要特殊样式）
        const lastMessage = aiMessages.lastChild; // 获取刚添加的消息元素
        lastMessage.classList.add('alert');
    });

    // 添加消息到列表
    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'ai-message-user' : 'ai-message-assistant';
        messageDiv.textContent = content;
        messageDiv.style.wordWrap = 'break-word';
        messageDiv.style.whiteSpace = 'pre-wrap';
        aiMessages.appendChild(messageDiv);
        aiMessages.scrollTop = aiMessages.scrollHeight;
    }

    // 添加AI加载状态
    function addLoadingState() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'ai-loading';
        loadingDiv.innerHTML = '<div class="spinner"></div><span>AI正在思考...</span>';
        aiMessages.appendChild(loadingDiv);
        aiMessages.scrollTop = aiMessages.scrollHeight;
        return loadingDiv;
    }

    // 发送消息逻辑
    function sendMessage() {
        const userInput = aiInput.value.trim();
        if (!userInput) return;

        addMessage(userInput, true);
        aiInput.value = '';

        const loadingDiv = addLoadingState();

        fetch('/ask_ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userInput })
        })
        .then(response => response.ok ? response.json() : Promise.reject())
        .then(data => {
            aiMessages.removeChild(loadingDiv);
            addMessage(data.reply, false);
        })
        .catch(() => {
            aiMessages.removeChild(loadingDiv);
            addMessage('抱歉，AI服务暂时不可用，请稍后再试。', false);
        });
    }

    // 绑定事件
    aiCloseBtn.addEventListener('click', () => {
        // 在分栏布局中可隐藏对话框（可选功能）
        aiDialog.style.display = 'none';
        // 可添加显示按钮的逻辑
    });

    aiSendBtn.addEventListener('click', sendMessage);
    aiInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // 初始欢迎消息
    setTimeout(() => {
        addMessage("你好！我是YOLO实时检测系统的AI助手，你可以问我：支持哪些目标检测、怎么提高帧率、视频延迟怎么解决等问题～");
    }, 500);
});