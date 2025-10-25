// frame.js

// --- 新增：定时采样计算帧率 ---

// 存储上一次采样的时间和帧数
let lastSampleTime = Date.now();
let framesSinceLastSample = 0;

// 定时检查函数，比如每秒检查一次
function checkFps() {
    // 每次调用，帧数加一
    framesSinceLastSample++;

    const currentTime = Date.now();
    const elapsedTime = currentTime - lastSampleTime;

    // 如果距离上次采样已经超过1秒（1000毫秒）
    if (elapsedTime >= 1000) {
        // 计算帧率：帧数 / 秒数
        const fps = (framesSinceLastSample / (elapsedTime / 1000)).toFixed(1);

        // 更新页面显示
        const fpsElement = document.getElementById('fps-value');
        if (fpsElement) {
            fpsElement.textContent = `${fps} FPS`;
        }

        // 重置采样数据，为下一次计算做准备
        lastSampleTime = currentTime;
        framesSinceLastSample = 0;
    }

    // 使用 requestAnimationFrame 可以让检查频率和浏览器的渲染频率同步，更高效
    requestAnimationFrame(checkFps);
}

// 页面加载完成后，直接启动这个定时检查函数
document.addEventListener('DOMContentLoaded', function() {
    // ... 你原有的其他初始化代码（如 initAiDialog, updateStatus） ...
    updateStatus();

    // 启动帧率检查
    checkFps();
});

