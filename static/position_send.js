document.addEventListener('DOMContentLoaded', function() {
    // 初始化历史记录显示和事件绑定
    history_appear();       // 绑定输入框聚焦/失焦事件
    renderHistoryList();    // 显示已保存的历史记录
    initCurrentLocation();  // 初始化当前位置显示
});


async function sendPosition(){

    const initial = document.getElementById("locationInput");
    const position = initial.value.trim()

        if (position) {
            const data = {
                current_position: position,
            }

            try {
                const response = await fetch('/position', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                if (response.ok){
                    updateLocationDisplay(position)
                    position_save(position)
                    console.log('位置更新成功：');

                }
            }

            catch (error){
                  console.error('请求出错：', error);
            }
        }


        }
        function updateLocationDisplay(location) {

        const displayElement = document.getElementById('currentLocationDisplay');

        displayElement.textContent = location || "未设置";
        displayElement.style.color = location ? '#16a34a' : '#64748b';

        }

        function initCurrentLocation() {
            const savedLocation = localStorage.getItem('lastSavedLocation');
            if (savedLocation) {
                updateLocationDisplay(savedLocation);
            }
        }


        function position_save(data) {
            let history = JSON.parse(localStorage.getItem('locationHistory') || '[]');

            // 去重：已存在的位置不重复保存
            if (history.includes(data)) {
                return;
            }

            // 新位置添加到最前面（最新的在顶部）
            history.unshift(data);

            // 限制最多保存10条记录
            if (history.length > 10) {
                history.pop(); // 删除最早的记录
            }

            // 保存到本地存储
            localStorage.setItem('locationHistory', JSON.stringify(history));
        }

        function renderHistoryList() {
            const historyList = document.getElementById('historyList');
            const history = JSON.parse(localStorage.getItem('locationHistory') || '[]');

            // 清空列表（避免重复渲染）
            historyList.innerHTML = '';

            // 无历史记录时显示提示
            if (history.length === 0) {
                const emptyItem = document.createElement('li');
                emptyItem.textContent = '暂无历史位置记录';
                emptyItem.style.color = '#64748b'; // 灰色提示文字
                historyList.appendChild(emptyItem);
                return;
            }

            // 遍历历史记录，生成可点击的列表项
            history.forEach(location => {
                const listItem = document.createElement('li');
                listItem.textContent = location;
                // 点击历史项自动填充到输入框
                listItem.addEventListener('click', () => {
                    document.getElementById('locationInput').value = location;
                });
                listItem.addEventListener('click', () => {
                    const input = document.getElementById('locationInput');
                    input.value = location;
                    input.focus(); // 填充后让输入框重新获得焦点，方便用户直接修改
                });
                historyList.appendChild(listItem);
            });
        }

        function history_appear() {
            const locationInput = document.getElementById('locationInput');
            const historyList = document.getElementById('historyList');

            // 输入框聚焦时显示历史列表
            locationInput.addEventListener('focus', () => {
                renderHistoryList(); // 确保显示最新记录
                historyList.style.display = 'block';
            });

            // 输入框失焦时延迟隐藏（避免点击历史项时立即消失）
            locationInput.addEventListener('blur', () => {
                setTimeout(() => {
                    historyList.style.display = 'none';
                }, 100);
            });
        }

        document.getElementById('clearBtn').addEventListener('click', () => {
            const historyList = document.getElementById('historyList');
            // 确认清空操作
            if (historyList.children.length <= 1 && historyList.children[0]?.textContent.includes('暂无')) {
                alert('没有可清空的历史记录');
                return;
            }

            if (confirm('确定要清空所有位置历史记录吗？')) {
                localStorage.removeItem('locationHistory');
                renderHistoryList(); // 刷新列表显示
                alert('历史记录已清空');
            }
        });











