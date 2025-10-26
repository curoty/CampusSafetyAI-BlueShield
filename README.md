# 校园AI多模态摔倒智能检测 — 青盾护卫

## 项目简介
本项目基于 YOLOv10 与 DeepSeek 大模型，构建校园场景下的智能安全识别系统。
主要功能：
- 📷 摄像头跌倒检测
- 🧠 DeepSeek 大模型语义分析与告警
- 🔔 实时多渠道预警与可视化前端

## 技术栈
- YOLOv10s + PyTorch
-  Flask + WebSocket
- DeepSeek-LLM-7B-Chat (4bit+LoRA 微调)
- Neo4j 知识图谱

## 运行方式
```bash
git clone https://github.com/zhuzihua-blue/CampusSafetyAI-BlueShield
cd CampusSafetyAI-BlueShield/main
python run_app.py
```

## 团队成员
| 姓名 | 负责模块 | GitHub账号   |
|------|-----------|------------|
| 朱梓华 | YOLO + DeepSeek + 数据安全 | curoty     |
| 李康 | 前后端框架开发 | 34liyaoling |
| 杜明阳 | 文档撰写、PPT、测试 | DUminy-star |

## 许可证
本项目仅用于教育与科研用途。
