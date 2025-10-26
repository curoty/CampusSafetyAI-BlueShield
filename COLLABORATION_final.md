# 团队协作规范（青盾护卫队）

## 一、团队信息
- **项目名称**：校园人类风险行为识别系统（CampusSafetyAI-BlueShield）  
- **仓库地址**：[https://github.com/curoty/CampusSafetyAI-BlueShield]  
- **分支结构**：采用单一主分支 `main` 模式集中开发，所有成员基于模块化目录上传与维护。  

---

## 二、成员分工与职责
| 姓名 | 职责模块 | 主要文件 / 文件夹 |
|------|-----------|------------------|
| **朱梓华（队长）** | YOLO 模型训练、DeepSeek 大模型微调、知识图谱构建、数据脱敏与安全证明 | `yolo/`, `DS_tuning/`, `knowledge_graph_making/`, `app_logs/` |
| **李康** | 后端与前端框架开发、WebSocket告警通信、系统集成测试 | `app.py`, `alert_send.py`, `static/`, `templates/`, `config.py` |
| **杜明阳** | 文档撰写、测试报告与PPT制作、项目计划书与汇报文档 | `/Docs/`, `README.md`, `项目说明书.docx`, `PPT/` |

---

## 三、开发与提交规范

### 1. 提交频率  
- 每位成员需在完成阶段性功能后进行一次 commit；  
- 建议每人每周至少一次有效提交；  
- 所有提交均需附带中英文注释与简洁的提交说明。  

### 2. 提交信息格式  
```
git commit -m "feat: add YOLOv10 fall detection training pipeline"
git commit -m "fix: resolve websocket reconnection issue in alert sender"
git commit -m "docs: add project plan and test report draft"
git commit -m "yolo模型训练脚本+Deepseek大模型微调脚本+知识图谱制作脚本"
git commit -m "Fall Detection system_前后端"


```

### 3. 文件命名与目录规范
- 使用全英文命名；  
- 模块以文件夹区分（如 yolo / DS_tuning / static / templates）；  
- 所有配置文件统一放置在根目录或对应模块下。

---

## 四、协作与代码审查流程
1. 团队使用 **单分支协作** 模式；
2. 每人完成任务后直接推送至 `main` 分支；
3. 队长（朱梓华）负责定期检查提交记录；
4. 李康负责测试后端接口稳定性；
5. 杜明阳同步更新文档与测试报告；
6. 每次更新由队长在群内确认版本号。

---

## 五、文档协同与成果管理
- **文档平台**：OneDrive + GitHub 同步；
- **核心文档**：  
  - 《项目计划书》  
  - 《模型说明书》  
  - 《数据脱敏处理证明》  
  - 《DeepSeek 微调语料合规性说明摘要》  
  - 《脱敏抽检统计表》  
- **定期汇总**：杜明阳每周整合一次进展与成果归档。

---

## 六、代码贡献评估
通过 GitHub Insights → Contributors 可查看成员贡献比例。   
- 朱梓华（模型与核心算法）约 35%  
- 李康（前后端框架与集成）约 35%  
- 杜明阳（文档与展示）约 30%

---

## 七、沟通与管理
- 团队沟通：微信群   
- 代码托管：GitHub 公共仓库  
- 文档备份：OneDrive  
- 项目负责人：**朱梓华（队长）**  
- 更新日期：2025 年 10 月 26 日  
