import json
import random
from datetime import datetime, timedelta
import itertools


class ExtendedTrainingDataGenerator:
    def __init__(self):
        self.synonyms = {
            "开发团队": ["开发小组", "项目团队", "技术团队", "研发队伍", "开发人员", "项目组", "技术小组", "开发阵容"],
            "大二学生": ["大学二年级学生", "二年级本科生", "大二在读生", "大二同学", "二年级同学", "大二学子"],
            "人工智能专业": ["AI专业", "智能科学与技术专业", "人工智能方向", "AI技术专业", "人工智能学科"],
            "扎实的": ["坚实的", "牢固的", "深厚的", "过硬的", "扎实的", "熟练的", "专业的"],
            "Python": ["Python编程", "Python语言", "Python开发", "Python技术", "Python程序"],
            "机器学习": ["ML技术", "机器学习算法", "智能算法", "AI算法", "机器学习方法"],
            "大模型部署": ["大模型应用部署", "大模型上线", "大模型工程化", "模型部署实施"],
            "微调": ["调优", "优化适配", "参数调整", "模型微调", "精细调整"],
            "摔倒检测": ["跌倒检测", "摔倒识别", "跌倒识别", "摔倒监测", "跌倒监测"],
            "YOLO": ["YOLO模型", "YOLO检测器", "YOLO识别模型", "YOLO算法", "目标检测模型"],
            "DeepSeek": ["DeepSeek大模型", "DeepSeek-7B", "DS7b模型", "DS大模型", "语言模型"],
            "状态机": ["状态判断机", "状态逻辑器", "状态控制器", "状态管理机", "计时状态机"],
            "RAG": ["RAG记忆", "RAG存储层", "检索增强生成", "RAG系统", "记忆检索层"],
            "告警": ["警报", "报警", "预警通知", "告警信息", "危险通知"],
            "置信度": ["可信度", "置信分数", "可靠度", "置信水平", "可信分数"],
            "助手": ["助理", "辅助者", "帮手", "服务者", "支持者"],
            "监控": ["监测", "监视", "观察", "检测", "巡查"],
            "帧": ["画面帧", "图像帧", "视频帧", "检测帧"],
            "比例": ["占比", "比率", "百分比", "份额", "比重"],
            "系统": ["平台", "体系", "方案", "解决方案", "系统平台"],
            "功能": ["能力", "作用", "性能", "效用", "功能特性"],
            "技术": ["科技", "技术手段", "技术方案", "技术方法", "技术实现"],
            "项目": ["系统", "平台", "方案", "工程", "项目系统"],
            "检测": ["识别", "监测", "感知", "发现", "侦测"],
            "告警": ["报警", "警报", "预警", "提醒", "通知"],
            "实时": ["即时", "立即", "马上", "即刻", "及时"],
            "历史": ["过往", "以前", "先前", "旧时", "过去"],
            "查询": ["查找", "搜索", "检索", "查看", "调取"],
            "存储": ["保存", "存储", "保留", "存档", "记录"],
            "分析": ["解析", "处理", "研判", "分析处理", "数据分析"],
            "实现": ["完成", "达成", "实施", "执行", "落实"],
            "部署": ["安装", "配置", "布置", "搭建", "设置"],
            "优化": ["改进", "提升", "完善", "增强", "改良"],
            "架构": ["结构", "框架", "体系结构", "设计方案", "技术架构"],
            "算法": ["计算方法", "处理逻辑", "运算规则", "核心算法", "处理算法"]
        }

        # 统一的指令模板
        self.alert_instruction = "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户"
        self.qa_instruction = "你是本项目的AI助手，请根据你的知识回答用户的问题"

        # 扩展的AI助手身份相关问答
        self.identity_questions = {
            "assistant_identity": {
                "core_question": "AI助手身份",
                "question_variations": [
                    "你好",
                    "你是谁？",
                    "给我介绍一下你自己",
                    "你是什么AI？",
                    "你的身份是什么？",
                    "请做个自我介绍",
                    "你叫什么名字？",
                    "你能做什么？",
                    "你的职责是什么？",
                    "简单介绍一下你",
                    "嗨，你好",
                    "你是谁开发的？",
                    "你的功能有哪些？",
                    "你是什么类型的助手？",
                    "你的定位是什么？",
                    "介绍一下你的背景",
                    "你的作用是什么？",
                    "你是做什么的？",
                    "请描述一下你自己",
                    "你的服务内容有哪些？",
                    "你好，能介绍一下自己吗？",
                    "请问您是谁？",
                    "可以自我介绍一下吗？",
                    "你的角色是什么？",
                    "你在这个项目中负责什么？",
                    "请告诉我你的基本信息",
                    "你是什么类型的AI？",
                    "你的主要功能是什么？",
                    "你能为我提供什么服务？",
                    "简单说明一下你的身份",
                    "请做个详细的自我介绍",
                    "介绍一下你的能力",
                    "你有哪些功能？",
                    "你的技术基础是什么？",
                    "你基于什么技术构建？",
                    "你的开发背景是什么？",
                    "你的技术特点是什么？",
                    "你的服务范围有哪些？",
                    "你能处理哪些问题？",
                    "你的专业领域是什么？"
                ],
                "answer_variations": [
                    "你好！我是摔倒检测告警助手，基于DeepSeek-7B大模型开发。我专门负责监控摔倒高危事件、生成告警信息，并回答关于本项目的各种问题。很高兴为您服务！",
                    "我是本项目的智能告警助手，专注于实时安全监护。我能够处理摔倒检测告警、查询历史事件记录，并解答关于项目团队、技术实现、功能模块等方面的疑问。",
                    "您好！我是摔倒检测系统的AI助手，由三位河南高校AI专业学生开发。我具备实时告警生成、历史数据查询和项目咨询三大核心能力，随时为您提供安全监护服务。",
                    "我是基于DeepSeek-7B大模型定制的摔倒检测告警助手。我的主要职责是：监控人员安全状态、生成高危事件告警、管理历史记录数据，并作为项目的信息咨询接口。",
                    "你好！我是这个摔倒检测项目的专属AI助手。我融合了计算机视觉检测与大语言模型智能，既能实时处理安全告警，又能智能回答各类项目相关问题，为您提供全方位的服务。",
                    "我是智能安全监护助手，专门为摔倒检测项目而生。我负责从视频分析到告警推送的整个流程，同时还能作为您的项目顾问，解答技术细节和使用方法等问题。",
                    "您好！我是项目的全栈AI助手，既处理后端的安全检测逻辑，又提供前端的智能交互服务。无论是实时告警还是历史查询，我都能快速准确地响应您的需求。",
                    "我是基于DeepSeek技术打造的摔倒检测专用助手。我的特色在于结合了YOLO视觉检测的准确性和大语言模型的交互智能，为您提供专业可靠的安全监护服务。",
                    "你好！我是这个系统的AI核心，负责协调视频分析、状态判断和告警生成的全流程。同时我也是您的智能顾问，可以详细解答项目的技术实现和功能特点。",
                    "我是项目的智能守护者，专注于人员安全监测。我具备实时风险识别、智能告警生成、历史数据管理和项目知识咨询四大能力，全力保障监控区域的安全。",
                    "您好！我是摔倒检测系统的AI助手，专门负责安全监控和告警处理。我可以帮您查询项目信息、处理安全事件，并提供技术咨询服务。",
                    "我是本项目的智能安全助手，基于先进的AI技术开发。我能够实时监测人员安全状态，及时生成告警信息，并解答您关于系统使用的各种问题。",
                    "你好！我是专为摔倒检测设计的AI助手。我整合了计算机视觉和自然语言处理技术，既能进行智能监控，又能提供友好的对话交互体验。",
                    "我是您身边的安全监护专家，基于DeepSeek大模型构建。我负责实时分析监控画面，识别潜在风险，并在必要时发出告警通知。",
                    "您好！我是摔倒检测平台的AI核心。我具备多重能力：实时视频分析、风险评估判断、智能告警生成和全方位信息咨询服务。",
                    "我是本项目的AI智能助手，专门为摔倒检测场景优化。我能够理解复杂的监控数据，生成准确的告警信息，并提供专业的技术咨询服务。",
                    "你好！我是基于DeepSeek-7B大模型开发的智能助手。我专门针对摔倒检测场景进行了优化训练，能够准确理解监控数据并生成专业的告警信息。",
                    "我是摔倒检测系统的智能核心，负责处理从视频分析到告警生成的全流程。同时我也能作为您的技术顾问，解答关于系统架构和实现细节的各种问题。",
                    "您好！我是本项目的全功能AI助手。我既能够处理实时的安全监控任务，也能够回答您关于技术实现、系统功能和使用方法的各类问题。",
                    "我是基于先进AI技术构建的摔倒检测助手。我整合了计算机视觉、自然语言处理和大模型技术，为您提供专业可靠的安全监护和咨询服务。"
                ]
            }
        }

        # 扩展的核心问题库
        self.core_questions = {
            "team_intro": {
                "core_question": "团队介绍",
                "question_variations": [
                    "项目开发团队是什么情况？",
                    "给我介绍一下本项目的开发者团队。",
                    "谁开发了这个摔倒检测项目？团队背景如何？",
                    "开发团队有几人？专业和技术基础怎样？",
                    "这个项目的开发者是谁？能介绍一下吗？",
                    "团队构成是怎样的？成员有什么技术背景？",
                    "开发人员都是什么背景？技术水平如何？",
                    "项目团队有多少人？他们的专业方向是什么？",
                    "能详细说说开发团队的情况吗？",
                    "团队成员都是什么学历和专业？",
                    "开发这个系统的是哪些人？他们有什么能力？",
                    "请介绍一下项目开发团队",
                    "项目的开发人员情况如何？",
                    "团队的技术实力怎么样？",
                    "开发者团队有多少成员？",
                    "开发团队的专业背景是什么？",
                    "项目是由谁开发的？",
                    "团队成员的技能水平如何？",
                    "开发团队的教育背景怎样？",
                    "请详细说明开发团队情况",
                    "团队的技术专长有哪些？",
                    "开发小组的成员构成？",
                    "项目研发团队背景介绍",
                    "技术团队的人员组成",
                    "开发者的技术能力如何？",
                    "项目团队的技术栈掌握情况",
                    "开发人员的项目经验",
                    "团队的技术优势是什么？",
                    "开发团队的学习背景",
                    "项目组成员的技术专长"
                ],
                "answer_variations": [
                    "本项目由三个河南高校人工智能专业的大二学生开发，他们具备扎实的Python、机器学习及大模型部署微调基础，能独立完成项目全流程开发。",
                    "开发这个项目的是三位河南高校的大二学生，专业均为人工智能，在Python编程、机器学习技术，以及大模型的部署与微调方面有扎实功底。",
                    "项目团队由三名河南高校人工智能专业大二学生组成，成员熟练掌握Python、机器学习，且在大模型部署和LoRA微调等技术上有丰富实操经验。",
                    "三位来自河南高校AI专业的大二同学共同开发了本项目，他们在Python开发、机器学习算法和大模型优化部署方面具备扎实的技术能力。",
                    "开发团队是三位人工智能专业的二年级本科生，来自河南高校，精通Python编程、机器学习理论，并拥有大模型微调与部署的实践经验。",
                    "本项目开发团队由三名河南高校AI专业大二学生构成，他们在Python编程、机器学习算法、大模型技术等方面有深厚的技术积累。",
                    "团队由三位河南高校人工智能方向的大二同学组成，具备扎实的编程基础和AI技术理解，能够独立完成复杂系统的设计与实现。",
                    "开发小组包括三名河南高校AI专业二年级学生，他们在Python开发、机器学习实践和大模型应用方面有丰富的项目经验。",
                    "项目研发团队是三位来自河南高校的大二AI专业学生，熟练掌握Python技术栈和机器学习框架，具备全栈开发能力。",
                    "本项目技术团队由三名河南高校人工智能专业大二学生牵头，他们在Python编程、AI算法和大模型优化方面有扎实的专业基础。",
                    "开发团队由三位河南高校AI专业大二学生组成，他们在Python编程、机器学习和大模型技术方面有扎实的理论基础和实践经验。",
                    "项目开发小组包括三名河南高校人工智能专业大二同学，他们精通Python开发、掌握机器学习算法，并具备大模型部署优化的实战能力。",
                    "技术团队由三位河南高校AI专业二年级学生构成，他们在Python编程、深度学习和大模型微调方面有丰富的项目开发经验。",
                    "开发团队包含三位河南高校人工智能专业大二学生，他们熟练掌握Python技术栈、机器学习框架，能够独立完成复杂AI系统的开发部署。",
                    "项目研发小组由三名河南高校AI专业大二同学组成，他们在Python编程、机器学习算法和大模型优化方面具备扎实的技术功底。"
                ]
            },

            "core_function": {
                "core_question": "核心功能",
                "question_variations": [
                    "项目主要能实现什么功能？",
                    "这个摔倒检测项目的核心作用是什么？",
                    "项目有哪些关键功能模块？",
                    "系统的主要功能有哪些？",
                    "这个检测系统能做什么？",
                    "项目的核心价值体现在哪里？",
                    "主要实现了哪些检测功能？",
                    "系统具备什么样的功能特点？",
                    "用户通过这个项目能获得什么服务？",
                    "功能模块都包括哪些内容？",
                    "给我介绍一下本项目的功能。",
                    "系统的主要作用是什么？",
                    "这个平台有什么功能？",
                    "项目能提供哪些服务？",
                    "核心功能特点有哪些？",
                    "系统的主要能力是什么？",
                    "这个解决方案有什么功能？",
                    "平台的功能特性有哪些？",
                    "请详细说明系统功能",
                    "项目的主要功能模块是什么？",
                    "系统能实现哪些具体功能？",
                    "功能体系包含哪些部分？",
                    "系统的核心能力有哪些？",
                    "项目功能架构是怎样的？",
                    "主要功能组件有哪些？",
                    "系统提供哪些核心服务？",
                    "功能实现的具体内容？",
                    "项目功能特色是什么？",
                    "系统功能组成结构？",
                    "核心功能实现方式？"
                ],
                "answer_variations": [
                    "项目核心功能为智能摔倒检测与高危告警：通过YOLO模型实时检测视频流中人体状态，结合状态机逻辑判断摔倒风险，当10秒内Fall Detected帧比例≥80%时，触发DeepSeek-7B大模型生成标准化告警信息并推送至用户；同时支持Web端实时监控查看、历史告警查询、大模型交互及1-2天监控视频存储。",
                    "系统核心功能包括实时摔倒检测、智能风险评估、自动告警推送和交互查询：利用YOLO识别人员状态，状态机基于帧比例验证风险持续性，大模型生成自然语言告警，Web界面提供全方位监控管理。",
                    "主要功能模块：1)实时视频分析检测 2)状态风险评估(基于帧比例) 3)智能告警生成 4)Web可视化界面 5)历史数据管理 6)智能对话交互，形成完整的摔倒监测解决方案。",
                    "功能体系：前端实时显示检测画面，后端进行状态识别与风险评估(帧比例分析)，大模型处理高危事件生成告警，RAG层存储历史数据，支持用户多维度查询与交互。",
                    "核心功能涵盖：视频流处理、人体状态检测、风险逻辑判断(帧比例≥80%)、告警信息生成、前端展示交互、历史数据存储六大方面，提供端到端的摔倒安全监护服务。",
                    "项目功能架构：基于YOLO的实时状态检测、基于帧比例的状态机风险评估、DeepSeek大模型的智能告警生成、Web前端的可视化展示、RAG记忆层的历史数据管理五大核心功能。",
                    "系统能力：实时视频流分析、人体姿态识别、摔倒风险判断(帧比例阈值)、智能告警推送、历史事件查询、AI对话交互六大功能模块，构建完整安全监护体系。",
                    "平台功能：1)实时监控与检测 2)风险评估与分析 3)智能告警生成 4)前端可视化 5)历史数据追溯 6)智能咨询服务，为用户提供全方位安全保障。",
                    "核心功能组成：视频采集处理、目标检测识别、状态风险评估、告警信息生成、前端界面展示、历史数据存储、智能对话交互七大功能组件。",
                    "系统功能特色：基于计算机视觉的实时检测、基于帧比例算法的风险评估、基于大模型的智能告警、基于Web的可视化界面、基于RAG的数据管理五大功能特性。",
                    "功能实现：通过YOLO模型实现实时状态检测，状态机进行帧比例风险评估，DeepSeek大模型生成告警信息，Web前端提供可视化界面，RAG系统管理历史数据。",
                    "核心功能模块：视频处理模块负责实时分析，检测模块识别人员状态，风险评估模块验证事件持续性，告警生成模块创建通知，前端模块展示结果，数据模块存储历史记录。",
                    "系统功能架构：感知层负责视频采集和状态识别，逻辑层进行风险评估和事件确认，生成层创建告警信息，展示层提供用户界面，存储层管理历史数据。",
                    "功能组成：实时监控功能提供视频流分析，风险评估功能验证事件真实性，告警生成功能创建通知信息，数据管理功能存储历史记录，交互功能支持用户查询。",
                    "核心功能实现：基于YOLO的视觉感知，基于状态机的逻辑判断，基于DeepSeek的信息生成，基于Web的前端展示，基于RAG的数据存储五大技术组件。"
                ]
            },

            # 其他问题类别类似扩展，这里省略详细代码以保持简洁
            # 实际使用时需要为所有15个问题类别都进行类似扩展
        }

        # 扩展告警地点池
        self.locations = [
            "一楼大厅", "二楼走廊东侧", "三楼卧室A", "康复中心训练区", "养老院201房间",
            "活动室中央", "卫生间门口", "楼梯口转角", "餐厅过道", "花园小径",
            "电梯厅前", "护士站旁边", "康复器材区", "休息区沙发旁", "阳台区域",
            "走廊尽头", "病房卫生间", "活动区角落", "楼梯平台", "门厅入口",
            "治疗室门口", "阅览室角落", "健身房中央", "厨房过道", "后院走廊",
            "日间照料中心", "理疗室内部", "娱乐活动区", "户外休息区", "紧急通道",
            "康复训练室", "老人活动中心", "医疗观察室", "护理站前", "日间休息区",
            "康复游泳池", "物理治疗室", "职业治疗区", "言语治疗室", "心理辅导室",
            "营养餐厅", "多功能活动厅", "康复花园", "阳光房", "休闲阅览区",
            "医疗设备间", "药品储存室", "急救准备室", "隔离观察区", "消毒处理间",
            "北区走廊", "南区活动室", "东侧楼梯间", "西侧康复区", "中央大厅",
            "康复治疗室", "老人居住区", "医护人员站", "紧急呼叫点", "安全监控区"
        ]

        # 扩展告警模板
        self.alert_templates = [
            "【高危告警】{time}在{location}检测到人员摔倒，10秒内Fall Detected帧比例达{percentage}%，请立即前往处理！",
            "🚨紧急通知：{time}于{location}发现摔倒事件，10秒内摔倒检测帧占比{percentage}%，建议立即派人查看！",
            "⚠安全警报：{time}{location}监测到摔倒情况，检测周期内摔倒帧比例{percentage}%，请及时安排处置！",
            "‼️紧急情况：{time}在{location}识别到人员跌倒，10秒检测期内摔倒状态占比{percentage}%，需要立即关注！",
            "📢系统告警：{time}{location}检测到摔倒高危事件，帧比例分析结果{percentage}%，请速处理！",
            "🔴高危事件：{time}{location}发生摔倒，检测帧比例{percentage}%，请尽快响应！",
            "🚑急救提醒：{time}在{location}检测到跌倒，帧分析占比{percentage}%，需要紧急援助！",
            "👥人员安全：{time}{location}发现摔倒人员，检测比例{percentage}%，请立即查看！",
            "📋事件报告：{time}于{location}监测到摔倒，帧占比{percentage}%，建议处理！",
            "💡风险提示：{time}在{location}识别摔倒风险，检测比例{percentage}%，请关注！",
            "🔔安全告警：{time}{location}检测到人员摔倒，帧比例{percentage}%，请及时处置！",
            "🚩危险事件：{time}在{location}发现跌倒情况，检测占比{percentage}%，需要关注！",
            "📞紧急呼叫：{time}{location}监测到摔倒事件，帧比例{percentage}%，请速往查看！",
            "👴老人安全：{time}在{location}检测到长者摔倒，帧分析{percentage}%，急需援助！",
            "🏥医疗关注：{time}{location}发生摔倒，检测比例{percentage}%，建议医疗检查！",
            "🚶人员安全告警：{time}{location}识别到跌倒事件，帧比例{percentage}%，请立即查看！",
            "📹监控检测告警：{time}在{location}监测到人员摔倒，检测占比{percentage}%，需要处理！",
            "👵长者安全提醒：{time}{location}发现老人跌倒，帧比例{percentage}%，请速往援助！",
            "🏃运动安全告警：{time}于{location}检测到运动摔倒，帧分析{percentage}%，建议检查！",
            "🆘紧急求助：{time}在{location}识别到人员倒地，检测比例{percentage}%，需要紧急响应！",
            "🔴🚨高危摔倒事件：{time}在{location}检测到人员跌倒，帧比例分析{percentage}%，请立即处理！",
            "⚠️📢安全系统告警：{time}{location}发生摔倒事件，检测帧占比{percentage}%，需要紧急关注！",
            "👵🚑长者安全告警：{time}在{location}监测到老人摔倒，帧分析比例{percentage}%，急需医疗援助！",
            "🏥🔴医疗紧急事件：{time}{location}识别到人员跌倒，检测比例{percentage}%，请速往处理！",
            "📋📊事件分析报告：{time}于{location}检测到摔倒，帧占比{percentage}%，建议立即查看！"
        ]

    def apply_synonym_replacement(self, text, replacement_rate=0.5):
        """应用同义词替换"""
        words = text.split()
        replaced_text = []

        for word in words:
            clean_word = word.strip('.,!?;:""''()[]{}')
            if clean_word in self.synonyms and random.random() < replacement_rate:
                replacement = random.choice(self.synonyms[clean_word])
                # 保持标点符号
                if word != clean_word:
                    replacement = replacement + word[len(clean_word):]
                replaced_text.append(replacement)
            else:
                replaced_text.append(word)

        return ' '.join(replaced_text)

    def generate_qa_pairs(self, num_samples=3500):
        """生成问答对样本"""
        qa_pairs = []

        # 合并身份问题和核心问题
        all_questions = {**self.identity_questions, **self.core_questions}

        # 计算每个问题类别需要生成的样本数
        questions_count = len(all_questions)
        samples_per_question = num_samples // questions_count
        remaining_samples = num_samples % questions_count

        for question_key, question_data in all_questions.items():
            question_variations = question_data["question_variations"]
            answer_variations = question_data["answer_variations"]

            # 为当前问题类别生成样本
            current_samples = samples_per_question
            if remaining_samples > 0:
                current_samples += 1
                remaining_samples -= 1

            # 生成当前问题类别的所有可能组合
            combinations = list(itertools.product(question_variations, answer_variations))

            # 如果组合数量不足，可以重复使用一些组合
            if len(combinations) < current_samples:
                # 重复组合直到达到所需数量
                repeated_combinations = []
                while len(repeated_combinations) < current_samples:
                    repeated_combinations.extend(combinations)
                combinations = repeated_combinations[:current_samples]
            else:
                # 随机选择所需数量的组合
                combinations = random.sample(combinations, current_samples)

            for question, answer in combinations:
                # 对答案进行同义词替换，增加多样性
                enhanced_answer = self.apply_synonym_replacement(answer)

                qa_pair = {
                    "instruction": self.qa_instruction,
                    "input": question,
                    "output": enhanced_answer
                }
                qa_pairs.append(qa_pair)

        return qa_pairs

    def generate_alert_samples(self, num_samples=1500):
        """生成告警训练样本"""
        alert_samples = []

        for i in range(num_samples):
            # 生成随机时间（最近30天内）
            base_time = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )

            # 生成帧比例数据 (80%-100%之间，符合触发条件)
            percentage = random.randint(80, 100)

            # 生成总检测帧数和Fall Detected帧数
            total_frames = random.randint(30, 60)  # 10秒内检测帧数
            fall_frames = int(total_frames * percentage / 100)

            event_time = base_time.strftime("%H:%M:%S")
            location = random.choice(self.locations)

            # 随机选择告警模板
            template = random.choice(self.alert_templates)

            # 多种输入格式变体
            input_variants = [
                f"时间：{event_time}，地点：{location}，10秒内总检测{total_frames}帧，其中检测到Fall Detected的帧数比例大于80%",
                f"高危事件：{event_time}在{location}，检测周期内Fall Detected帧比例超过阈值",
                f"摔倒检测告警：时间{event_time}，位置{location}，帧比例分析结果符合高危条件",
                f"安全事件：{location}在{event_time}检测到持续摔倒状态，需要生成告警",
                f"监控告警触发：{event_time}于{location}识别到摔倒高危事件，帧比例验证通过"
            ]

            input_text = random.choice(input_variants)
            output_text = template.format(
                time=event_time,
                location=location,
                percentage=percentage
            )

            alert_samples.append({
                "instruction": self.alert_instruction,
                "input": input_text,
                "output": output_text
            })

        return alert_samples

    def save_dataset(self, dataset, filename, split_ratio=0.9):
        """保存数据集并分割训练集/验证集"""
        # 打乱数据
        random.shuffle(dataset)

        # 分割数据集
        split_index = int(len(dataset) * split_ratio)
        train_data = dataset[:split_index]
        val_data = dataset[split_index:]

        # 保存完整数据集
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        # 保存训练集和验证集
        base_name = filename.replace('.json', '')
        with open(f"{base_name}_train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(f"{base_name}_val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        return len(dataset), len(train_data), len(val_data)


def main():
    generator = ExtendedTrainingDataGenerator()

    print("开始生成扩展训练数据...")
    print("目标：5000条微调样本（3500条问答对 + 1500条告警样本）")

    # 生成问答对数据
    print("生成3500条问答对数据...")
    qa_pairs = generator.generate_qa_pairs(3500)

    # 生成告警数据
    print("生成1500条告警训练数据...")
    alert_samples = generator.generate_alert_samples(1500)

    # 合并所有数据
    all_data = qa_pairs + alert_samples
    random.shuffle(all_data)

    # 保存数据集
    total_size, train_size, val_size = generator.save_dataset(
        all_data, "extended_5000_training_data.json"
    )

    # 输出统计信息
    print(f"\n=== 数据生成完成 ===")
    print(f"总数据量: {total_size} 条")
    print(f"训练集: {train_size} 条")
    print(f"验证集: {val_size} 条")

    # 指令类型统计
    print(f"\n=== 指令类型统计 ===")
    alert_count = sum(1 for item in all_data if item["instruction"] == generator.alert_instruction)
    qa_count = sum(1 for item in all_data if item["instruction"] == generator.qa_instruction)
    print(f"告警指令样本: {alert_count} 条")
    print(f"问答指令样本: {qa_count} 条")

    # 数据质量检查
    print(f"\n=== 数据质量检查 ===")
    print(f"样本多样性:")
    print(f"  - 问答问题类型: {len(generator.identity_questions) + len(generator.core_questions)} 类")
    print(f"  - 告警地点多样性: {len(generator.locations)} 个不同地点")
    print(f"  - 告警模板多样性: {len(generator.alert_templates)} 种模板")
    print(f"  - 帧比例范围: 80%-100%")

    # 随机展示几个样本
    print(f"\n=== 随机样本展示 ===")
    for i in range(3):
        sample = random.choice(all_data)
        print(f"样本 {i + 1}:")
        print(f"  指令: {sample['instruction']}")
        print(f"  输入: {sample['input'][:50]}...")
        print(f"  输出: {sample['output'][:50]}...")
        print()


if __name__ == "__main__":
    main()