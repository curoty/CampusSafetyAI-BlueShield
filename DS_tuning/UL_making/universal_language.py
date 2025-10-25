import json
import random
from datetime import datetime, timedelta


class ExtendedTrainingDataGenerator:
    def __init__(self):
        # 指令模板
        self.general_instruction = "请根据你的知识回答以下问题"
        self.mixed_instruction = "请结合你的通用知识和项目专业知识回答以下问题"

        # 通用知识问答库 - 涵盖多个领域
        self.general_knowledge_qa = {
            "science_technology": [
                {
                    "question": "什么是人工智能？",
                    "answer": "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器和软件。这些任务包括学习、推理、问题解决、感知和语言理解。"
                },
                {
                    "question": "机器学习有哪些主要类型？",
                    "answer": "机器学习主要分为监督学习、无监督学习、半监督学习和强化学习。监督学习使用标记数据训练模型，无监督学习发现数据中的模式，半监督学习结合两者，强化学习通过试错学习最优策略。"
                },
                {
                    "question": "Python是什么编程语言？",
                    "answer": "Python是一种高级、解释型的通用编程语言，以其简洁的语法和强大的库生态系统而闻名。它广泛应用于Web开发、数据科学、人工智能、自动化和科学计算等领域。"
                },
                {
                    "question": "深度学习与机器学习有什么区别？",
                    "answer": "深度学习是机器学习的一个子集，使用包含多个层次的人工神经网络。与传统机器学习相比，深度学习能够自动学习数据的层次化特征表示，在处理复杂模式识别任务时表现更好。"
                },
                {
                    "question": "什么是神经网络？",
                    "answer": "神经网络是受人脑结构启发的计算模型，由相互连接的神经元层组成。每个神经元接收输入，进行加权求和并通过激活函数产生输出，通过训练调整权重来学习数据中的模式。"
                },
                {
                    "question": "计算机视觉是什么？",
                    "answer": "计算机视觉是人工智能的一个领域，使计算机能够从数字图像或视频中获取、处理、分析和理解高级信息。它包括图像识别、目标检测、图像分割等任务。"
                },
                {
                    "question": "什么是自然语言处理？",
                    "answer": "自然语言处理是人工智能的一个分支，关注计算机与人类语言之间的交互。它包括语言理解、语言生成、机器翻译、情感分析等技术，使计算机能够处理和理解人类语言。"
                },
                {
                    "question": "大数据的特点是什么？",
                    "answer": "大数据通常用4V来描述：Volume（大量）、Velocity（高速）、Variety（多样）、Veracity（真实性）。这些特点使得传统数据处理工具难以有效处理大数据。"
                },
                {
                    "question": "云计算有哪些服务模式？",
                    "answer": "云计算主要有三种服务模式：IaaS（基础设施即服务）、PaaS（平台即服务）和SaaS（软件即服务）。IaaS提供基础计算资源，PaaS提供开发平台，SaaS提供完整应用程序。"
                },
                {
                    "question": "什么是物联网？",
                    "answer": "物联网是指通过互联网相互连接的物理设备网络，这些设备能够收集和交换数据。物联网设备包括传感器、家用电器、车辆等，实现智能化和自动化控制。"
                }
            ],
            "programming_development": [
                {
                    "question": "如何学习编程？",
                    "answer": "学习编程可以从选择一门适合初学者的语言开始，如Python，然后通过在线教程、实践项目和参与开源社区来提升技能。重要的是要多写代码，理解基本概念，并逐步挑战更复杂的项目。"
                },
                {
                    "question": "什么是版本控制？",
                    "answer": "版本控制是管理代码变更的系统，允许开发人员跟踪修改、协作开发并在需要时回滚到之前的版本。Git是最流行的版本控制系统，GitHub和GitLab是基于Git的代码托管平台。"
                },
                {
                    "question": "软件开发的生命周期是什么？",
                    "answer": "软件开发生命周期包括需求分析、设计、实现、测试、部署和维护等阶段。常见的开发模型有瀑布模型、敏捷开发、DevOps等，不同模型强调不同的开发流程和协作方式。"
                },
                {
                    "question": "什么是API？",
                    "answer": "API是应用程序编程接口，定义了不同软件组件之间交互的规范。它允许不同的应用程序相互通信和共享数据，是现代软件开发中实现模块化和集成的基础。"
                },
                {
                    "question": "如何调试程序？",
                    "answer": "调试程序的方法包括使用调试器逐步执行代码、添加日志语句、分析错误信息、代码审查和单元测试。系统性的调试需要理解程序逻辑、数据流和可能的错误模式。"
                },
                {
                    "question": "什么是数据结构？",
                    "answer": "数据结构是组织和存储数据的方式，常见的数据结构包括数组、链表、栈、队列、树、图等。选择合适的数据结构对算法效率和程序性能有重要影响。"
                },
                {
                    "question": "算法的时间复杂度是什么？",
                    "answer": "时间复杂度描述算法执行时间随输入规模增长的变化趋势，常用大O表示法表示。常见的时间复杂度有O(1)、O(log n)、O(n)、O(n log n)、O(n²)等，帮助评估算法效率。"
                },
                {
                    "question": "什么是面向对象编程？",
                    "answer": "面向对象编程是一种编程范式，基于对象的概念，对象包含数据和方法。OOP的核心概念包括封装、继承、多态和抽象，提高了代码的可重用性、可维护性和可扩展性。"
                },
                {
                    "question": "如何优化代码性能？",
                    "answer": "代码性能优化包括算法优化、数据结构选择、减少不必要的计算、使用缓存、并行处理等方法。性能分析工具可以帮助识别瓶颈，优化应该在保证代码可读性和可维护性的前提下进行。"
                },
                {
                    "question": "什么是设计模式？",
                    "answer": "设计模式是解决常见软件设计问题的可重用方案，包括创建型模式、结构型模式和行为型模式。常见的设计模式有单例模式、工厂模式、观察者模式等，它们提供了经过验证的解决方案。"
                }
            ],
            "academic_education": [
                {
                    "question": "如何提高学习效率？",
                    "answer": "提高学习效率的方法包括制定明确的学习目标、使用主动学习方法、定期复习、保持充足睡眠、减少干扰、采用多种学习资源以及实践应用所学知识。"
                },
                {
                    "question": "什么是批判性思维？",
                    "answer": "批判性思维是一种理性、反思的思维方式，涉及对信息的分析、评估和推理。它包括识别假设、评估证据、识别逻辑错误和形成基于证据的结论。"
                },
                {
                    "question": "大学教育的重要性是什么？",
                    "answer": "大学教育不仅提供专业知识和技能，还培养批判性思维、解决问题的能力、沟通技巧和终身学习的能力。它为个人发展、职业准备和社会参与提供了重要基础。"
                },
                {
                    "question": "如何写好学术论文？",
                    "answer": "写好学术论文需要明确研究问题、进行充分的文献综述、构建清晰的论文结构、使用恰当的研究方法、准确呈现结果、进行深入讨论以及遵循学术写作规范。"
                },
                {
                    "question": "团队合作的重要性是什么？",
                    "answer": "团队合作能够整合不同成员的技能和视角，提高问题解决的效率和质量。它培养沟通能力、冲突解决能力和领导力，是现代工作环境中不可或缺的能力。"
                },
                {
                    "question": "什么是终身学习？",
                    "answer": "终身学习是指在个人一生中持续获取知识、技能和态度的过程。在快速变化的现代社会中，终身学习对于个人发展、职业适应和社会参与至关重要。"
                },
                {
                    "question": "如何管理时间？",
                    "answer": "有效的时间管理包括设定优先级、制定计划、避免拖延、减少干扰、合理分配时间和定期反思调整。工具如待办事项列表、日历和时间跟踪应用可以提供帮助。"
                },
                {
                    "question": "什么是研究性学习？",
                    "answer": "研究性学习是一种以学生为中心的教学方法，学生通过提出问题、收集信息、分析数据和得出结论来主动构建知识。它培养独立思考和解决问题的能力。"
                },
                {
                    "question": "如何准备考试？",
                    "answer": "有效准备考试需要制定复习计划、理解而非死记硬背、进行练习测试、组织学习小组、保持健康的生活方式以及在考试前保证充足休息。"
                },
                {
                    "question": "在线学习的优缺点是什么？",
                    "answer": "在线学习的优点包括灵活性、可访问性和丰富的资源；缺点可能包括缺乏面对面互动、需要更强的自律性和可能的技术问题。成功的在线学习需要良好的时间管理和自我激励。"
                }
            ],
            "daily_life": [
                {
                    "question": "如何保持健康的生活方式？",
                    "answer": "保持健康的生活方式包括均衡饮食、规律运动、充足睡眠、压力管理和避免有害习惯。定期体检和保持积极的社会联系也对整体健康很重要。"
                },
                {
                    "question": "什么是心理健康？",
                    "answer": "心理健康指情感、心理和社会福祉的状态，影响我们的思考、感受和行动。它涉及有效应对生活压力、实现潜能、生产力工作和为社会做贡献的能力。"
                },
                {
                    "question": "如何管理压力？",
                    "answer": "压力管理技巧包括识别压力源、练习放松技巧如深呼吸和冥想、保持体育活动、建立支持网络、设定现实目标以及必要时寻求专业帮助。"
                },
                {
                    "question": "什么是有效的沟通？",
                    "answer": "有效沟通包括清晰表达想法、积极倾听、理解非语言信号、考虑受众和情境以及提供建设性反馈。它对于建立良好关系和成功协作至关重要。"
                },
                {
                    "question": "如何建立良好的人际关系？",
                    "answer": "建立良好人际关系需要真诚、尊重、信任、有效沟通、同理心和相互支持。花时间了解他人、保持开放心态和解决冲突的能力也很重要。"
                },
                {
                    "question": "什么是财务规划？",
                    "answer": "财务规划是管理个人或家庭财务以实现生活目标的过程，包括预算编制、储蓄、投资、保险和退休规划。良好的财务规划有助于实现财务安全和独立。"
                },
                {
                    "question": "如何做出重要决策？",
                    "answer": "做出重要决策需要明确问题、收集相关信息、考虑各种选项、评估利弊、咨询他人意见、考虑长期影响，并在充分思考后采取行动。"
                },
                {
                    "question": "什么是职业规划？",
                    "answer": "职业规划是评估个人技能、兴趣和价值观，设定职业目标，并制定实现这些目标的策略的过程。它包括技能发展、网络建设和持续学习。"
                },
                {
                    "question": "如何培养创造力？",
                    "answer": "培养创造力可以通过接触多样化的体验、练习头脑风暴、接受不确定性、从失败中学习、保持好奇心和为创造性思维留出专门时间来实现。"
                },
                {
                    "question": "什么是情商？",
                    "answer": "情商是识别、理解和管理自己及他人情绪的能力。它包括自我意识、自我调节、动机、同理心和社交技能，对个人和职业成功有重要影响。"
                }
            ]
        }

        # 项目相关知识 - 用于混合知识问答
        self.project_knowledge = {
            "team": "本项目由三位河南高校人工智能专业的大二学生开发，他们具备扎实的Python、机器学习及大模型部署微调基础。",
            "function": "项目核心功能为智能摔倒检测与高危告警，通过YOLO模型实时检测视频流中人体状态，结合状态机逻辑判断摔倒风险。",
            "technology": "项目采用YOLO进行目标检测，DeepSeek-7B大模型生成告警，状态机进行帧比例风险评估，RAG记忆层存储历史数据。",
            "workflow": "系统工作流程包括视频采集、状态检测、风险评估、告警生成和结果推送，形成完整的处理闭环。",
            "application": "项目适用于养老院、医院、学校等需要安全监护的场所，特别关注老年人、病人等高风险群体。"
        }

        # 同义词库用于增加多样性
        self.synonyms = {
            "什么是": ["请解释", "请说明", "请介绍", "什么是", "何谓", "请定义"],
            "如何": ["怎样", "怎么", "如何", "请说明方法", "请介绍方式"],
            "优点": ["优势", "好处", "益处", "长处", "积极方面"],
            "缺点": ["不足", "局限", "弱点", "缺陷", "消极方面"],
            "重要": ["关键", "必要", "紧要", "重要", "至关重要"],
            "方法": ["方式", "途径", "手段", "策略", "技巧"],
            "包括": ["包含", "涵盖", "涉及", "由...组成", "包括"],
            "因为": ["由于", "鉴于", "因为", "基于", "出于"],
            "所以": ["因此", "因而", "于是", "所以", "从而"],
            "但是": ["然而", "可是", "不过", "但", "却"]
        }

    def apply_synonym_replacement(self, text, replacement_rate=0.3):
        """应用同义词替换增加多样性"""
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

    def generate_general_knowledge_samples(self, num_samples=3200):
        """生成通用知识样本"""
        samples = []

        # 计算每个类别大致需要的样本数
        categories = list(self.general_knowledge_qa.keys())
        samples_per_category = num_samples // len(categories)

        for category in categories:
            qa_pairs = self.general_knowledge_qa[category]

            # 重复使用QA对直到达到所需数量
            for i in range(samples_per_category):
                qa_pair = random.choice(qa_pairs)

                # 对问题和答案进行同义词替换增加多样性
                question = self.apply_synonym_replacement(qa_pair["question"])
                answer = self.apply_synonym_replacement(qa_pair["answer"])

                sample = {
                    "instruction": self.general_instruction,
                    "input": question,
                    "output": answer
                }
                samples.append(sample)

        # 如果样本数量不足，用随机QA对补足
        while len(samples) < num_samples:
            category = random.choice(categories)
            qa_pair = random.choice(self.general_knowledge_qa[category])

            question = self.apply_synonym_replacement(qa_pair["question"])
            answer = self.apply_synonym_replacement(qa_pair["answer"])

            sample = {
                "instruction": self.general_instruction,
                "input": question,
                "output": answer
            }
            samples.append(sample)

        return samples

    def generate_mixed_knowledge_samples(self, num_samples=3200):
        """生成混合知识样本"""
        samples = []

        # 混合知识问题模板
        mixed_question_templates = [
            "从{general_topic}的角度来看，{project_aspect}有什么意义？",
            "如何将{general_concept}应用到{project_context}中？",
            "在{project_domain}领域，{general_principle}起到了什么作用？",
            "比较{general_topic}和{project_technology}在{context}中的异同",
            "如何用{general_methodology}来优化{project_functionality}？",
            "从{general_perspective}分析{project_application}的优势和挑战",
            "{general_concept}如何影响{project_development}的发展？",
            "在{project_scenario}中，如何运用{general_approach}解决问题？",
            "{general_technology}与{project_technology}在{domain}中的协同作用",
            "如何基于{general_framework}设计更好的{project_solution}？"
        ]

        # 通用话题和概念
        general_topics = [
            "人工智能发展", "机器学习原理", "深度学习架构", "计算机视觉技术",
            "自然语言处理", "大数据分析", "云计算平台", "物联网应用",
            "软件工程方法", "项目管理", "用户体验设计", "数据安全",
            "算法优化", "系统架构", "技术创新", "行业发展"
        ]

        general_concepts = [
            "监督学习", "无监督学习", "神经网络", "卷积神经网络",
            "循环神经网络", "Transformer架构", "迁移学习", "强化学习",
            "敏捷开发", "DevOps", "微服务架构", "容器化技术",
            "持续集成", "测试驱动开发", "代码重构", "性能优化"
        ]

        # 项目相关元素
        project_aspects = [
            "摔倒检测系统", "实时监控方案", "高危告警机制", "状态机设计",
            "帧比例分析", "YOLO目标检测", "DeepSeek大模型", "RAG记忆层",
            "Web前端展示", "历史数据管理", "风险评估逻辑", "视频流处理"
        ]

        project_contexts = [
            "老年人安全监护", "医疗康复环境", "智能养老场景", "公共场所监控",
            "实时风险识别", "高危事件处理", "历史数据分析", "系统性能优化"
        ]

        for i in range(num_samples):
            template = random.choice(mixed_question_templates)

            # 根据模板类型填充内容
            if "general_topic" in template and "project_aspect" in template:
                question = template.format(
                    general_topic=random.choice(general_topics),
                    project_aspect=random.choice(project_aspects)
                )
            elif "general_concept" in template and "project_context" in template:
                question = template.format(
                    general_concept=random.choice(general_concepts),
                    project_context=random.choice(project_contexts)
                )
            elif "project_domain" in template and "general_principle" in template:
                question = template.format(
                    project_domain=random.choice(["摔倒检测", "安全监控", "智能告警"]),
                    general_principle=random.choice(general_concepts)
                )
            else:
                # 通用填充
                question = template.format(
                    general_topic=random.choice(general_topics),
                    project_aspect=random.choice(project_aspects),
                    general_concept=random.choice(general_concepts),
                    project_context=random.choice(project_contexts),
                    project_domain=random.choice(["摔倒检测", "安全监控"]),
                    general_principle=random.choice(general_concepts),
                    context=random.choice(["技术实现", "应用场景", "性能优化"]),
                    general_methodology=random.choice(general_concepts),
                    project_functionality=random.choice(project_aspects),
                    general_perspective=random.choice(general_topics),
                    project_application=random.choice(project_contexts),
                    general_approach=random.choice(general_concepts),
                    project_development=random.choice(project_aspects),
                    project_scenario=random.choice(project_contexts),
                    general_technology=random.choice(general_concepts),
                    project_technology=random.choice(["YOLO检测", "状态机逻辑", "大模型生成"]),
                    domain=random.choice(["智能监控", "安全保障"]),
                    general_framework=random.choice(general_concepts),
                    project_solution=random.choice(project_aspects)
                )

            # 生成混合知识回答
            answer = self.generate_mixed_answer(question)

            sample = {
                "instruction": self.mixed_instruction,
                "input": question,
                "output": answer
            }
            samples.append(sample)

        return samples

    def generate_mixed_answer(self, question):
        """为混合知识问题生成回答"""
        # 基于问题内容生成相关的混合回答
        answer_templates = [
            "从通用知识角度来看，{general_insight}。在项目实践中，{project_application}。这种结合使得{benefit}。",
            "通用领域中的{general_concept}为项目提供了{theoretical_basis}。在我们的摔倒检测系统中，{project_implementation}。这种应用带来了{advantage}。",
            "基于{general_principle}，我们设计了{project_solution}。具体来说，{technical_details}。这种方法在{scenario}中表现出{effectiveness}。",
            "通用技术如{general_technology}与项目需求相结合，形成了{integrated_approach}。在我们的实现中，{specific_implementation}，这导致了{improvement}。",
            "从{general_perspective}分析，{project_domain}需要{requirement}。我们的解决方案结合了{technical_combination}，实现了{achievement}。"
        ]

        template = random.choice(answer_templates)

        # 填充回答模板
        general_insights = [
            "机器学习的基本原理强调从数据中学习模式",
            "深度学习通过多层次特征提取提升识别精度",
            "计算机视觉技术能够从图像中提取有价值信息",
            "实时系统设计需要考虑性能和响应时间的平衡",
            "软件工程的最佳实践强调模块化和可维护性"
        ]

        project_applications = [
            "我们利用状态机进行持续风险评估",
            "通过帧比例分析确保告警的准确性",
            "YOLO模型提供了高效的目标检测能力",
            "DeepSeek大模型生成自然语言告警信息",
            "RAG记忆层实现了历史数据的智能管理"
        ]

        benefits = [
            "系统既具备理论基础又满足实际应用需求",
            "解决方案在准确性和效率之间取得良好平衡",
            "技术组合提供了可靠的安全保障能力",
            "系统设计兼顾了技术创新和实用价值",
            "这种方法确保了长期可维护性和扩展性"
        ]

        # 根据模板类型填充内容
        if "general_insight" in template:
            answer = template.format(
                general_insight=random.choice(general_insights),
                project_application=random.choice(project_applications),
                benefit=random.choice(benefits)
            )
        else:
            answer = template.format(
                general_insight=random.choice(general_insights),
                project_application=random.choice(project_applications),
                benefit=random.choice(benefits),
                general_concept=random.choice(["神经网络", "深度学习", "计算机视觉"]),
                theoretical_basis=random.choice(["理论基础", "方法指导", "技术支撑"]),
                project_implementation=random.choice(project_applications),
                advantage=random.choice(benefits),
                general_principle=random.choice(["模块化设计", "实时处理", "风险评估"]),
                project_solution=random.choice(["帧比例验证机制", "多级检测流程", "智能告警生成"]),
                technical_details=random.choice(project_applications),
                scenario=random.choice(["养老院监控", "医院安全", "公共场所"]),
                effectiveness=random.choice(["良好的检测效果", "高准确率", "及时响应"]),
                general_technology=random.choice(["目标检测", "自然语言处理", "状态管理"]),
                integrated_approach=random.choice(["综合解决方案", "协同技术架构", "一体化设计"]),
                specific_implementation=random.choice(project_applications),
                improvement=random.choice(["性能提升", "准确性提高", "用户体验改善"]),
                general_perspective=random.choice(["技术架构", "用户体验", "系统性能"]),
                project_domain=random.choice(["摔倒检测", "安全监控", "智能告警"]),
                requirement=random.choice(["高可靠性", "实时响应", "准确识别"]),
                technical_combination=random.choice(["视觉检测与语言生成", "实时分析与历史管理", "风险评估与告警推送"]),
                achievement=random.choice(["全面的安全监护", "高效的异常处理", "智能的风险预警"])
            )

        return answer

    def save_to_jsonl(self, data, filename):
        """保存数据为JSONL格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    generator = ExtendedTrainingDataGenerator()

    print("开始生成通用知识和混合知识微调样本...")

    # 生成通用知识样本
    print("生成3200条通用知识样本...")
    general_samples = generator.generate_general_knowledge_samples(3200)

    # 生成混合知识样本
    print("生成3200条混合知识样本...")
    mixed_samples = generator.generate_mixed_knowledge_samples(3200)

    # 保存为JSONL文件
    print("保存通用知识样本...")
    generator.save_to_jsonl(general_samples, "general_knowledge_samples.jsonl")

    print("保存混合知识样本...")
    generator.save_to_jsonl(mixed_samples, "mixed_knowledge_samples.jsonl")

    # 统计信息
    print(f"\n=== 数据生成完成 ===")
    print(f"通用知识样本: {len(general_samples)} 条")
    print(f"混合知识样本: {len(mixed_samples)} 条")
    print(f"通用知识指令: {generator.general_instruction}")
    print(f"混合知识指令: {generator.mixed_instruction}")

    # 显示样本示例
    print(f"\n=== 样本示例 ===")
    print("通用知识样本示例:")
    print(f"指令: {general_samples[0]['instruction']}")
    print(f"输入: {general_samples[0]['input']}")
    print(f"输出: {general_samples[0]['output'][:100]}...")

    print(f"\n混合知识样本示例:")
    print(f"指令: {mixed_samples[0]['instruction']}")
    print(f"输入: {mixed_samples[0]['input']}")
    print(f"输出: {mixed_samples[0]['output'][:100]}...")


if __name__ == "__main__":
    main()