import json
import random
import re


def update_percentages(input_file, output_file):
    """
    更新百分比数据，确保input和output中的百分比一致
    """
    # 20种详细的告警处理指导模板
    alert_output_templates = [
        "📋📊事件分析报告：{time}于{location}检测到摔倒，摔倒检测准确度为{percentage}%，您收到监控AI的摔倒告警，烦请先通过监控核实现场情况，随即通知就近安保/医护人员赶赴，危急时拨打120并说明位置，留存监控录像，跟进进展并上报，事后完成事件记录。",

        "🚨紧急处理通知：{time}在{location}发生摔倒事件，系统检测准确度{percentage}%。请立即：1)查看实时监控确认情况 2)通知现场人员前往协助 3)必要时呼叫急救 4)记录事件详情 5)持续关注后续状态。",

        "⚠️安全事件响应：{time}于{location}检测到人员摔倒，准确率{percentage}%。处理步骤：①核实监控画面 ②派遣人员现场查看 ③评估是否需要医疗援助 ④维护现场秩序 ⑤完整记录处理过程。",

        "🔴高危事件处理：{time}在{location}识别摔倒，检测可信度{percentage}%。请执行：1.远程确认现场状况 2.协调安保医疗资源 3.准备应急响应 4.保护当事人隐私 5.系统记录归档。",

        "👥人员安全告警：{time}于{location}监测到跌倒，准确度{percentage}%。处理流程：查看监控→派人协助→医疗评估→事件记录→后续跟进，确保人员安全为第一要务。",

        "📞应急响应指南：{time}在{location}检测摔倒，系统置信度{percentage}%。操作步骤：1.立即调取该区域监控 2.通知距离最近的工作人员 3.准备急救设备 4.如需医疗支援立即拨打120 5.完整记录响应时间及处理过程。",

        "🏥医疗安全告警：{time}于{location}发现摔倒，检测准确率{percentage}%。请按医疗应急流程：①确认意识状态 ②检查有无外伤 ③评估移动能力 ④决定是否送医 ⑤完成医疗记录。",

        "🆘紧急处理预案：{time}在{location}识别跌倒事件，准确度{percentage}%。执行：A.监控核实 B.人员派遣 C.医疗评估 D.家属通知 E.报告撰写 F.预防措施复查。",

        "📹监控事件处理：{time}于{location}检测到摔倒，系统准确度{percentage}%。处理：1.保存事件前后录像 2.多角度监控确认 3.现场人员实地查看 4.医疗需求判断 5.完整事件报告生成。",

        "👴长者关怀响应：{time}在{location}监测老人摔倒，准确率{percentage}%。特别注意：温柔询问感受、检查骨骼损伤、避免匆忙移动、评估认知状态、联系家属告知、建议医疗检查。",

        "🔍事件深度处理：{time}于{location}检测摔倒，可信度{percentage}%。专业流程：现场评估→生命体征检查→伤害程度判断→医疗干预决策→康复计划制定→预防措施强化。",

        "📊数据分析响应：{time}在{location}发生摔倒，检测精度{percentage}%。基于数据驱动响应：分析摔倒模式、评估环境因素、优化监控布局、加强高风险区域巡查、完善应急预案。",

        "🏃‍♂️快速响应协议：{time}于{location}识别跌倒，准确度{percentage}%。快速行动：30秒内监控确认、1分钟内人员派遣、3分钟内现场评估、5分钟内医疗决策、10分钟内完整记录。",

        "🛡️安全防护处理：{time}在{location}检测摔倒，系统置信度{percentage}%。防护措施：立即设置警示区域、疏散围观人员、保护当事人隐私、防止二次伤害、进行安全整改。",

        "💊医疗应急处理：{time}于{location}发现摔倒，准确率{percentage}%。医疗优先：评估意识状态→检查生命体征→处理外伤→决定转运方式→联系接收医院→交接医疗信息。",

        "📝全流程管理：{time}在{location}监测摔倒，检测准确度{percentage}%。管理闭环：事件发现→现场响应→医疗处置→家属沟通→原因分析→整改落实→效果评估。",

        "🌐智慧照护响应：{time}于{location}识别跌倒，可信度{percentage}%。智能照护流程：AI检测告警→远程监控确认→智能调度资源→数字化记录→数据分析优化。",

        "⚕️专业护理指导：{time}在{location}检测摔倒，准确率{percentage}%。护理规范：不随意移动伤者、检查受伤部位、监测生命体征、记录症状变化、及时专业医疗介入。",

        "🔔持续监护方案：{time}于{location}发生摔倒，系统精度{percentage}%。后续监护：加强该区域监控、增加巡查频次、评估照护方案、进行安全培训、定期复查效果。",

        "📈质量改进响应：{time}在{location}识别跌倒事件，检测可信度{percentage}%。改进循环：立即处理当前事件→分析根本原因→制定预防措施→培训相关人员→监控改进效果。"
    ]

    processed_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())

                # 只处理告警助手类型的样本
                if (data.get(
                        "instruction") == "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户"):

                    # 从input中提取信息
                    input_text = data["input"]

                    # 使用正则表达式提取时间、地点信息
                    time_match = re.search(r'(\d{2}:\d{2}:\d{2})', input_text)
                    location_match = re.search(r'[在：]([^，。！？\d]+?)(?=检测|发现|监测|识别)', input_text)

                    # 如果没有匹配到，尝试其他模式
                    if not time_match:
                        time_match = re.search(r'时间[：:](\d{2}:\d{2}:\d{2})', input_text)
                    if not location_match:
                        location_match = re.search(r'地点[：:]([^，。！？\d]+)', input_text)

                    time = time_match.group(1) if time_match else "未知时间"
                    location = location_match.group(1).strip() if location_match else "未知区域"

                    # 检查input中是否有"大于80%"
                    if "大于80%" in input_text:
                        # 生成81-99之间的随机数
                        new_percentage = random.randint(81, 99)

                        # 随机选择替换方式："为X%"或"是X%"
                        replacement = random.choice(["为", "是"]) + f"{new_percentage}%"

                        # 替换input中的"大于80%"
                        new_input = input_text.replace("大于80%", replacement)
                        data["input"] = new_input

                        # 使用相同的百分比生成新的output
                        template = random.choice(alert_output_templates)
                        new_output = template.format(
                            time=time,
                            location=location,
                            percentage=new_percentage
                        )
                        data["output"] = new_output
                    else:
                        # 如果input中没有"大于80%"，检查是否有其他百分比
                        input_percentage_match = re.search(r'(\d{1,3})%', input_text)

                        if input_percentage_match:
                            # 如果input中有百分比，使用相同的百分比
                            percentage = int(input_percentage_match.group(1))
                        else:
                            # 如果input中没有百分比，随机生成一个
                            percentage = random.randint(81, 99)

                        # 使用百分比生成新的output
                        template = random.choice(alert_output_templates)
                        new_output = template.format(
                            time=time,
                            location=location,
                            percentage=percentage
                        )
                        data["output"] = new_output

                    processed_count += 1

                # 写入处理后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"处理第{line_num}行时出错: {e}")
                error_count += 1
                # 出错时原样写入
                outfile.write(line)

    return processed_count, error_count


def main():
    input_file = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\enhanced_alert_samples.jsonl"
    output_file = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\ddddddddddd.jsonl"

    print("开始更新百分比数据...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    processed_count, error_count = update_percentages(input_file, output_file)

    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {processed_count} 条告警样本")
    print(f"处理错误: {error_count} 条")
    print(f"输出文件已保存至: {output_file}")

    # 显示几个示例
    print(f"\n=== 处理前后对比示例 ===")
    with open(output_file, 'r', encoding='utf-8') as f:
        alert_samples = []
        for line in f:
            data = json.loads(line.strip())
            if data.get("instruction") == "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户":
                alert_samples.append(data)
                if len(alert_samples) >= 3:
                    break

        for i, sample in enumerate(alert_samples, 1):
            print(f"示例 {i}:")
            print(f"  Input: {sample['input']}")

            # 从input中提取百分比
            input_percentage_match = re.search(r'(\d{1,3})%', sample["input"])
            input_percentage = input_percentage_match.group(1) if input_percentage_match else "未找到"

            # 从output中提取百分比
            output_percentage_match = re.search(r'(\d{1,3})%', sample["output"])
            output_percentage = output_percentage_match.group(1) if output_percentage_match else "未找到"

            print(f"  Input百分比: {input_percentage}%")
            print(f"  Output百分比: {output_percentage}%")
            print(f"  Output: {sample['output']}")
            print()


if __name__ == "__main__":
    main()