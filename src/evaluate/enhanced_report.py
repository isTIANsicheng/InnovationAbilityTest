#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版报告生成模块
基于完整对话历史和评估数据生成全面的学生创新思维能力评估报告
"""

import os
import json
from datetime import datetime


def format_dialogue_history(dialogue_list):
    """
    格式化对话历史
    
    参数:
        dialogue_list: 对话历史列表
        
    返回:
        str: 格式化后的对话历史文本
    """
    if not dialogue_list:
        return "无对话记录"
    
    formatted_text = ""
    for i, entry in enumerate(dialogue_list):
        formatted_text += f"==== 第 {i+1} 轮对话 ====\n"
        formatted_text += f"问题: {entry.get('question', '未知问题')}\n"
        formatted_text += f"回答: {entry.get('response', '未知回答')}\n"
        
        # 添加评分信息
        if 'raw_score' in entry:
            formatted_text += f"原始分数: {entry.get('raw_score', 0)}\n"
        
        if 'adjusted_score' in entry:
            formatted_text += f"调整后分数: {entry.get('adjusted_score', 0)}\n"
        
        # 添加反馈信息
        if 'feedback' in entry:
            feedback = entry.get('feedback', '无反馈')
            # 限制反馈长度，避免报告过长
            if len(feedback) > 200:
                feedback = feedback[:200] + "..."
            formatted_text += f"反馈: {feedback}\n"
        
        formatted_text += "\n"
    
    return formatted_text


def format_feedback_records(feedback_records, feedback_type):
    """
    格式化反馈记录
    
    参数:
        feedback_records: 反馈记录列表
        feedback_type: 反馈类型 ('success' 或 'failure')
        
    返回:
        str: 格式化后的反馈记录文本
    """
    if not feedback_records:
        return "无反馈记录"
    
    formatted_text = ""
    for i, record in enumerate(feedback_records):
        if record.get('type') == feedback_type:
            formatted_text += f"- 轮次 {record.get('step', i+1)}:\n"
            formatted_text += f"  问题: {record.get('question', '未知问题')}\n"
            formatted_text += f"  回答: {record.get('response', '未知回答')}\n"
            formatted_text += f"  分数: {record.get('adjusted_score', 0)}\n"
            formatted_text += f"  分数变化: {record.get('score_change', 0)}\n"
            
            # 添加反馈信息
            if 'feedback' in record:
                feedback = record.get('feedback', '无反馈')
                # 限制反馈长度
                if len(feedback) > 150:
                    feedback = feedback[:150] + "..."
                formatted_text += f"  反馈: {feedback}\n"
            
            formatted_text += "\n"
    
    if not formatted_text:
        return "无该类型反馈记录"
    
    return formatted_text


def format_adjustment_history(adjustment_history):
    """
    格式化分数调整历史
    
    参数:
        adjustment_history: 分数调整历史字典
        
    返回:
        str: 格式化后的调整历史文本
    """
    if not adjustment_history:
        return "无调整历史"
    
    formatted_text = ""
    for dimension, adjustments in adjustment_history.items():
        formatted_text += f"## {dimension}维度调整历史\n"
        if not adjustments:
            formatted_text += "无调整记录\n\n"
            continue
            
        # 只取最近几次调整
        recent_adjustments = adjustments[-5:] if len(adjustments) > 5 else adjustments
        
        for i, adj in enumerate(recent_adjustments):
            formatted_text += f"- 调整 {i+1}:\n"
            formatted_text += f"  原始分数: {adj.get('original_score', 0)}\n"
            formatted_text += f"  历史平均: {adj.get('history_avg', 0)}\n"
            formatted_text += f"  最终调整: {adj.get('final_adjustment', 0)}\n"
            formatted_text += f"  原因: {adj.get('reason', '未知原因')}\n\n"
    
    return formatted_text


def generate_enhanced_report(llm, evaluation_results):
    """
    生成增强版评估报告
    
    参数:
        llm: 语言模型接口
        evaluation_results: 包含完整评估数据的结果字典
        
    返回:
        str: 生成的报告文本
    """
    student_id = evaluation_results.get('student_id', 'unknown')
    timestamp = evaluation_results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 提取各维度评分
    dimension_scores = {}
    dimension_confidences = {}
    dimension_reports = {}
    
    if 'dimensions' in evaluation_results:
        for dimension, result in evaluation_results['dimensions'].items():
            dimension_scores[dimension] = result.get('score', 0)
            dimension_confidences[dimension] = result.get('confidence', 0.7)
            dimension_reports[dimension] = result
    
    overall_score = evaluation_results.get('overall_score', 0)
    if overall_score == 0 and dimension_scores:
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
    
    # 格式化每个维度的对话历史
    formatted_dialogues = {}
    for dimension, report in dimension_reports.items():
        if 'dialogue_history' in report:
            formatted_dialogues[dimension] = format_dialogue_history(report['dialogue_history'])
        else:
            formatted_dialogues[dimension] = "无对话记录"
    
    # 分析学习趋势
    learning_trends = {}
    for dimension, report in dimension_reports.items():
        if 'dialogue_history' in report and len(report['dialogue_history']) > 1:
            scores = [entry.get('adjusted_score', 0) for entry in report['dialogue_history']]
            
            if len(scores) >= 3:
                first_score = scores[0]
                last_score = scores[-1]
                
                # 计算总体变化
                overall_change = last_score - first_score
                
                # 计算平均变化率
                changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
                avg_change = sum(changes) / len(changes)
                
                # 判断趋势
                if overall_change > 1:
                    trend = "显著提升"
                elif overall_change > 0.5:
                    trend = "稳步提升"
                elif overall_change > -0.5:
                    trend = "基本稳定"
                elif overall_change > -1:
                    trend = "略有下降"
                else:
                    trend = "明显下降"
                
                learning_trends[dimension] = {
                    "first_score": first_score,
                    "last_score": last_score,
                    "overall_change": overall_change,
                    "avg_change": avg_change,
                    "trend": trend,
                    "scores": scores
                }
            else:
                learning_trends[dimension] = {"trend": "数据不足，无法分析趋势"}
        else:
            learning_trends[dimension] = {"trend": "无对话记录，无法分析趋势"}
    
    # 获取成功/失败反馈记录
    feedback_records = evaluation_results.get('feedback_records', {})
    
    # 格式化成功/失败反馈
    formatted_feedback = {}
    for dimension in ['flexibility', 'fluency', 'effectiveness']:
        dimension_records = feedback_records.get(dimension, [])
        formatted_feedback[f"{dimension}_success"] = format_feedback_records(dimension_records, 'success')
        formatted_feedback[f"{dimension}_failure"] = format_feedback_records(dimension_records, 'failure')
    
    # 格式化分数调整历史
    adjustment_history = evaluation_results.get('adjustment_history', {})
    formatted_adjustments = format_adjustment_history(adjustment_history)
    
    # 构建完整对话历史摘要
    conversation_summary = (
        "【灵活性维度对话历史】\n" + formatted_dialogues.get('flexibility', "无对话记录") + "\n\n" +
        "【流畅性维度对话历史】\n" + formatted_dialogues.get('fluency', "无对话记录") + "\n\n" +
        "【有效性维度对话历史】\n" + formatted_dialogues.get('effectiveness', "无对话记录")
    )
    
    # 准备LLM提示
    report_prompt = f"""你是一位专业的创新思维能力评估专家。请根据以下信息，生成一份全面详细的学生创新思维能力评估报告。

## 基本信息
- 学生ID: {student_id}
- 评估时间: {timestamp}
- 总体得分: {overall_score:.2f}/10

## 各维度得分
- 灵活性(flexibility): {dimension_scores.get('flexibility', 0):.2f}/10，置信度: {dimension_confidences.get('flexibility', 0.7):.2f}
- 流畅性(fluency): {dimension_scores.get('fluency', 0):.2f}/10，置信度: {dimension_confidences.get('fluency', 0.7):.2f}
- 有效性(effectiveness): {dimension_scores.get('effectiveness', 0):.2f}/10，置信度: {dimension_confidences.get('effectiveness', 0.7):.2f}

## 学习趋势分析
- 灵活性维度: {learning_trends.get('flexibility', {}).get('trend', '无趋势数据')}，分数变化: {str(learning_trends.get('flexibility', {}).get('scores', []))}
- 流畅性维度: {learning_trends.get('fluency', {}).get('trend', '无趋势数据')}，分数变化: {str(learning_trends.get('fluency', {}).get('scores', []))}
- 有效性维度: {learning_trends.get('effectiveness', {}).get('trend', '无趋势数据')}，分数变化: {str(learning_trends.get('effectiveness', {}).get('scores', []))}

## 成功/失败反馈记录
### 灵活性维度成功反馈:
{formatted_feedback.get('flexibility_success', '无成功反馈')}

### 灵活性维度失败反馈:
{formatted_feedback.get('flexibility_failure', '无失败反馈')}

### 流畅性维度成功反馈:
{formatted_feedback.get('fluency_success', '无成功反馈')}

### 流畅性维度失败反馈:
{formatted_feedback.get('fluency_failure', '无失败反馈')}

### 有效性维度成功反馈:
{formatted_feedback.get('effectiveness_success', '无成功反馈')}

### 有效性维度失败反馈:
{formatted_feedback.get('effectiveness_failure', '无失败反馈')}

## 分数调整历史
{formatted_adjustments}

## 对话历史
{conversation_summary}

请生成一份综合评估报告，包括以下部分：
1. 总体评价：学生在创新思维三个维度的总体表现及其优势劣势
2. 各维度详细分析：
   - 灵活性维度：学生表现、特点及学习进步情况
   - 流畅性维度：学生表现、特点及学习进步情况
   - 有效性维度：学生表现、特点及学习进步情况
3. 关键能力评估：基于对话历史和反馈记录，评估学生在以下方面的能力：
   - 多角度思考能力
   - 快速生成想法的能力
   - 提出可行解决方案的能力
   - 创新性思维的深度和广度
4. 学习发展趋势：根据各轮对话中分数的变化分析学生的能力发展趋势
5. 具体改进建议：针对学生表现的不足，提出3-5条具体可行的提升建议
6. 后续培养建议：为教师提供针对该学生的个性化教学或训练建议

请确保报告全面客观，同时具有针对性和建设性，可以帮助学生和教师明确优势和改进方向。必须根据整个对话过程中的所有信息进行分析，而不仅仅是最后一轮对话。"""

    # 调用LLM生成报告
    try:
        report_text = llm.query(report_prompt)
        return report_text
    except Exception as e:
        print(f"生成增强报告时出错: {str(e)}")
        # 提供一个简单的后备报告
        return f"""# 创新思维能力评估报告（自动生成）

## 基本信息
- 学生ID: {student_id}
- 评估时间: {timestamp}
- 总体得分: {overall_score:.2f}/10

## 各维度得分
- 灵活性(flexibility): {dimension_scores.get('flexibility', 0):.2f}/10
- 流畅性(fluency): {dimension_scores.get('fluency', 0):.2f}/10
- 有效性(effectiveness): {dimension_scores.get('effectiveness', 0):.2f}/10

## 报告生成失败
抱歉，由于技术原因无法生成完整报告。请联系系统管理员。
错误信息: {str(e)}
"""


def save_enhanced_report(evaluation_results, report_text):
    """
    保存增强版报告
    
    参数:
        evaluation_results: 原始评估结果
        report_text: 生成的报告文本
        
    返回:
        str: 保存的报告文件路径
    """
    try:
        student_id = evaluation_results.get('student_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 确保报告目录存在
        current_dir = os.path.dirname(os.path.abspath(__file__))
        report_dir = os.path.join(current_dir, "enhanced_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存文本报告
        txt_filename = f"enhanced_report_{student_id}_{timestamp}.txt"
        txt_filepath = os.path.join(report_dir, txt_filename)
        
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # 保存JSON报告数据
        json_filename = f"enhanced_report_{student_id}_{timestamp}.json"
        json_filepath = os.path.join(report_dir, json_filename)
        
        report_data = {
            "student_id": student_id,
            "timestamp": timestamp,
            "original_results": evaluation_results,
            "enhanced_report": report_text
        }
        
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        return txt_filepath
    except Exception as e:
        print(f"保存增强报告时出错: {str(e)}")
        return None