#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版评估报告生成程序
用于读取已生成的评估结果并生成基于完整对话历史的增强版评估报告
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

# 导入增强报告模块和语言模型
from enhanced_report import generate_enhanced_report, save_enhanced_report
from LLM import SimpleLLM  # 假设您有此类或使用其他LLM接口

def load_evaluation_results(filepath):
    """
    加载已有的评估结果文件
    
    参数:
        filepath: 评估结果JSON文件路径
        
    返回:
        dict: 评估结果字典
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载评估结果文件出错: {e}")
        return None

def list_available_reports(report_dir=None):
    """
    列出可用的评估报告文件
    
    参数:
        report_dir: 报告目录路径
        
    返回:
        list: 报告文件列表
    """
    if report_dir is None:
        report_dir = os.path.join(current_dir, "evaluation_reports")
    
    if not os.path.exists(report_dir):
        print(f"报告目录不存在: {report_dir}")
        return []
    
    reports = []
    for filename in os.listdir(report_dir):
        if filename.startswith("full_report_") and filename.endswith(".json"):
            reports.append(os.path.join(report_dir, filename))
    
    return reports

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成增强版评估报告")
    
    parser.add_argument("--report", "-r", type=str, help="评估结果文件路径")
    parser.add_argument("--latest", "-l", action="store_true", help="使用最新的评估结果")
    parser.add_argument("--student_id", "-s", type=str, help="学生ID，用于查找相关报告")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", help="使用的语言模型")
    parser.add_argument("--api_key", "-k", type=str, help="API密钥")
    
    return parser.parse_args()

def main():
    """主程序入口点"""
    args = parse_args()
    
    # 设置API密钥
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未设置API密钥，请通过--api_key参数或环境变量OPENAI_API_KEY设置")
    
    # 确定评估结果文件
    report_file = args.report
    
    if not report_file and args.latest:
        # 使用最新的评估结果
        reports = list_available_reports()
        if not reports:
            print("未找到可用的评估报告")
            return 1
        
        # 按修改时间排序
        reports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        report_file = reports[0]
        print(f"使用最新的评估报告: {report_file}")
    
    elif not report_file and args.student_id:
        # 根据学生ID查找报告
        reports = list_available_reports()
        matching_reports = [r for r in reports if f"full_report_{args.student_id}_" in os.path.basename(r)]
        
        if not matching_reports:
            print(f"未找到学生 {args.student_id} 的评估报告")
            return 1
            
        # 按修改时间排序
        matching_reports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        report_file = matching_reports[0]
        print(f"使用学生 {args.student_id} 最新的评估报告: {report_file}")
    
    if not report_file:
        print("请指定评估结果文件路径或使用--latest/-l参数选择最新的评估结果")
        return 1
    
    # 加载评估结果
    evaluation_results = load_evaluation_results(report_file)
    if not evaluation_results:
        print(f"无法加载评估结果: {report_file}")
        return 1
    
    # 从evaluator.py中导入LLM类
    try:
        from LLM import LLM
        llm = LLM(
            role_messages=[{"role": "system", "content": "你是一位专业的创新思维评估报告生成专家。"}],
            model=args.model
        )
    except Exception as e:
        print(f"初始化LLM失败: {e}")
        print("尝试替代方案...")
        
        try:
            from LLM import SimpleLLM
            llm = SimpleLLM(model_name=args.model)
        except Exception as e2:
            print(f"初始化SimpleLLM也失败: {e2}")
            return 1
    
    student_id = evaluation_results.get('student_id', 'unknown')
    print(f"\n开始为学生 {student_id} 生成增强版评估报告...")
    
    # 生成增强版报告
    enhanced_report_text = generate_enhanced_report(llm, evaluation_results)
    
    # 保存增强版报告
    report_path = save_enhanced_report(evaluation_results, enhanced_report_text)
    
    if report_path:
        print(f"\n增强版评估报告已保存至: {report_path}")
        return 0
    else:
        print("生成增强版评估报告失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())