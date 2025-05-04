import sys
import os
import json
import argparse
import queue
import threading
from colorama import init, Fore, Style

# 初始化colorama
init()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creative_thinking import CreativeThinkingEvaluator
from creative_thinking.main import SamplingParameters

# 创建一个队列用于存储学生的回答
answer_queue = queue.Queue()
evaluation_ready = threading.Event()

def answer_provider(question):
    """提供学生回答的函数
    打印问题并等待用户输入
    """
    print(f"\n{Fore.GREEN}问题: {question}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}请输入你的回答 (按Enter提交):{Style.RESET_ALL}")
    answer = input()
    return answer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="创新思维评估系统 - 交互式命令行界面")
    
    parser.add_argument("--model", type=str, default="gpt-4o", help="使用的大语言模型")
    parser.add_argument("--source", type=str, default="openai", choices=["openai", "huggingface", "debug"], help="模型来源")
    parser.add_argument("--prompt_template_path", type=str, default="src/creative_thinking/prompts.csv", help="提示模板路径")
    parser.add_argument("--question_bank_path", type=str, default="src/creative_thinking/question_bank.json", help="题库路径")
    parser.add_argument("--student_id", type=str, required=True, help="学生ID")
    parser.add_argument("--memory_k", type=int, default=5, help="记忆保留的对话轮数")
    parser.add_argument("--max_steps", type=int, default=10, help="每个维度的最大对话轮数")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print(f"\n{Fore.CYAN}===== 创新思维评估系统 - 交互式命令行界面 ====={Style.RESET_ALL}")
    print(f"{Fore.CYAN}学生ID: {args.student_id}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}使用模型: {args.model}{Style.RESET_ALL}")
    
    # 检查问题库文件
    question_bank_exists = os.path.exists(args.question_bank_path)
    
    # 如果统一问题库不存在，检查分离问题文件
    if not question_bank_exists:
        dir_path = os.path.dirname(args.question_bank_path)
        flex_path = os.path.join(dir_path, 'question_flex.json')
        flu_path = os.path.join(dir_path, 'question_flu.json')
        eff_path = os.path.join(dir_path, 'question_eff.json')
        
        # 检查是否至少有一个问题文件存在
        separate_files_exist = any(os.path.exists(p) for p in [flex_path, flu_path, eff_path])
        
        if not separate_files_exist:
            print(f"{Fore.RED}错误: 既没有找到统一问题库 {args.question_bank_path}，也没有找到分离问题文件{Style.RESET_ALL}")
            print(f"{Fore.RED}请确保以下文件至少有一个存在:{Style.RESET_ALL}")
            print(f"  - {args.question_bank_path}")
            print(f"  - {flex_path}")
            print(f"  - {flu_path}")
            print(f"  - {eff_path}")
            return
    
    # 确保提示模板路径存在
    if not os.path.exists(args.prompt_template_path):
        print(f"{Fore.YELLOW}警告: 文件 {args.prompt_template_path} 不存在，将尝试使用单独的提示文件{Style.RESET_ALL}")
    
    # 创建采样参数
    sampling_params = SamplingParameters(
        max_tokens=1024,
        t=0.7,
        top_p=0.95,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    print(f"\n{Fore.YELLOW}正在初始化评估器...{Style.RESET_ALL}")
    
    try:
        # 初始化评估器
        evaluator = CreativeThinkingEvaluator(
            source=args.source,
            lm_id=args.model,
            prompt_template_path=args.prompt_template_path,
            question_bank_path=args.question_bank_path,
            sampling_parameters=sampling_params,
            K=args.memory_k,
            max_steps=args.max_steps,
            stability_threshold=0.05
        )
        
        print(f"{Fore.GREEN}评估器初始化成功！{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}开始评估学生 {args.student_id} 的创新思维能力...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}请根据提示输入你的回答{Style.RESET_ALL}")
        
        # 运行评估
        summary_report = evaluator.run_evaluation(args.student_id, answer_provider)
        
        # 打印评估结果
        print(f"\n{Fore.CYAN}==== 评估结果摘要 ===={Style.RESET_ALL}")
        print(f"{Fore.CYAN}学生ID: {args.student_id}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}维度得分:{Style.RESET_ALL}")
        
        for dimension, score in summary_report['dimension_scores'].items():
            print(f"  - {dimension}: {score:.1f}/10")
        
        # 打印学习趋势
        print(f"\n{Fore.CYAN}学习趋势:{Style.RESET_ALL}")
        for dimension, trend in summary_report['learning_trends'].items():
            print(f"  - {dimension}: {trend['description']}")
        
        # 保存报告到JSON文件
        output_file = f"report_{args.student_id}_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{Fore.GREEN}报告已保存到: {output_file}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}评估过程中发生错误: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import time
    main() 