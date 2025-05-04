import argparse
import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creative_thinking.SimpleLLM import SimpleLLM
from creative_thinking.evaluator import Evaluator


@dataclass
class SamplingParameters:
    """采样参数配置"""
    max_tokens: int = 1024
    t: float = 0.7
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    debug: bool = False
    task_name: str = "CreativeThinking"


def parse_args():
    parser = argparse.ArgumentParser(description="创新思维评估系统")
    
    parser.add_argument("--model", type=str, default="gpt-4o", help="使用的大语言模型")
    parser.add_argument("--prompt_template_path", type=str, default="prompts.csv", help="提示模板路径")
    parser.add_argument("--question_bank_path", type=str, default="question_flex.json", help="题库路径")
    parser.add_argument("--student_id", type=str, required=True, help="学生ID")
    parser.add_argument("--answers_file", type=str, help="预先准备好的学生回答JSON文件路径")
    parser.add_argument("--interactive", action="store_true", help="使用交互式命令行输入回答")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API Key")
    parser.add_argument("--memory_k", type=int, default=3, help="对话历史保留的轮数")
    parser.add_argument("--max_steps", type=int, default=5, help="每个维度最大对话步数")
    parser.add_argument("--use_dual_model", action="store_true", help="启用双模型评分系统")
    parser.add_argument("--second_model", type=str, default="gpt-3.5-turbo", help="第二个模型名称")
    parser.add_argument("--confidence_threshold", type=float, default=1.5, help="模型评分一致性阈值")
    parser.add_argument("--low_confidence_threshold", type=float, default=3.0, help="低可信度阈值")
    
    return parser.parse_args()


def main():
    """主程序入口"""
    args = parse_args()
    
    # 当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建文件路径
    prompt_template_path = os.path.join(current_dir, args.prompt_template_path)
    question_bank_path = os.path.join(current_dir, args.question_bank_path)
    
    # 确保提示模板路径存在
    if not os.path.exists(prompt_template_path):
        print(f"警告: 文件 {prompt_template_path} 不存在，将尝试使用单独的提示文件")
    
    # 检查其他提示模板文件
    template_dir = os.path.dirname(prompt_template_path)
    template_files = [
        'scoring_prompts.csv',
        'scoring_reference_prompts.csv',
        'followup_prompts.csv',
        'report_prompts.csv'
    ]
    
    missing_templates = []
    for template_file in template_files:
        template_path = os.path.join(template_dir, template_file)
        if not os.path.exists(template_path):
            missing_templates.append(template_file)
    
    if missing_templates:
        print(f"警告: 以下提示模板文件不存在，将使用默认提示:")
        for missing in missing_templates:
            print(f"  - {missing}")
    
    # 检查问题库文件
    question_bank_exists = os.path.exists(question_bank_path)
    if not question_bank_exists:
        print(f"警告: 问题库文件 {question_bank_path} 不存在，将使用默认问题")
    
    # 如果指定了答案文件，检查其是否存在
    answer_provider = None
    if args.answers_file:
        answers_file_path = os.path.join(current_dir, args.answers_file)
        if not os.path.exists(answers_file_path):
            print(f"错误: 答案文件 {answers_file_path} 不存在")
            return
        
        try:
            with open(answers_file_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            # 创建答案提供函数
            def get_answer(question):
                # 尝试在答案中查找匹配的问题
                for q_pattern, answer in answers_data.items():
                    if q_pattern.lower() in question.lower() or question.lower() in q_pattern.lower():
                        return answer
                
                # 如果没有找到精确匹配，则返回默认答案或者提示用户手动输入
                if args.interactive:
                    print(f"在答案文件中未找到匹配的问题: {question}")
                    return input("请输入回答: ")
                else:
                    # 使用默认回答
                    print(f"警告: 在答案文件中未找到匹配的问题: {question}")
                    print(f"使用默认回答")
                    return "这是一个默认回答，因为在答案文件中没有找到匹配的问题。"
            
            answer_provider = get_answer
            print(f"已加载答案文件: {answers_file_path}")
            
        except Exception as e:
            print(f"加载答案文件失败: {e}")
            if args.interactive:
                print("将使用交互式输入")
            else:
                return
    
    # 初始化语言模型
    try:
        # 设置API密钥，优先使用命令行参数
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("警告: 未设置OpenAI API密钥，请通过--openai_api_key参数或环境变量OPENAI_API_KEY设置")
            return

        # 确保设置全局API密钥
        import openai
        openai.api_key = api_key
        
        # 初始化主模型
        llm = SimpleLLM(
            model_name=args.model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=1024
        )
        
        # 初始化第二个模型（如果启用双模型评分）
        second_model = None
        if args.use_dual_model:
            print(f"启用双模型评分系统: 主模型={args.model}, 第二模型={args.second_model}")
            second_model = SimpleLLM(
                model_name=args.second_model,
                api_key=api_key,
                temperature=0.7,
                max_tokens=1024
            )
    except Exception as e:
        print(f"初始化语言模型失败: {e}")
        return
    
    # 初始化评估器
    try:
        evaluator = Evaluator(
            llm=llm,
            prompt_template_path=prompt_template_path,
            question_bank_path=question_bank_path,
            K=args.memory_k,  # 传递memory_k参数
            second_model=second_model,  # 传递第二个模型
            second_model_name=args.second_model if args.use_dual_model else None
        )
        
        # 如果启用了双模型评分，设置阈值参数
        if args.use_dual_model:
            evaluator.model_agreement_threshold = args.confidence_threshold
            evaluator.low_confidence_threshold = args.low_confidence_threshold
            print(f"设置模型一致性阈值: {evaluator.model_agreement_threshold}")
            print(f"设置低可信度阈值: {evaluator.low_confidence_threshold}")
    except Exception as e:
        print(f"初始化评估器失败: {e}")
        return
    
    # 运行评估
    try:
        print(f"\n开始对学生 {args.student_id} 进行创新思维评估")
        start_time = time.time()
        
        evaluation_results = evaluator.run_evaluation(args.student_id, answer_provider, max_steps=args.max_steps)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 打印评估结果
        print("\n==== 评估结果摘要 ====")
        print(f"学生ID: {args.student_id}")
        print(f"评估耗时: {duration:.1f}秒")
        print(f"总分: {evaluation_results['overall_score']:.1f}/10")
        
        print("\n维度得分:")
        for dimension, result in evaluation_results['dimensions'].items():
            score = result.get('score', 0)
            print(f"  - {dimension}: {score:.1f}/10")
        
        # 展示报告摘要
        print("\n报告摘要:")
        report_lines = evaluation_results.get('report', '').split('\n')
        for line in report_lines[:5]:
            print(f"  {line}")
        if len(report_lines) > 5:
            print("  ...")
        
        print(f"\n完整评估报告已保存至: {evaluation_results.get('report_path', '未知')}")
        
        return evaluation_results
    except Exception as e:
        print(f"评估过程中出错: {e}")
        return None


if __name__ == "__main__":
    main() 