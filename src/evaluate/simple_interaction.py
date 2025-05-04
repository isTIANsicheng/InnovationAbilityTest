import sys
import os
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creative_thinking import CreativeThinkingEvaluator
from creative_thinking.main import SamplingParameters

class InteractiveEvaluator:
    """交互式评估器，提供简单的接口与学生交互"""
    
    def __init__(self, student_id, model="gpt-4o", source="openai", 
                 memory_k=5, max_steps=10, stability_threshold=0.05):
        """初始化交互式评估器
        
        参数:
            student_id: 学生ID
            model: 使用的大语言模型
            source: 模型来源
            memory_k: 记忆保留的对话轮数
            max_steps: 每个维度的最大对话轮数
            stability_threshold: 分数稳定的阈值
        """
        self.student_id = student_id
        self.model = model
        self.source = source
        self.memory_k = memory_k
        self.max_steps = max_steps
        self.stability_threshold = stability_threshold
        
        # 初始化队列和事件
        self.question_queue = queue.Queue()
        self.answer_queue = queue.Queue()
        self.evaluation_complete = threading.Event()
        self.current_question = None
        self.has_next_question = threading.Event()
        
        # 初始化评估器
        self._initialize_evaluator()
        
        # 评估线程
        self.evaluation_thread = None
        self.summary_report = None
    
    def _initialize_evaluator(self):
        """初始化评估器"""
        # 创建采样参数
        sampling_params = SamplingParameters(
            max_tokens=1024,
            t=0.7,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # 检查问题库文件路径
        question_bank_path = "src/creative_thinking/question_bank.json"
        prompt_template_path = "src/creative_thinking/prompts.csv"
        
        # 初始化评估器
        self.evaluator = CreativeThinkingEvaluator(
            source=self.source,
            lm_id=self.model,
            prompt_template_path=prompt_template_path,
            question_bank_path=question_bank_path,
            sampling_parameters=sampling_params,
            K=self.memory_k,
            max_steps=self.max_steps,
            stability_threshold=self.stability_threshold
        )
    
    def _answer_provider(self, question):
        """提供回答的函数，由评估器调用"""
        # 保存当前问题并通知有新问题
        self.current_question = question
        self.has_next_question.set()
        self.question_queue.put(question)
        
        # 等待回答
        answer = self.answer_queue.get()
        
        # 重置事件
        self.has_next_question.clear()
        
        return answer
    
    def start_evaluation(self):
        """开始评估，在后台线程运行"""
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            return False
        
        # 重置状态
        self.evaluation_complete.clear()
        
        # 创建并启动评估线程
        self.evaluation_thread = threading.Thread(
            target=self._run_evaluation
        )
        self.evaluation_thread.daemon = True
        self.evaluation_thread.start()
        
        return True
    
    def _run_evaluation(self):
        """在线程中运行评估"""
        try:
            # 运行评估
            self.summary_report = self.evaluator.run_evaluation(
                self.student_id, self._answer_provider
            )
            
            # 标记评估完成
            self.evaluation_complete.set()
            
        except Exception as e:
            print(f"评估过程中发生错误: {str(e)}")
            # 确保即使出错也标记评估完成，避免死锁
            self.evaluation_complete.set()
    
    def get_next_question(self, timeout=None):
        """获取下一个问题
        
        参数:
            timeout: 等待超时时间（秒），None表示无限等待
            
        返回:
            问题文本，如果评估已完成则返回None
        """
        if self.evaluation_complete.is_set():
            return None
        
        try:
            return self.question_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def wait_for_question(self, timeout=None):
        """等待下一个问题
        
        参数:
            timeout: 等待超时时间（秒），None表示无限等待
            
        返回:
            是否有新问题
        """
        return self.has_next_question.wait(timeout=timeout)
    
    def provide_answer(self, answer):
        """提供学生的回答
        
        参数:
            answer: 学生的回答文本
            
        返回:
            是否成功提交回答
        """
        if self.evaluation_complete.is_set():
            return False
        
        self.answer_queue.put(answer)
        return True
    
    def is_evaluation_complete(self):
        """检查评估是否已完成"""
        return self.evaluation_complete.is_set()
    
    def get_current_question(self):
        """获取当前问题"""
        return self.current_question
    
    def get_summary_report(self):
        """获取评估报告
        
        返回:
            评估报告，如果评估未完成则返回None
        """
        if not self.evaluation_complete.is_set():
            return None
        
        return self.summary_report

# 使用示例
if __name__ == "__main__":
    # 创建交互式评估器
    evaluator = InteractiveEvaluator("test_student", model="gpt-4o")
    
    # 开始评估
    evaluator.start_evaluation()
    
    # 简单的命令行交互循环
    print("评估开始，请回答问题...")
    
    while not evaluator.is_evaluation_complete():
        # 等待下一个问题
        if evaluator.wait_for_question(timeout=1):
            # 获取当前问题
            question = evaluator.get_current_question()
            print(f"\n问题: {question}")
            
            # 获取学生回答
            answer = input("请输入你的回答: ")
            
            # 提供回答
            evaluator.provide_answer(answer)
        
    # 获取评估报告
    report = evaluator.get_summary_report()
    
    if report:
        print("\n==== 评估结果摘要 ====")
        
        # 打印维度得分
        print("维度得分:")
        for dimension, score in report['dimension_scores'].items():
            print(f"  - {dimension}: {score:.1f}/10")
        
        # 打印学习趋势
        print("\n学习趋势:")
        for dimension, trend in report['learning_trends'].items():
            print(f"  - {dimension}: {trend['description']}")
    else:
        print("评估过程中发生错误，无法获取报告") 