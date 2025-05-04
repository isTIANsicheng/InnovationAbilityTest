import sys
import os
import json
import gradio as gr
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creative_thinking import CreativeThinkingEvaluator
from creative_thinking.main import SamplingParameters, parse_args

# 创建一个队列用于存储学生的回答
answer_queue = queue.Queue()
next_question = queue.Queue()
current_question = ""
evaluation_complete = threading.Event()

def answer_provider(question):
    """提供学生回答的函数
    这个函数会将问题放入队列，然后等待学生的回答
    """
    global current_question
    current_question = question
    next_question.put(question)
    # 等待学生的回答
    answer = answer_queue.get()
    return answer

def start_evaluation(student_id, model, source="openai", memory_k=5, max_steps=10):
    """启动评估线程"""
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
    evaluator = CreativeThinkingEvaluator(
        source=source,
        lm_id=model,
        prompt_template_path=prompt_template_path,
        question_bank_path=question_bank_path,
        sampling_parameters=sampling_params,
        K=memory_k,
        max_steps=max_steps,
        stability_threshold=0.05
    )
    
    # 运行评估
    summary_report = evaluator.run_evaluation(student_id, answer_provider)
    
    # 标记评估完成
    evaluation_complete.set()
    
    # 返回评估结果
    return summary_report

def submit_answer(answer):
    """提交学生的回答"""
    if evaluation_complete.is_set():
        return "评估已完成", ""
    
    # 将回答放入队列
    answer_queue.put(answer)
    
    # 等待下一个问题（如果有）
    try:
        question = next_question.get(timeout=1)
        return "回答已提交", question
    except queue.Empty:
        return "正在处理...", current_question

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="创新思维评估系统") as app:
        gr.Markdown("# 创新思维评估系统")
        
        with gr.Row():
            with gr.Column():
                student_id = gr.Textbox(label="学生ID", placeholder="请输入学生ID")
                model = gr.Dropdown(["gpt-4o", "gpt-4", "gpt-3.5-turbo"], label="评估模型", value="gpt-4o")
                memory_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="记忆轮数")
                max_steps = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="最大对话轮数")
                start_btn = gr.Button("开始评估")
            
            with gr.Column():
                question_display = gr.Textbox(label="当前问题", interactive=False)
                answer_input = gr.Textbox(label="你的回答", placeholder="请输入你的回答", lines=5)
                submit_btn = gr.Button("提交回答")
                status = gr.Textbox(label="状态", interactive=False)
        
        def on_start(student_id, model, memory_k, max_steps):
            if not student_id:
                return "请输入学生ID", ""
            
            # 重置状态
            answer_queue.queue.clear()
            next_question.queue.clear()
            evaluation_complete.clear()
            
            # 启动评估线程
            thread = threading.Thread(
                target=start_evaluation, 
                args=(student_id, model, "openai", int(memory_k), int(max_steps))
            )
            thread.daemon = True
            thread.start()
            
            # 等待第一个问题
            try:
                first_question = next_question.get(timeout=10)
                return "评估已开始", first_question
            except queue.Empty:
                return "等待问题超时，请检查连接", ""
        
        start_btn.click(
            on_start,
            inputs=[student_id, model, memory_k, max_steps],
            outputs=[status, question_display]
        )
        
        submit_btn.click(
            submit_answer,
            inputs=[answer_input],
            outputs=[status, question_display]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) 