import os
import json
import openai
import time
import re

class SimpleLLM:
    """
    简化版的语言模型接口，使用OpenAI API
    提供更强大的错误处理和消息格式化功能
    """
    
    def __init__(self, model_name="gpt-4o", api_key=None, temperature=0.7, max_tokens=1000):
        """
        初始化语言模型接口
        
        参数:
            model_name (str): 模型名称，如"gpt-4o"
            api_key (str): OpenAI API密钥，默认使用环境变量
            temperature (float): 生成温度，控制随机性
            max_tokens (int): 最大生成令牌数
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 设置API密钥
        if api_key:
            openai.api_key = api_key
        
        # 系统提示
        self.system_prompt = "你是一位创新思维评估助手，专注于帮助评估学生的创新思维能力。"
        
        # 消息历史
        self.history = []
        
    def set_system_prompt(self, prompt):
        """
        设置系统提示
        
        参数:
            prompt (str): 系统提示文本
        """
        if prompt and isinstance(prompt, str):
            self.system_prompt = prompt
        else:
            print("警告: 无效的系统提示，保持原有提示不变")
    
    def query(self, user_content):
        """
        向语言模型发送查询并获取回复
        
        参数:
            user_content (str): 用户输入内容
            
        返回:
            str: 模型回复
        """
        # 确保用户内容非空
        if not user_content or not isinstance(user_content, str) or user_content.strip() == "":
            print("警告: 用户内容为空，使用默认提示")
            user_content = "请继续"
        
        # 创建消息数组
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加历史消息（最多10轮）
        for msg in self.history[-10:]:
            messages.append(msg)
        
        # 添加当前用户消息
        user_message = {"role": "user", "content": user_content}
        messages.append(user_message)
        
        # 确保所有消息格式正确
        validated_messages = []
        for msg in messages:
            # 确保每条消息都有有效的角色和内容
            if not isinstance(msg, dict):
                print(f"警告: 跳过无效消息格式: {msg}")
                continue
                
            # 确保角色有效
            if "role" not in msg or msg["role"] not in ["system", "user", "assistant"]:
                print(f"警告: 消息角色无效，设置为'user': {msg}")
                msg["role"] = "user"
                
            # 确保内容有效
            if "content" not in msg or msg["content"] is None:
                print(f"警告: 消息内容为空，设置为空字符串: {msg}")
                msg["content"] = ""
                
            validated_messages.append(msg)
        
        # 确保至少有一条有效消息
        if not validated_messages:
            validated_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "请提供一些思考"}
            ]
        
        try:
            # 使用旧版本的OpenAI API调用方式
            try:
                # 调用API
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=validated_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # 提取回复
                assistant_response = response.choices[0].message.content
            except AttributeError:
                # 可能是API格式不同
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=validated_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # 尝试不同格式提取回复
                assistant_response = response["choices"][0]["message"]["content"]
            
            # 添加到历史
            self.history.append(user_message)
            self.history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            print(f"API调用错误: {e}")
            error_response = f"抱歉，无法完成请求。错误信息: {str(e)}"
            
            # 添加错误响应到历史
            self.history.append(user_message)
            self.history.append({"role": "assistant", "content": error_response})
            
            return error_response
    
    def clear_history(self):
        """清除对话历史"""
        self.history = []
        
    def extract_score(self, text):
        """
        从文本中提取分数
        
        参数:
            text (str): 包含分数的文本
            
        返回:
            float: 提取的分数，如果未找到则返回5.0
        """
        try:
            # 尝试查找分数模式
            score_patterns = [
                r'分数[:：]\s*(\d+(?:\.\d+)?)',
                r'得分[:：]\s*(\d+(?:\.\d+)?)',
                r'评分[:：]\s*(\d+(?:\.\d+)?)',
                r'score[:：]\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*[/／]\s*10',
                r'(\d+(?:\.\d+)?)\s*分'
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # 确保分数在0-10范围内
                    return max(0, min(10, score))
            
            # 如果没有找到数字分数，尝试查找分数单词
            if re.search(r'非常差|极差|terrible', text, re.IGNORECASE):
                return 1.0
            elif re.search(r'差|poor', text, re.IGNORECASE):
                return 3.0
            elif re.search(r'一般|中等|average', text, re.IGNORECASE):
                return 5.0
            elif re.search(r'良好|good', text, re.IGNORECASE):
                return 7.0
            elif re.search(r'优秀|excellent', text, re.IGNORECASE):
                return 9.0
                
            # 默认返回
            return 5.0
        except Exception as e:
            print(f"提取分数时出错: {e}")
            return 5.0 