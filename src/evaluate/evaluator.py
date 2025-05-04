import random
import json
import os
import pandas as pd
import torch
from collections import defaultdict
import time
import backoff
import re
from datetime import datetime
from .confidence_estimator import ConfidenceEstimator
from .LLM import LLM



class DynamicScoreAdjuster:
    """
    根据历史表现和当前评估结果动态更新学生能力评分
    """
    def __init__(
        self,
        initial_weight=0.7,     
        min_weight=0.3,         
        learning_rate=0.05,     
        stability_factor=0.8,   
        consistency_threshold=1.0, 
        adjustment_cap=2.0      
    ):
        self.history_weight = initial_weight  
        self.min_weight = min_weight
        self.learning_rate = learning_rate
        self.stability_factor = stability_factor
        self.consistency_threshold = consistency_threshold
        self.adjustment_cap = adjustment_cap
        
        
        self.dimension_history = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        
        self.score_differences = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        
        self.evaluation_standards = {
            'flexibility': {'base_standard': 1.0, 'adjustments': []},
            'fluency': {'base_standard': 1.0, 'adjustments': []},
            'effectiveness': {'base_standard': 1.0, 'adjustments': []}
        }
        
        
        self.learning_slopes = {
            'flexibility': 0.0,
            'fluency': 0.0,
            'effectiveness': 0.0
        }
    
    def add_score(self, dimension, new_score):
        """添加新的分数到历史记录"""
        try:
            # 确保new_score是浮点数
            new_score = float(new_score)
            
            if dimension not in self.dimension_history:
                self.dimension_history[dimension] = []
                
            self.dimension_history[dimension].append(new_score)
            
            if len(self.dimension_history[dimension]) >= 3:
                recent_scores = self.dimension_history[dimension][-3:]
                x = [0, 1, 2]
                y = recent_scores
                slope = self._calculate_slope(x, y)
                self.learning_slopes[dimension] = slope
        except (ValueError, TypeError) as e:
            print(f"添加分数时出错: {e}，忽略此分数: {new_score}")
    
    def get_adjusted_score(self, dimension, current_score, model_confidence):
        """
        根据历史表现和当前评估结果动态调整分数
        
        参数:
            dimension: 评估维度
            current_score: 当前评估分数
            model_confidence: 模型可信度 (0-1之间的浮点数)
            
        返回:
            tuple: (调整后的分数, 调整信息)
        """
        try:
            # 确保输入数据类型正确
            current_score = float(current_score)
            
            # 处理model_confidence，确保是0-1之间的浮点数
            if isinstance(model_confidence, dict) and 'score' in model_confidence:
                confidence = float(model_confidence['score'])
            else:
                try:
                    confidence = float(model_confidence)
                except (ValueError, TypeError):
                    print(f"警告: 置信度值无法转换为浮点数: {model_confidence}，使用默认值0.7")
                    confidence = 0.7
                    
            # 确保置信度在0-1之间
            if confidence > 1.0:
                if confidence <= 10.0:  # 可能是按10分制
                    confidence = confidence / 10.0
                else:
                    confidence = 0.7  # 使用默认值
            
            confidence = max(0.1, min(1.0, confidence))  # 确保在0.1-1.0之间
                
            # 获取历史数据
            history = self.dimension_history.get(dimension, [])
            
            if not history:
                return current_score, {
                    "reason": "首次评估，无历史数据", 
                    "adjustment": 0.0,
                    "confidence": confidence
                }
                
            # 计算历史平均分
            history_avg = sum(history) / len(history)
            score_diff = current_score - history_avg
            
            # 记录分数差异
            if dimension not in self.score_differences:
                self.score_differences[dimension] = []
                
            self.score_differences[dimension].append(score_diff)
            
            # 初始权重分配
            current_weight = 1.0 - self.history_weight
            
            # 根据历史稳定性调整权重
            if len(history) >= 3:
                std_dev = self._calculate_std_dev(history[-3:])
                stability = max(0.2, min(0.9, 1.0 - std_dev / 5.0)) 
                adjusted_history_weight = self.history_weight * self.stability_factor * stability
                adjusted_history_weight = max(self.min_weight, adjusted_history_weight)
                current_weight = 1.0 - adjusted_history_weight

            # 使用模型可信度调整权重 - 可信度越高，当前分数权重越大
            current_weight = current_weight * confidence

            # 归一化权重
            total_weight = current_weight + self.history_weight
            norm_current_weight = current_weight / total_weight
            norm_history_weight = self.history_weight / total_weight
            
            # 计算加权分数
            weighted_score = (current_score * norm_current_weight) + (history_avg * norm_history_weight)

            # 学习趋势调整
            learning_adjustment = 0.0
            if self.learning_slopes[dimension] > 0.2:  
                learning_adjustment = min(0.5, self.learning_slopes[dimension])
            elif self.learning_slopes[dimension] < -0.2: 
                learning_adjustment = max(-0.5, self.learning_slopes[dimension])

            # 最终调整分数
            adjusted_score = weighted_score + learning_adjustment

            # 限制调整幅度
            max_adjustment = self.adjustment_cap
            if abs(adjusted_score - current_score) > max_adjustment:
                direction = 1 if adjusted_score > current_score else -1
                adjusted_score = current_score + (direction * max_adjustment)

            # 确保分数在0-10范围内
            adjusted_score = max(0, min(10, adjusted_score))

            # 记录调整信息
            adjustment_info = {
                "original_score": float(current_score),
                "history_avg": float(history_avg),
                "weighted_score": float(weighted_score),
                "learning_adjustment": float(learning_adjustment),
                "final_adjustment": float(adjusted_score - current_score),
                "history_weight": float(norm_history_weight),
                "current_weight": float(norm_current_weight),
                "learning_slope": float(self.learning_slopes[dimension]),
                "applied_confidence": float(confidence),  # 添加实际使用的置信度
                "reason": self._generate_adjustment_reason(dimension, current_score, adjusted_score, 
                                                      learning_adjustment, norm_history_weight)
            }
            
            # 更新评估标准
            self._update_evaluation_standard(dimension, current_score, adjusted_score)
            
            return float(adjusted_score), adjustment_info
            
        except Exception as e:
            print(f"调整分数时出错: {e}")
            import traceback
            print(traceback.format_exc())
            # 出错时返回原始分数
            return float(current_score), {
                "reason": f"调整出错: {str(e)}", 
                "adjustment": 0.0,
                "confidence": 0.7  # 默认置信度
            }
    
    def _calculate_std_dev(self, values):
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_slope(self, x, y):
        n = len(x)
        if n != len(y) or n < 2:
            return 0.0
            
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def _update_evaluation_standard(self, dimension, current_score, adjusted_score):
        adjustment = adjusted_score - current_score
        self.evaluation_standards[dimension]['adjustments'].append(adjustment)
        
        if len(self.evaluation_standards[dimension]['adjustments']) >= 3:
            recent_adjustments = self.evaluation_standards[dimension]['adjustments'][-3:]

            if all(adj > 0.2 for adj in recent_adjustments):
                self.evaluation_standards[dimension]['base_standard'] *= (1.0 - self.learning_rate)
            elif all(adj < -0.2 for adj in recent_adjustments):
                self.evaluation_standards[dimension]['base_standard'] *= (1.0 + self.learning_rate)
    
    def _generate_adjustment_reason(self, dimension, current_score, adjusted_score, 
                                   learning_adjustment, history_weight):
        adjustment = adjusted_score - current_score
        
        if abs(adjustment) < 0.1:
            return "当前评估与历史表现高度一致，保持原分数"
        
        reason_parts = []
        
        if abs(learning_adjustment) > 0.1:
            if learning_adjustment > 0:
                reason_parts.append(f"检测到学生在{dimension}维度有持续进步趋势")
            else:
                reason_parts.append(f"检测到学生在{dimension}维度有下滑趋势")
        
        if history_weight > 0.6:
            if adjustment > 0:
                reason_parts.append("历史表现优于当前评估")
            else:
                reason_parts.append("历史表现弱于当前评估")
        else:
            if adjustment > 0:
                reason_parts.append("当前表现优于历史平均水平")
            else:
                reason_parts.append("当前表现弱于历史平均水平")
        
        if not reason_parts:
            reason_parts = ["综合历史和当前表现的权衡结果"]
        
        return "，".join(reason_parts)
    
    def get_evaluation_standard_factor(self, dimension):
        return self.evaluation_standards[dimension]['base_standard']
    
    def get_learning_trend(self, dimension):
        slope = self.learning_slopes[dimension]
        
        if slope > 0.5:
            trend = "显著提升"
        elif slope > 0.2:
            trend = "稳步提升"
        elif slope > -0.2:
            trend = "稳定"
        elif slope > -0.5:
            trend = "略有下降"
        else:
            trend = "明显下降"
        
        return {
            "slope": slope,
            "trend": trend,
            "description": f"学生在{dimension}维度的学习曲线呈{trend}趋势"
        }
    
    def update_weights(self, dimension, model_agreement):
        """
        根据模型一致性动态更新权重
        
        当模型评估结果一致性高时，减少历史权重，更信任当前评估
        当模型评估结果一致性低时，增加历史权重，更信任历史表现
        """
        if model_agreement < self.consistency_threshold / 2:
            self.history_weight = max(
                self.min_weight,
                self.history_weight - self.learning_rate
            )
        elif model_agreement > self.consistency_threshold:
            self.history_weight = min(
                0.9,
                self.history_weight + self.learning_rate
            )

class CreativeThinkingEvaluator:
    """
    基于大模型的学生创新思维自动评价系统
    评估三个维度：灵活性(flexibility)、流畅性(fluency)、有效性(effectiveness)
    使用多模型评分提高可信度
    """
    def __init__(self, llm, prompt_template_path=None, question_bank_path=None, K=3, 
               second_model=None, second_model_name=None):
        """
        初始化评估器
        参数:
            llm: 主要语言模型接口
            prompt_template_path: 提示模板文件路径
            question_bank_path: 问题库文件路径
            K: 对话历史保留的轮数
            second_model: 第二个语言模型接口，用于双模型评分对比
            second_model_name: 第二个模型名称
        """
        self.llm = llm
        
        self.use_dual_model = second_model is not None
        self.second_model = second_model
        self.second_model_name = second_model_name or "SecondaryModel"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.prompt_template_path = prompt_template_path or os.path.join(current_dir, "prompts.csv")
        
        self.question_bank_path = question_bank_path or os.path.join(current_dir, "question_flex.json")
        
        self.dimensions = ['flexibility', 'fluency', 'effectiveness']
        
        self.dialog_history_by_dimension = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        if not os.path.exists(self.prompt_template_path):
            print(f"警告: 提示模板文件 {self.prompt_template_path} 不存在")
            
        if not os.path.exists(self.question_bank_path):
            print(f"警告: 问题库文件 {self.question_bank_path} 不存在")
            
        self.question_bank = self.load_question_bank()
        
        os.makedirs(os.path.join(current_dir, "evaluation_results"), exist_ok=True)
        os.makedirs(os.path.join(current_dir, "evaluation_reports"), exist_ok=True)
        
        if 'datetime' not in globals():
            global datetime
            from datetime import datetime
            
        if 're' not in globals():
            global re
            import re
            
        if 'random' not in globals():
            global random
            import random
        
        self.K = K  # 对话历史保留的轮数
        self.max_steps = 10  # 默认最大步数
        
        self.model_agreement_threshold = 1.5  # 模型评分一致性阈值
        self.low_confidence_threshold = 3.0   # 低可信度阈值
        
        self.success_threshold = 1.0
        self.failure_threshold = 1.0
        self.stability_window = 3
        self.stability_tolerance = 0.5
        self.feedback_influence = 0.5
        
        self.feedback_records = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.dimension_states = {
            'flexibility': 'pending',  # 待测试
            'fluency': 'pending',
            'effectiveness': 'pending'
        }
        
        self.score_changes = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.current_dimension = None
        
        self.needs_guidance = {
            'flexibility': False,
            'fluency': False,
            'effectiveness': False
        }
        
        try:
            if os.path.exists(self.prompt_template_path):
                df = pd.read_csv(self.prompt_template_path)
                self.prompt_templates = {
                    'flexibility': df['flexibility_prompt'][0] if 'flexibility_prompt' in df.columns else None,
                    'fluency': df['fluency_prompt'][0] if 'fluency_prompt' in df.columns else None,
                    'effectiveness': df['effectiveness_prompt'][0] if 'effectiveness_prompt' in df.columns else None,
                    'summary': df['summary_prompt'][0] if 'summary_prompt' in df.columns else None
                }
            else:
                print(f"未找到提示模板文件 {self.prompt_template_path}，尝试使用单独的提示文件")
                self.prompt_templates = {}
            
            template_dir = os.path.dirname(self.prompt_template_path)
            
            for dimension in self.dimensions:
                dimension_file = os.path.join(template_dir, f"{dimension}_prompt.csv")
                if os.path.exists(dimension_file):
                    try:
                        dimension_df = pd.read_csv(dimension_file)
                        dimension_key = f"{dimension}_prompt"
                        
                        if dimension_key in dimension_df.columns:
                            self.prompt_templates[dimension] = dimension_df[dimension_key][0]
                            print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 {dimension_key} 列)")
                        elif dimension in dimension_df.columns:
                            self.prompt_templates[dimension] = dimension_df[dimension][0]
                            print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 {dimension} 列)")
                        elif 'prompt' in dimension_df.columns:
                            if 'dimension' in dimension_df.columns:
                                matching_rows = dimension_df[dimension_df['dimension'] == dimension]
                                if not matching_rows.empty:
                                    self.prompt_templates[dimension] = matching_rows['prompt'].iloc[0]
                                    print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 dimension-prompt 对)")
                                else:
                                    self.prompt_templates[dimension] = dimension_df['prompt'].iloc[0]
                                    print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 prompt 列第一行)")
                            else:
                                self.prompt_templates[dimension] = dimension_df['prompt'].iloc[0]
                                print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 prompt 列)")
                        elif 'content' in dimension_df.columns:
                            if 'name' in dimension_df.columns:
                                matching_rows = dimension_df[dimension_df['name'] == f"{dimension}_prompt"]
                                if not matching_rows.empty:
                                    self.prompt_templates[dimension] = matching_rows['content'].iloc[0]
                                    print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 name-content 对)")
                                else:
                                    self.prompt_templates[dimension] = dimension_df['content'].iloc[0]
                                    print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 content 列第一行)")
                            else:
                                self.prompt_templates[dimension] = dimension_df['content'].iloc[0]
                                print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用 content 列)")
                        else:
                            first_col = dimension_df.columns[0]
                            if first_col != 'Unnamed: 0':
                                self.prompt_templates[dimension] = dimension_df[first_col].iloc[0]
                                print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用第一列 {first_col})")
                            elif len(dimension_df.columns) > 1:
                                second_col = dimension_df.columns[1]
                                self.prompt_templates[dimension] = dimension_df[second_col].iloc[0]
                                print(f"已加载 {dimension} 维度提示模板: {dimension_file} (使用第二列 {second_col})")
                            else:
                                raise ValueError(f"在文件 {dimension_file} 中未找到可用的提示模板列")
                    except Exception as e:
                        print(f"加载 {dimension} 维度提示模板时出错: {str(e)}")
                        if 'dimension_df' in locals():
                            print(f"CSV列名: {list(dimension_df.columns)}")
                            print(f"CSV行数: {len(dimension_df)}")
                            if len(dimension_df) > 0:
                                print(f"第一行数据: {dict(dimension_df.iloc[0])}")
                        import traceback
                        print(traceback.format_exc())
            
            for dimension in self.dimensions:
                if dimension not in self.prompt_templates or self.prompt_templates[dimension] is None:
                    print(f"警告: 未找到 {dimension} 维度的提示模板，使用默认提示")
                    default_prompts = {
                        'flexibility': "你是一位专业的创新思维评估助手，专注于评估学生的创新思维中的灵活性维度。灵活性指的是从不同角度思考问题，生成多样化解决方案的能力。",
                        'fluency': "你是一位专业的创新思维评估助手，专注于评估学生的创新思维中的流畅性维度。流畅性指的是快速、大量产生相关想法的能力。",
                        'effectiveness': "你是一位专业的创新思维评估助手，专注于评估学生的创新思维中的有效性维度。有效性指的是解决方案的实用性、可行性和价值。"
                    }
                    self.prompt_templates[dimension] = default_prompts.get(dimension)
            
            if 'summary' not in self.prompt_templates or self.prompt_templates['summary'] is None:
                summary_file = os.path.join(template_dir, "summary_prompt.csv")
                if os.path.exists(summary_file):
                    try:
                        summary_df = pd.read_csv(summary_file)
                        if 'prompt' in summary_df.columns:
                            self.prompt_templates['summary'] = summary_df['prompt'][0]
                        elif 'content' in summary_df.columns:
                            self.prompt_templates['summary'] = summary_df['content'][0]
                    except Exception as e:
                        print(f"加载summary提示模板时出错: {e}")
                        self.prompt_templates['summary'] = "请综合评估学生的创新思维能力。"
                else:
                    self.prompt_templates['summary'] = "请综合评估学生的创新思维能力。"
                    
        except Exception as e:
            print(f"读取提示模板时出错: {e}")
            self.prompt_templates = {
                'flexibility': "请评估学生在灵活性方面的创新思维能力。",
                'fluency': "请评估学生在流畅性方面的创新思维能力。",
                'effectiveness': "请评估学生在有效性方面的创新思维能力。",
                'summary': "请综合评估学生的创新思维能力。"
            }
        
        self.conversation_history = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.feedback_history = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.score_trajectory = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.confidence_scores = {
            'flexibility': 0.7,
            'fluency': 0.7,
            'effectiveness': 0.7,
            'overall': 0.7
        }
        
        self.score_adjuster = DynamicScoreAdjuster(
            initial_weight=0.7,
            min_weight=0.3,
            learning_rate=0.05,
            stability_factor=0.8
        )
        
        self.original_scores = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.score_adjustments = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        self.model_agreements = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }

    def _load_question_bank(self):
        """从问题库文件中加载问题"""
        questions = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        try:
            dir_path = os.path.dirname(self.question_bank_path)
            
            flex_path = os.path.join(dir_path, 'question_flex.json')
            flu_path = os.path.join(dir_path, 'question_flu.json')
            eff_path = os.path.join(dir_path, 'question_eff.json')
            
            if os.path.exists(self.question_bank_path):
                print(f"使用统一问题库: {self.question_bank_path}")
                with open(self.question_bank_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            print(f"未找到统一问题库，尝试加载分离的问题文件...")
            
            if os.path.exists(flex_path):
                with open(flex_path, 'r', encoding='utf-8') as f:
                    flex_data = json.load(f)
                    if 'flexibility' in flex_data:
                        questions['flexibility'] = flex_data['flexibility']
                        print(f"已加载灵活性问题: {len(questions['flexibility'])}题")
            else:
                print(f"警告: 未找到灵活性问题文件 {flex_path}")
            
            if os.path.exists(flu_path):
                with open(flu_path, 'r', encoding='utf-8') as f:
                    flu_data = json.load(f)
                    if 'fluency' in flu_data:
                        questions['fluency'] = flu_data['fluency']
                        print(f"已加载流畅性问题: {len(questions['fluency'])}题")
            else:
                print(f"警告: 未找到流畅性问题文件 {flu_path}")
            
            if os.path.exists(eff_path):
                with open(eff_path, 'r', encoding='utf-8') as f:
                    eff_data = json.load(f)
                    if 'effectiveness' in eff_data:
                        questions['effectiveness'] = eff_data['effectiveness']
                        print(f"已加载有效性问题: {len(questions['effectiveness'])}题")
            else:
                print(f"警告: 未找到有效性问题文件 {eff_path}")
            
            total_questions = sum(len(questions[dim]) for dim in questions)
            if total_questions == 0:
                print("错误: 未找到任何问题，评估将无法进行！")
            else:
                print(f"已成功加载问题库: 灵活性({len(questions['flexibility'])}题), "
                     f"流畅性({len(questions['fluency'])}题), "
                     f"有效性({len(questions['effectiveness'])}题)")
            
        except Exception as e:
            print(f"加载问题库失败: {e}")
        
        return questions
    
    def select_question(self, dimension, current_score):
        """
        从题库中选择适合当前水平的问题，考虑反馈历史
        """
        candidates = self.question_bank[dimension]
        
        recent_feedbacks = self.feedback_records[dimension][-3:] if self.feedback_records[dimension] else []
        
        target_difficulty = current_score

        if recent_feedbacks:
            recent_successes = len([f for f in recent_feedbacks if f['type'] == 'success'])
            recent_failures = len([f for f in recent_feedbacks if f['type'] == 'failure'])
            
            if recent_successes > recent_failures:
                adjustment = self.feedback_influence * (recent_successes - recent_failures)
                target_difficulty = min(10, current_score + adjustment)
            elif recent_failures > recent_successes:
                adjustment = self.feedback_influence * (recent_failures - recent_successes)
                target_difficulty = max(1, current_score - adjustment)
            needs_guidance = self.needs_guidance[dimension]
        else:

            needs_guidance = False
        
        difficulty_range = 0.5 if needs_guidance else 1.5
        suitable_questions = [
            q for q in candidates 
            if 'difficulty' in q and 
            abs(q['difficulty'] - target_difficulty) <= difficulty_range
        ]
        
        if not suitable_questions:
            suitable_questions = candidates
        
        if needs_guidance:
            guided_questions = [q for q in suitable_questions if 'guidance' in q and q['guidance']]
            if guided_questions:
                question = random.choice(guided_questions)
                self.needs_guidance[dimension] = False
                return question
        
        selected_question = random.choice(suitable_questions)
        return selected_question

    def update_score_with_memory(self, dimension, question, response, history=None, previous_score=None):
        """
        使用记忆机制和双模型评分机制以及历史机制评估回答
        
        参数:
            dimension (str): 评估维度
            question (str): 问题
            response (str): 学生回答
            history (list): 对话历史
            previous_score (float): 上一次的分数
            
        返回:
            tuple: (分数, 可信度)
        """
        try:
            # 确保previous_score是浮点数
            if previous_score is not None:
                try:
                    previous_score = float(previous_score)
                except (ValueError, TypeError):
                    print(f"警告: previous_score无法转换为浮点数: {previous_score}，使用默认值5.0")
                    previous_score = 5.0
            else:
                previous_score = 0.0  # 如果没有之前的分数，则默认为0

            # 生成评分参考标准
            try:
                scoring_references = self.generate_scoring_reference(dimension, {"question": question})
            except Exception as e:
                print(f"生成评分参考标准时出错: {e}")
                # 使用默认标准
                scoring_references = {
                    2: "回答简单，缺乏创意或深度思考",
                    4: "回答包含一些基本思路，但创意有限",
                    6: "回答展示了良好的思考能力和一定创意",
                    8: "回答高度创新，思路清晰且具深度"
                }
            
            # 评估回答
            raw_score, feedback, confidence_raw = self.evaluate_exploratory_response(
                dimension, question, response, scoring_references
            )
            
            # 标准化置信度
            confidence = self._normalize_confidence(confidence_raw)
            
            # 确保raw_score是浮点数
            try:
                raw_score = float(raw_score)
            except (ValueError, TypeError):
                print(f"警告: raw_score无法转换为浮点数: {raw_score}，使用默认值5.0")
                raw_score = 5.0
            
            # 使用动态分数调整
            try:
                if hasattr(self, 'score_adjuster'):
                    adjusted_score, adjustment_info = self.score_adjuster.get_adjusted_score(
                        dimension, raw_score, confidence
                    )
                    
                    # 添加分数到历史记录
                    self.score_adjuster.add_score(dimension, adjusted_score)
                    
                    # 记录调整信息
                    if dimension in self.score_adjustments:
                        self.score_adjustments[dimension].append(adjustment_info)
                    
                    # 添加调整信息到反馈
                    adjustment_reason = adjustment_info.get('reason', '')
                    if adjustment_reason:
                        feedback += f"\n\n[系统评分调整信息] {adjustment_reason}"
                    
                    # 根据调整程度可能提高或降低置信度
                    adjustment = abs(float(adjustment_info.get('final_adjustment', 0)))
                    if adjustment > 1.5:  # 大幅调整
                        # 大幅调整降低置信度，但仍保持在合理范围
                        confidence = max(0.3, confidence - 0.1)
                else:
                    adjusted_score = raw_score
                    
            except Exception as e:
                print(f"分数调整时出错: {e}")
                import traceback
                print(traceback.format_exc())
                adjusted_score = raw_score  # 如果调整出错，使用原始分数
            
            # 记录评分历史
            if hasattr(self, 'score_trajectory') and dimension in self.score_trajectory:
                self.score_trajectory[dimension].append(float(adjusted_score))
            
            # 确保返回值都是标准化的浮点数
            return float(adjusted_score), float(confidence)
            
        except Exception as e:
            print(f"使用记忆评估回答时出错: {e}")
            import traceback
            print(traceback.format_exc())
            # 返回默认值
            return float(previous_score or 5.0), 0.5

    def evaluate_exploratory_response(self, dimension, question, response, scoring_references):
        """
        评估学生的试探性回答
        
        参数:
            dimension: 评估维度
            question: 问题内容
            response: 学生回答
            scoring_references: 评分参考标准
            
        返回:
            tuple: (分数, 反馈, 可信度)，分数和可信度均为浮点数
        """
        try:
            # 准备评估提示模板
            template = self.load_prompt_template("exploratory_evaluation")
            
            # 构建评分参考文本
            scoring_ref_text = ""
            for level, description in scoring_references.items():
                scoring_ref_text += f"{level}分水平: {description}\n"
            
            # 确保question是字符串
            question_text = ""
            if isinstance(question, dict) and "question" in question:
                question_text = question["question"]
            elif isinstance(question, str):
                question_text = question
            else:
                question_text = str(question)
            
            # 构建评估提示
            prompt = self.format_prompt(template, 
                                      dimension=dimension,
                                      question=question_text,
                                      response=response,
                                      scoring_references=scoring_ref_text)
            
            # 调用主要模型评分
            primary_response = ""
            try:
                primary_response = self.llm.query(prompt)
                print(f"主模型评估完成，响应长度: {len(primary_response)}")
            except Exception as e:
                print(f"主模型评估出错: {e}")
                return 5.0, "评估过程中发生错误，使用默认分数。", 0.5
            
            # 提取主模型评分和反馈
            primary_score, primary_feedback = self.extract_score_and_feedback(primary_response)
            
            # 确保primary_score是浮点数
            try:
                primary_score = float(primary_score)
            except (ValueError, TypeError):
                print(f"警告: 无法将主模型分数转换为浮点数: {primary_score}，使用默认值5.0")
                primary_score = 5.0
            
            # 如果启用双模型评估
            if self.use_dual_model and self.second_model is not None:
                try:
                    # 调用第二个模型进行评分
                    secondary_response = self.second_model.query(prompt)
                    secondary_score, secondary_feedback = self.extract_score_and_feedback(secondary_response)
                    
                    # 确保secondary_score是浮点数
                    try:
                        secondary_score = float(secondary_score)
                    except (ValueError, TypeError):
                        print(f"警告: 无法将第二个模型的分数转换为浮点数: {secondary_score}，使用默认值5.0")
                        secondary_score = 5.0
                    
                    # 计算评分差异和可信度
                    score_diff = abs(primary_score - secondary_score)
                    
                    if score_diff <= self.model_agreement_threshold:
                        # 评分一致，使用平均值
                        model_consistent = True
                        final_score = (primary_score + secondary_score) / 2
                        final_feedback = primary_feedback  # 使用主模型反馈
                        confidence = 0.9 - (score_diff / 10)  # 根据差异计算可信度
                        print(f"双模型评分一致: {primary_score} vs {secondary_score}, 差异: {score_diff}, 可信度: {confidence}")
                    elif score_diff >= self.low_confidence_threshold:
                        # 评分差异过大，标记为低可信度
                        model_consistent = False
                        final_score = primary_score  # 暂时使用主模型评分
                        final_feedback = f"[低可信度评分] {primary_feedback}\n\n备注: 模型评分差异较大({primary_score} vs {secondary_score})，建议人工审核。"
                        confidence = 0.4  # 低可信度
                        print(f"警告: 双模型评分差异较大: {primary_score} vs {secondary_score}, 差异: {score_diff}, 可信度: {confidence}")
                    else:
                        # 评分有一定差异但在可接受范围内
                        model_consistent = False
                        final_score = primary_score  # 使用主模型评分
                        final_feedback = primary_feedback
                        confidence = 0.7 - (score_diff / 10)  # 中等可信度
                        print(f"双模型评分存在差异: {primary_score} vs {secondary_score}, 差异: {score_diff}, 可信度: {confidence}")
                    
                    # 更新模型权重
                    if hasattr(self, 'score_adjuster'):
                        self.score_adjuster.update_weights(dimension, model_consistent)
                    
                    # 记录模型一致性
                    if hasattr(self, 'model_agreements') and dimension in self.model_agreements:
                        self.model_agreements[dimension].append(score_diff)
                    
                    # 返回统一格式的结果
                    return float(final_score), final_feedback, float(confidence)
                    
                except Exception as e:
                    print(f"双模型评估出错: {e}")
                    # 出错时使用主模型结果，降低可信度
                    if hasattr(self, 'model_agreements') and dimension in self.model_agreements:
                        self.model_agreements[dimension].append(1.0)  # 记录一个中等差异值
                    return float(primary_score), primary_feedback, 0.6
            
            # 如果没有启用双模型评估，直接返回主模型结果
            # 记录模型一致性（单模型情况）
            if hasattr(self, 'model_agreements') and dimension in self.model_agreements:
                self.model_agreements[dimension].append(0.5)  # 单模型时使用中等一致性值
                
            return float(primary_score), primary_feedback, 0.7  # 默认单模型可信度0.7
            
        except Exception as e:
            print(f"评估探索性回答时出错: {e}")
            import traceback
            print(traceback.format_exc())
            # 发生错误时，返回默认评分和反馈
            return 5.0, "由于技术原因无法完成评估，使用默认中等分数。", 0.5

    def human_review_needed(self, confidence, dimension, score, feedback, question, response):
        """
        判断是否需要人工审核
        
        参数:
            confidence: 可信度
            dimension: 评估维度
            score: 评分
            feedback: 反馈
            question: 问题
            response: 学生回答
            
        返回:
            bool: 是否需要人工审核
        """
        if confidence < 0.5:
            print(f"低可信度评分({confidence})，建议人工审核: {dimension}维度，分数:{score}")
            
            review_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dimension': dimension,
                'question': question,
                'response': response,
                'model_score': score,
                'model_feedback': feedback,
                'confidence': confidence
            }
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            review_dir = os.path.join(current_dir, "human_review")
            os.makedirs(review_dir, exist_ok=True)
            
            review_file = os.path.join(review_dir, f"review_{dimension}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            try:
                with open(review_file, 'w', encoding='utf-8') as f:
                    json.dump(review_data, f, ensure_ascii=False, indent=2)
                print(f"已保存到人工审核队列: {review_file}")
            except Exception as e:
                print(f"保存审核数据时出错: {e}")
            
            return True
        
        return False

    def generate_follow_up_question(self, dimension, history, is_guidance=False):
        """生成针对学生回答的深入提问，可选是否提供引导"""
        history_content = []
        
        recent_conversations = history[-self.K:] if len(history) > 0 else []
        for i, conv in enumerate(recent_conversations):
            history_content.append(f"问题 {i+1}: {conv['question']}")
            history_content.append(f"学生回答: {conv['response']}")
            if 'score' in conv:
                history_content.append(f"得分: {conv['score']}")
        
        history_text = "\n".join(history_content)
        
        if is_guidance or self.needs_guidance[dimension]:
            template = self.load_prompt_template("guidance_followup")
            prompt = self.format_prompt(template,
                dimension=dimension,
                dialogue_history=history_text
            )
        else:
            template = self.load_prompt_template("followup_question")
            prompt = self.format_prompt(template,
                dimension=dimension,
                dialogue_history=history_text
            )
        
        try:
            response = self.llm.query(prompt)
        except Exception as e:
            print(f"生成跟进问题时出错: {e}")
            default_followups = {
                'flexibility': '你能从不同角度思考这个问题吗？请提供至少两种不同的思路。',
                'fluency': '你能扩展你的想法，提供更多细节和可能性吗？',
                'effectiveness': '你能详细说明这个解决方案如何具体实施，以及它的可行性和影响吗？'
            }
            return default_followups.get(dimension, '请进一步解释你的想法，并提供更多细节。')
        
        lines = response.strip().split('\n')
        follow_up = lines[-1] if lines else response
        
        if len(follow_up) > 100:
            question_marks = [i for i, char in enumerate(follow_up) if char in '?？']
            if question_marks:
                last_mark = question_marks[-1]
                start = max(0, last_mark - 100)
                end = min(len(follow_up), last_mark + 1)
                follow_up = follow_up[start:end]
        
        return follow_up

    def is_dimension_stable(self, dimension):
        """判断某维度的分数是否稳定"""
        scores = [entry['score'] for entry in self.dimension_history[dimension]]
        
        if len(scores) < self.stability_window:
            return False
        
        recent_scores = scores[-self.stability_window:]
        
        changes = [abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))]
        
        return all(change <= self.stability_tolerance for change in changes)

    def evaluate_dimension(self, dimension, student_id, max_steps=10, answer_provider=None):
        """
        评估特定维度的创新思维
        
        参数:
            dimension (str): 要评估的维度，例如"flexibility", "fluency", "effectiveness"
            student_id (str): 学生ID
            max_steps (int): 最大对话步数
            answer_provider (callable, optional): 接收问题并返回回答的函数，优先于input()
        
        返回:
            dict: 包含评分和反馈的评估结果
        """
        try:
            results = {
                'dimension': dimension,
                'student_id': student_id,
                'steps': [],
                'score': 0,
                'feedback': '',
                'dialogue_history': [],  
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"\n开始评估维度: {dimension}")
            
            prompt_name = f"{dimension}_prompt"
            prompt_template = self.load_prompt_template(prompt_name)
            
            self.llm.set_system_prompt(prompt_template)
            
            # 初始化对话历史和分数跟踪
            self.dialog_history_by_dimension[dimension] = []
            dialogue_history = []
            current_step = 0
            current_score = 0.0  # 初始分数为0
            previous_score = 0.0  # 初始上一轮分数为0
            current_confidence = 0.7  # 默认初始置信度
            is_stable = False
            
            while current_step < max_steps and not is_stable:
                current_step += 1
                print(f"\n{dimension} 维度 - 第 {current_step} 轮对话")
                
                if current_step == 1:
                    selected_question = self.select_exploratory_question(dimension)
                else:
                    selected_question = self.generate_follow_up_question(
                        dimension, 
                        dialogue_history,
                        is_guidance=self.needs_guidance.get(dimension, False)
                    )
                
                question_text = selected_question
                if isinstance(selected_question, dict) and 'question' in selected_question:
                    question_text = selected_question['question']
                
                print(f"\n问题: {question_text}")
                
                # 添加问题到对话历史
                self.add_msg_to_dialog_history(dimension, {"role": "user", "content": question_text})
                
                if answer_provider is not None:
                    answer = answer_provider(question_text)
                else:
                    answer = input("请输入回答: ")
                
                print(f"回答: {answer}")
                
                # 添加回答到对话历史
                self.add_msg_to_dialog_history(dimension, {"role": "assistant", "content": answer})
                
                if current_step == 1 or not isinstance(selected_question, str):
                    reference_prompt_name = f"{dimension}_reference"
                    reference_prompt_template = self.load_prompt_template(reference_prompt_name)
                    reference_prompt = self.format_prompt(reference_prompt_template, dimension=dimension, question=question_text)
                    
                    try:
                        scoring_references_text = self.llm.query(reference_prompt)
                        
                        score_patterns = [
                            (2, r"(2分|分数2|level_2)[：:](.*?)(?=(4分|分数4|level_4)[：:]|$)"),
                            (4, r"(4分|分数4|level_4)[：:](.*?)(?=(6分|分数6|level_6)[：:]|$)"),
                            (6, r"(6分|分数6|level_6)[：:](.*?)(?=(8分|分数8|level_8)[：:]|$)"),
                            (8, r"(8分|分数8|level_8)[：:](.*?)(?=$)")
                        ]
                        
                        scoring_references = {}
                        for score, pattern in score_patterns:
                            match = re.search(pattern, scoring_references_text, re.DOTALL)
                            if match:
                                scoring_references[score] = match.group(2).strip()
                        
                        if len(scoring_references) < 2:
                            scoring_references = {
                                2: "回答简单，缺乏创新性，未能充分回应问题",
                                4: "回答基本合理，有一定创新性，但缺乏深度思考",
                                6: "回答创新性良好，思路清晰，有一定实用价值",
                                8: "回答高度创新，思路深入，具有很强的实用价值和可行性"
                            }
                    except Exception as e:
                        print(f"获取评分参考标准时出错: {e}")
                        scoring_references = {
                            2: "回答简单，缺乏创新性，未能充分回应问题",
                            4: "回答基本合理，有一定创新性，但缺乏深度思考",
                            6: "回答创新性良好，思路清晰，有一定实用价值",
                            8: "回答高度创新，思路深入，具有很强的实用价值和可行性"
                        }
                  
                try:
                    if current_step == 1:
                        # 第一轮对话评估
                        raw_score, feedback, eval_confidence = self.evaluate_exploratory_response(
                            dimension, 
                            {"question": question_text}, 
                            answer,
                            scoring_references
                        )
                        
                        # 标准化置信度，确保是0-1之间的浮点数
                        confidence_value = self._normalize_confidence(eval_confidence)
                        
                        # 第一轮不基于历史调整，原始分数为0，调整后的分数为当前评估分数
                        adjusted_score = float(raw_score)
                        previous_score = 0.0  # 第一轮原始分数显示为0
                    else:
                        # 后续轮对话评估
                        raw_score, feedback, eval_confidence = self.evaluate_exploratory_response(
                            dimension, 
                            {"question": question_text}, 
                            answer,
                            scoring_references
                        )
                        
                        # 标准化置信度
                        temp_confidence = self._normalize_confidence(eval_confidence)
                        
                        # 使用历史分数和当前评估进行调整
                        adjusted_score, confidence_data = self.update_score_with_memory(
                            dimension, 
                            question_text, 
                            answer, 
                            dialogue_history,
                            current_score  # 使用当前分数作为上一轮分数
                        )
                        
                        # 标准化来自update_score_with_memory的置信度
                        confidence_value = self._normalize_confidence(confidence_data)
                    
                    # 更新当前置信度
                    current_confidence = confidence_value
                    
                    # 记录本轮对话
                    dialogue_entry = {
                        'step': current_step,
                        'question': question_text,
                        'response': answer, 
                        'raw_score': float(raw_score),
                        'adjusted_score': float(adjusted_score),
                        'confidence': confidence_value,
                        'feedback': feedback
                    }
                    
                    dialogue_history.append(dialogue_entry)
                    results['dialogue_history'] = dialogue_history
                    
                    # 记录步骤
                    step = {
                        'step': current_step,
                        'question': question_text,
                        'answer': answer,
                        'raw_score': previous_score if current_step > 1 else 0.0,  # 显示上一轮分数作为原始分数
                        'adjusted_score': float(adjusted_score),
                        'confidence': confidence_value,
                        'feedback': feedback
                    }
                    results['steps'].append(step)
                    
                    # 更新当前分数
                    previous_score = current_score  # 保存上一轮分数
                    current_score = float(adjusted_score)  # 更新当前分数
                    
                    # 检查是否满足稳定条件
                    if current_step >= 3:  # 至少需要3轮对话才能检测稳定性
                        recent_scores = [float(entry['adjusted_score']) for entry in dialogue_history[-3:]]
                        max_deviation = max([abs(recent_scores[i] - recent_scores[i-1]) for i in range(1, len(recent_scores))])
                        
                        if max_deviation <= 1:  # 连续三轮波动不超过1
                            print(f"\n{dimension}维度分数已稳定，连续三轮波动不超过1")
                            is_stable = True
                    
                    # 打印评估结果
                    if current_step == 1:
                        print(f"\n第{current_step}轮评估: 原始分数={0.0}, 调整后分数={adjusted_score}, 置信度={confidence_value:.2f}")
                    else:
                        print(f"\n第{current_step}轮评估: 原始分数={previous_score}, 调整后分数={adjusted_score}, 置信度={confidence_value:.2f}")
                    print(f"反馈: {feedback}")
                    
                    # 记录成功/失败反馈
                    if current_step > 1 and len(dialogue_history) >= 2:
                        score_change = float(adjusted_score) - float(dialogue_history[-2]['adjusted_score'])
                        if abs(score_change) >= 0.15:
                            feedback_type = 'success' if score_change > 0 else 'failure'
                            feedback_record = {
                                'type': feedback_type,
                                'step': current_step,
                                'question': question_text,
                                'response': answer,
                                'raw_score': float(raw_score),
                                'adjusted_score': float(adjusted_score),
                                'score_change': float(score_change),
                                'confidence': confidence_value,  # 添加置信度
                                'feedback': feedback
                            }
                            
                            if dimension not in self.feedback_records:
                                self.feedback_records[dimension] = []
                            
                            self.feedback_records[dimension].append(feedback_record)
                            print(f"原始分数：{raw_score}, 置信度：{confidence_value:.2f}")
                            print(f"记录{feedback_type}反馈: 分数变化={score_change}")
                    
                except Exception as e:
                    print(f"评估回答时出错: {e}")
                    import traceback
                    print(traceback.format_exc())
                    if current_step > 1 and dialogue_history:
                        current_score = float(dialogue_history[-1]['adjusted_score'])
                        current_confidence = float(dialogue_history[-1]['confidence'])
                    else:
                        current_score = 5.0  # 默认中等分数
                        current_confidence = 0.7  # 默认中等置信度
            
            # 对话结束，记录最终结果
            if dialogue_history:
                final_entry = dialogue_history[-1]
                results['score'] = float(final_entry['adjusted_score'])
                results['feedback'] = final_entry['feedback']
                results['raw_score'] = float(final_entry['raw_score'])
                results['confidence'] = float(final_entry['confidence'])
            else:
                results['score'] = 5.0
                results['feedback'] = "无法完成评估，给予默认中等分数。"
                results['confidence'] = 0.7
            
            self.save_evaluation_results(results)
            
            print(f"\n评估结束: {dimension} 维度 - 共{current_step}轮对话")
            print(f"最终得分: {results['score']}/10")
            print(f"最终置信度: {results['confidence']:.2f}")
            print(f"反馈: {results['feedback']}")
            
            return results
        except Exception as e:
            print(f"评估维度 {dimension} 时发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 返回错误信息
            return {
                'dimension': dimension,
                'student_id': student_id,
                'steps': [],
                'score': 0,
                'confidence': 0.5,  # 添加默认置信度
                'feedback': f'评估过程中出错: {str(e)}',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
            
    def _normalize_confidence(self, confidence_value):
        """
        标准化置信度值，确保返回0-1之间的浮点数
        
        参数:
            confidence_value: 可能是浮点数、整数、字典或其他类型的置信度值
            
        返回:
            float: 标准化后的0-1之间的置信度浮点数
        """
        try:
            # 如果是字典类型，尝试提取score键的值
            if isinstance(confidence_value, dict) and 'score' in confidence_value:
                confidence = float(confidence_value['score'])
            # 如果是数值类型，直接转换为浮点数
            elif isinstance(confidence_value, (int, float)):
                confidence = float(confidence_value)
            # 其他情况使用默认值
            else:
                print(f"警告: 无法解析的置信度值类型: {type(confidence_value)}, 使用默认值0.7")
                return 0.7
                
            # 确保置信度在0-1范围内
            if confidence > 1.0:
                # 如果置信度超过1，可能是按照0-10的分数比例，将其归一化
                if confidence <= 10.0:
                    confidence = confidence / 10.0
                else:
                    confidence = 0.7  # 超出正常范围，使用默认值
            
            # 避免非常小的置信度值
            confidence = max(0.1, min(1.0, confidence))
            
            return confidence
        except (ValueError, TypeError):
            print(f"警告: 置信度值 {confidence_value} 无法转换为浮点数，使用默认值0.7")
            return 0.7

    def run_evaluation(self, student_id, answer_provider=None, max_steps=None):
        """
        运行完整的创新思维评估过程，评估所有维度
        
        参数:
            student_id (str): 学生ID
            answer_provider (callable, optional): 接收问题并返回回答的函数
            max_steps (int, optional): 每个维度的最大对话步数，默认使用self.max_steps
            
        返回:
            dict: 包含所有维度评估结果的完整报告
        """
        if max_steps is not None:
            self.max_steps = max_steps
            
        start_time = datetime.now()
        
        evaluation_results = {
            'student_id': student_id,
            'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'dimensions': {},
            'overall_score': 0,
            'report': '',
            'duration': 0
        }
        
        try:
            print(f"\n===== 开始学生 {student_id} 的创新思维评估 =====")
            
            for dimension in self.dimensions:
                print(f"\n----- 开始评估 {dimension} 维度 -----")
                
                try:
                    dimension_result = self.evaluate_dimension(
                        dimension, 
                        student_id, 
                        max_steps=self.max_steps,
                        answer_provider=answer_provider
                    )
                    
                    evaluation_results['dimensions'][dimension] = dimension_result
                except Exception as e:
                    print(f"评估维度 {dimension} 时出错: {e}")
                    evaluation_results['dimensions'][dimension] = {
                        'dimension': dimension,
                        'error': str(e),
                        'score': 0,
                        'feedback': f'评估过程中出错: {str(e)}'
                    }
            
            valid_dimensions = [d for d in evaluation_results['dimensions'].values() 
                               if 'error' not in d and d.get('score', 0) > 0]
            
            if valid_dimensions:
                total_score = sum(d.get('score', 0) for d in valid_dimensions)
                evaluation_results['overall_score'] = total_score / len(valid_dimensions)
            else:
                evaluation_results['overall_score'] = 0
                evaluation_results['error'] = "所有维度评估均失败"
            
            try:
                report = self.generate_report(evaluation_results)
                evaluation_results['report'] = report
            except Exception as e:
                print(f"生成报告时出错: {e}")
                evaluation_results['report'] = f"无法生成完整报告。错误: {str(e)}"
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            evaluation_results['duration'] = duration
            
            print(f"\n===== 评估完成 =====")
            print(f"总体得分: {evaluation_results['overall_score']:.2f}/10")
            print(f"评估耗时: {duration:.2f}秒")
            
            self.save_final_results(evaluation_results)
            
            return evaluation_results
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            evaluation_results['error'] = str(e)
            evaluation_results['duration'] = duration
            evaluation_results['overall_score'] = 0
            
            try:
                self.save_final_results(evaluation_results)
            except:
                pass
                
            return evaluation_results

    def evaluate_exploratory_response(self, dimension, question, response, scoring_references):
        """评估探索性问题的回答，使用相对评分标准"""
        template = self.load_prompt_template("exploratory_evaluation")
        
        scoring_reference_text = "\n".join([
            f"分数{score}标准：{standard}" for score, standard in scoring_references.items()
        ])
        
        prompt = self.format_prompt(template,
            dimension=dimension,
            question=question["question"] if isinstance(question, dict) and "question" in question else question,
            response=response,
            scoring_references=scoring_reference_text
        )
        
        try:
            evaluation_text = self.llm.query(prompt)
            score, feedback = self.extract_score_and_feedback(evaluation_text)
            
            self.model_agreements[dimension].append(0.5)
            
            return score, feedback, {"score": score}
        except Exception as e:
            print(f"评估回答时出错: {e}")
            return 5.0, "无法完成评估，给予默认中等分数。", {"error": str(e)}

    def generate_summary_report(self, student_id, dimension_results, confidence_scores, dimension_reports):
        """生成综合评估报告，基于整个评估过程"""
        overall_score = sum(dimension_results.values()) / len(dimension_results)
        
        learning_trends = {
            dimension: self.score_adjuster.get_learning_trend(dimension)
            for dimension in dimension_results.keys()
        }
        
        template = self.load_prompt_template("summary_report")
        
        # 获取完整对话历史
        flexibility_dialogues = dimension_reports.get("flexibility", {}).get("dialogue_history", [])
        fluency_dialogues = dimension_reports.get("fluency", {}).get("dialogue_history", [])
        effectiveness_dialogues = dimension_reports.get("effectiveness", {}).get("dialogue_history", [])
        
        # 格式化维度对话历史
        formatted_flexibility_dialogues = self._format_dimension_dialogues(flexibility_dialogues)
        formatted_fluency_dialogues = self._format_dimension_dialogues(fluency_dialogues)
        formatted_effectiveness_dialogues = self._format_dimension_dialogues(effectiveness_dialogues)
        
        # 格式化成功/失败反馈记录
        formatted_feedback = self._format_feedback_records()
        
        # 格式化学习轨迹
        formatted_learning_trajectory = self._format_learning_trajectory()
        
        # 构建完整对话历史摘要
        conversation_summary = (
            "【灵活性维度对话历史】\n" + formatted_flexibility_dialogues + "\n\n" +
            "【流畅性维度对话历史】\n" + formatted_fluency_dialogues + "\n\n" +
            "【有效性维度对话历史】\n" + formatted_effectiveness_dialogues
        )
        
        # 构建反馈历史摘要
        feedback_summary = (
            "【灵活性维度成功反馈】\n" + formatted_feedback.get('flexibility_success', '无成功反馈') + "\n\n" +
            "【灵活性维度失败反馈】\n" + formatted_feedback.get('flexibility_failure', '无失败反馈') + "\n\n" +
            "【流畅性维度成功反馈】\n" + formatted_feedback.get('fluency_success', '无成功反馈') + "\n\n" +
            "【流畅性维度失败反馈】\n" + formatted_feedback.get('fluency_failure', '无失败反馈') + "\n\n" +
            "【有效性维度成功反馈】\n" + formatted_feedback.get('effectiveness_success', '无成功反馈') + "\n\n" +
            "【有效性维度失败反馈】\n" + formatted_feedback.get('effectiveness_failure', '无失败反馈')
        )
        
        # 准备评分轨迹
        score_trajectories = {
            "flexibility": [entry.get('adjusted_score', 0) for entry in flexibility_dialogues],
            "fluency": [entry.get('adjusted_score', 0) for entry in fluency_dialogues],
            "effectiveness": [entry.get('adjusted_score', 0) for entry in effectiveness_dialogues]
        }
        
        # 获取各维度的关键问题回答
        key_responses = self._extract_key_responses()
        
        context = {
            "student_id": student_id,
            "flexibility_score": dimension_results.get("flexibility", 0),
            "fluency_score": dimension_results.get("fluency", 0),
            "effectiveness_score": dimension_results.get("effectiveness", 0),
            "overall_score": overall_score,
            "flexibility_confidence": confidence_scores.get("flexibility", 0.7),
            "fluency_confidence": confidence_scores.get("fluency", 0.7),
            "effectiveness_confidence": confidence_scores.get("effectiveness", 0.7),
            "overall_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.7,
            "flexibility_trend": learning_trends.get("flexibility", {}).get("trend", "稳定"),
            "fluency_trend": learning_trends.get("fluency", {}).get("trend", "稳定"),
            "effectiveness_trend": learning_trends.get("effectiveness", {}).get("trend", "稳定"),
            "conversation_summary": conversation_summary,
            "feedback_summary": feedback_summary,
            "flexibility_trajectory": str(score_trajectories["flexibility"]),
            "fluency_trajectory": str(score_trajectories["fluency"]),
            "effectiveness_trajectory": str(score_trajectories["effectiveness"]),
            "key_responses": key_responses,
            "learning_trajectory": formatted_learning_trajectory
        }
        
        prompt = self.format_prompt(template, **context)
        
        try:
            report_text = self.llm.query(prompt)
        except Exception as e:
            print(f"生成报告时出错: {e}")
            report_text = f"无法生成完整的评估报告。错误: {str(e)}\n\n总分: {overall_score}/10\n\n各维度得分:\n"
            for dimension, score in dimension_results.items():
                report_text += f"- {dimension}: {score}/10\n"
        
        try:
            feedback_analysis = self.generate_feedback_analysis(
                student_id, 
                dimension_results, 
                confidence_scores, 
                dimension_reports, 
                learning_trends
            )
        except Exception as e:
            print(f"生成深入分析报告时出错: {e}")
            feedback_analysis = {
                "analysis_text": f"无法生成深入分析报告。错误: {str(e)}"
            }
        
        summary_report = {
            "student_id": student_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimension_scores": dimension_results,
            "overall_score": overall_score,
            "confidence_scores": confidence_scores,
            "learning_trends": learning_trends,
            "dimension_reports": dimension_reports,
            "report_text": report_text,
            "feedback_analysis": feedback_analysis,
            "adjustment_history": {
                "flexibility": self.score_adjustments["flexibility"],
                "fluency": self.score_adjustments["fluency"],
                "effectiveness": self.score_adjustments["effectiveness"]
            },
            "model_agreement_history": {
                "flexibility": self.model_agreements.get("flexibility", []),
                "fluency": self.model_agreements.get("fluency", []),
                "effectiveness": self.model_agreements.get("effectiveness", [])
            },
            "score_trajectories": score_trajectories,
            "key_responses": key_responses
        }
        
        return summary_report
    
    def _format_dimension_dialogues(self, dialogues):
        """格式化维度对话历史
        
        参数:
            dialogues: 对话历史列表
            
        返回:
            str: 格式化后的对话历史文本
        """
        if not dialogues:
            return "无对话记录"
            
        formatted_text = ""
        for i, dialogue in enumerate(dialogues):
            round_num = i + 1
            question = dialogue.get("question", "未知问题")
            response = dialogue.get("response", "未知回答")
            raw_score = dialogue.get("raw_score", 0)
            adjusted_score = dialogue.get("adjusted_score", 0)
            feedback = dialogue.get("feedback", "无反馈")
            
            # 截断过长的内容
            if len(question) > 150:
                question = question[:147] + "..."
            if len(response) > 150:
                response = response[:147] + "..."
            if len(feedback) > 200:
                feedback = feedback[:197] + "..."
            
            formatted_text += f"轮次 {round_num}:\n"
            formatted_text += f"问题: {question}\n"
            formatted_text += f"回答: {response}\n"
            formatted_text += f"原始分数: {raw_score}/10\n"
            formatted_text += f"调整后分数: {adjusted_score}/10\n"
            formatted_text += f"反馈: {feedback}\n\n"
            
        return formatted_text
    
    def _format_feedback_records(self):
        """格式化成功/失败反馈记录
        
        返回:
            dict: 包含各维度成功/失败反馈的格式化文本
        """
        result = {}
        
        for dimension in ['flexibility', 'fluency', 'effectiveness']:
            # 成功反馈
            success_records = [r for r in self.feedback_records.get(dimension, []) if r.get('type') == 'success']
            if success_records:
                success_text = ""
                for i, record in enumerate(success_records):
                    success_text += f"- 轮次 {record.get('step', i+1)}:\n"
                    success_text += f"  问题: {record.get('question', '未知问题')[:150]}\n"
                    success_text += f"  回答: {record.get('response', '未知回答')[:150]}\n"
                    success_text += f"  分数: {record.get('adjusted_score', 0)}/10\n"
                    success_text += f"  分数变化: +{record.get('score_change', 0)}\n"
                    
                    if 'feedback' in record:
                        feedback = record.get('feedback', '')[:200]
                        success_text += f"  反馈: {feedback}\n\n"
                
                result[f"{dimension}_success"] = success_text
            else:
                result[f"{dimension}_success"] = "无成功反馈记录"
                
            # 失败反馈
            failure_records = [r for r in self.feedback_records.get(dimension, []) if r.get('type') == 'failure']
            if failure_records:
                failure_text = ""
                for i, record in enumerate(failure_records):
                    failure_text += f"- 轮次 {record.get('step', i+1)}:\n"
                    failure_text += f"  问题: {record.get('question', '未知问题')[:150]}\n"
                    failure_text += f"  回答: {record.get('response', '未知回答')[:150]}\n"
                    failure_text += f"  分数: {record.get('adjusted_score', 0)}/10\n"
                    failure_text += f"  分数变化: {record.get('score_change', 0)}\n"
                    
                    if 'feedback' in record:
                        feedback = record.get('feedback', '')[:200]
                        failure_text += f"  反馈: {feedback}\n\n"
                
                result[f"{dimension}_failure"] = failure_text
            else:
                result[f"{dimension}_failure"] = "无失败反馈记录"
        
        return result
    
    def _format_learning_trajectory(self):
        """格式化学习轨迹数据
        
        返回:
            str: 格式化后的学习轨迹文本
        """
        formatted_text = ""
        
        for dimension in ['flexibility', 'fluency', 'effectiveness']:
            formatted_text += f"【{dimension}维度学习轨迹】\n"
            
            # 获取分数轨迹
            history = self.dimension_history.get(dimension, [])
            if history:
                formatted_text += f"历史分数: {history}\n"
                
                # 计算趋势
                if len(history) >= 3:
                    first_score = history[0]
                    last_score = history[-1]
                    change = last_score - first_score
                    
                    if change > 1:
                        trend = "显著提升"
                    elif change > 0.5:
                        trend = "稳步提升"
                    elif change > -0.5:
                        trend = "基本稳定"
                    elif change > -1:
                        trend = "略有下降"
                    else:
                        trend = "明显下降"
                        
                    formatted_text += f"趋势: {trend} (变化: {change})\n"
                else:
                    formatted_text += "数据点不足，无法分析趋势\n"
            else:
                formatted_text += "无历史数据\n"
                
            formatted_text += "\n"
        
        return formatted_text
    
    def _extract_key_responses(self):
        """提取关键问题和回答
        
        返回:
            str: 格式化后的关键问题回答文本
        """
        formatted_text = ""
        
        for dimension in ['flexibility', 'fluency', 'effectiveness']:
            formatted_text += f"【{dimension}维度关键回答】\n"
            
            # 获取对话历史
            dimension_reports = getattr(self, 'dimension_reports', {})
            dialogues = dimension_reports.get(dimension, {}).get('dialogue_history', [])
            
            if not dialogues:
                formatted_text += "无对话记录\n\n"
                continue
                
            # 找出得分最高和最低的回答
            if len(dialogues) > 0:
                # 最高分回答
                highest_score_dialogue = max(dialogues, key=lambda x: x.get('adjusted_score', 0))
                highest_score = highest_score_dialogue.get('adjusted_score', 0)
                highest_question = highest_score_dialogue.get('question', '未知问题')[:150]
                highest_response = highest_score_dialogue.get('response', '未知回答')[:150]
                
                formatted_text += f"最高分回答 ({highest_score}/10):\n"
                formatted_text += f"问题: {highest_question}\n"
                formatted_text += f"回答: {highest_response}\n\n"
                
                # 最低分回答
                lowest_score_dialogue = min(dialogues, key=lambda x: x.get('adjusted_score', 0))
                lowest_score = lowest_score_dialogue.get('adjusted_score', 0)
                lowest_question = lowest_score_dialogue.get('question', '未知问题')[:150]
                lowest_response = lowest_score_dialogue.get('response', '未知回答')[:150]
                
                formatted_text += f"最低分回答 ({lowest_score}/10):\n"
                formatted_text += f"问题: {lowest_question}\n"
                formatted_text += f"回答: {lowest_response}\n\n"
            
        return formatted_text
    
    def generate_feedback_analysis(self, student_id, dimension_results, confidence_scores, dimension_reports, learning_trends):
        """
        基于成功/失败反馈记录生成深入的能力分析报告
        综合学生在整个测评过程中的表现变化、对关键问题的响应质量以及系统生成的引导策略
        生成针对性的能力分析和学习建议
        """
        # 获取成功/失败反馈记录
        success_feedback_records = {
            dimension: [
                record for record in self.feedback_records.get(dimension, []) 
                if record.get('type', '') == 'success'
            ]
            for dimension in dimension_results.keys()
        }
        
        failure_feedback_records = {
            dimension: [
                record for record in self.feedback_records.get(dimension, []) 
                if record.get('type', '') == 'failure'
            ]
            for dimension in dimension_results.keys()
        }
        
        # 提取每个维度的关键反馈内容
        key_responses = {}
        for dimension in dimension_results.keys():
            key_responses[dimension] = []
            
            # 添加成功反馈中的关键问题和回答
            for record in success_feedback_records[dimension]:
                key_responses[dimension].append({
                    'question': record.get('question', '')[:150],
                    'response': record.get('response', '')[:150],
                    'score': record.get('adjusted_score', 0),
                    'feedback_type': 'success',
                    'score_change': record.get('score_change', 0)
                })
            
            # 添加失败反馈中的关键问题和回答
            for record in failure_feedback_records[dimension]:
                key_responses[dimension].append({
                    'question': record.get('question', '')[:150],
                    'response': record.get('response', '')[:150],
                    'score': record.get('adjusted_score', 0),
                    'feedback_type': 'failure',
                    'score_change': record.get('score_change', 0)
                })
        
        # 识别需要引导的情况
        guidance_history = {
            dimension: [
                {
                    'step': i+1,
                    'needed_guidance': self.needs_guidance.get(dimension, False),
                    'question': entry.get('question', '')[:150],
                    'response': entry.get('response', '')[:150],
                    'score': entry.get('adjusted_score', 0)
                }
                for i, entry in enumerate(dimension_reports[dimension].get('dialogue_history', []))
                if self.needs_guidance.get(dimension, False)
            ]
            for dimension in dimension_results.keys()
        }
        
        # 获取分数调整历史
        score_adjustment_history = {
            dimension: self.score_adjustments.get(dimension, [])
            for dimension in dimension_results.keys()
        }
        
        # 分析学习趋势
        learning_trends_text = ""
        for dimension, trend in learning_trends.items():
            trend_value = trend.get('trend', '稳定')
            slope = trend.get('slope', 0)
            description = trend.get('description', '')
            
            learning_trends_text += f"{dimension}: {trend_value} (斜率: {slope})\n"
            if description:
                learning_trends_text += f"{description}\n"
        
        # 构建对话历史摘要
        dialogue_history_text = {}
        for dimension, reports in dimension_reports.items():
            if 'dialogue_history' in reports:
                history = reports['dialogue_history']
                formatted_history = self._format_dimension_dialogues(history)
                dialogue_history_text[dimension] = formatted_history
            else:
                dialogue_history_text[dimension] = "无对话历史"
        
        # 加载分析提示模板
        template = self.load_prompt_template("feedback_analysis")
        
        # 构建提示上下文
        context = {
            "student_id": student_id,
            "flexibility_score": dimension_results.get("flexibility", 0),
            "fluency_score": dimension_results.get("fluency", 0),
            "effectiveness_score": dimension_results.get("effectiveness", 0),
            "overall_score": sum(dimension_results.values()) / len(dimension_results) if dimension_results else 0,
            "flexibility_confidence": confidence_scores.get("flexibility", 0.7),
            "fluency_confidence": confidence_scores.get("fluency", 0.7),
            "effectiveness_confidence": confidence_scores.get("effectiveness", 0.7),
            "overall_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.7,
            "flexibility_trend": learning_trends.get("flexibility", {}).get("trend", "稳定"),
            "fluency_trend": learning_trends.get("fluency", {}).get("trend", "稳定"),
            "effectiveness_trend": learning_trends.get("effectiveness", {}).get("trend", "稳定"),
            "learning_trends": learning_trends_text,
            "flexibility_dialogues": dialogue_history_text.get("flexibility", "无对话历史"),
            "fluency_dialogues": dialogue_history_text.get("fluency", "无对话历史"),
            "effectiveness_dialogues": dialogue_history_text.get("effectiveness", "无对话历史"),
            "success_feedback_records": json.dumps(success_feedback_records, ensure_ascii=False, indent=2),
            "failure_feedback_records": json.dumps(failure_feedback_records, ensure_ascii=False, indent=2),
            "key_responses": json.dumps(key_responses, ensure_ascii=False, indent=2),
            "guidance_history": json.dumps(guidance_history, ensure_ascii=False, indent=2),
            "score_adjustment_history": json.dumps(score_adjustment_history, ensure_ascii=False, indent=2)
        }
        
        # 格式化提示
        prompt = self.format_prompt(template, **context)
        
        # 调用大模型生成分析报告
        try:
            analysis_text = self.llm.query(prompt)
        except Exception as e:
            print(f"生成分析报告时出错: {e}")
            analysis_text = f"无法生成详细分析报告。错误: {str(e)}"
        
        # 返回分析结果
        return {
            "analysis_text": analysis_text,
            "success_feedback_records": success_feedback_records,
            "failure_feedback_records": failure_feedback_records,
            "key_responses": key_responses,
            "guidance_history": guidance_history,
            "score_adjustment_history": score_adjustment_history,
            "learning_trends": learning_trends
        }

    def load_prompt_template(self, template_name):
        """
        从提示模板文件中加载特定名称的模板
        根据模板名称前缀自动选择正确的提示文件
        """
        try:
            template_path = self.prompt_template_path  
            
            template_dir = os.path.dirname(self.prompt_template_path)
            
            if template_name in ['flexibility_prompt', 'fluency_prompt', 'effectiveness_prompt']:
                dimension = template_name.split('_')[0]  
                dimension_template_path = os.path.join(template_dir, f"{dimension}_prompt.csv")
                
                if os.path.exists(dimension_template_path):
                    print(f"加载{dimension}维度特定提示文件: {dimension_template_path}")
                    try:
                        df = pd.read_csv(dimension_template_path)
                        
                        if 'prompt' in df.columns and len(df) > 0:
                            template_content = df['prompt'].iloc[0]
                            if template_content and not pd.isna(template_content):
                                print(f"已成功从{dimension}_prompt.csv加载提示模板")
                                return template_content
                        
                        if 'content' in df.columns and len(df) > 0:
                            template_content = df['content'].iloc[0]
                            if template_content and not pd.isna(template_content):
                                print(f"已成功从{dimension}_prompt.csv加载提示模板")
                                return template_content
                            
                    except Exception as e:
                        print(f"读取{dimension}维度特定提示文件时出错: {e}")
            
            if template_name.endswith('_scoring') or template_name in ['flexibility_prompt', 'fluency_prompt', 'effectiveness_prompt']:
                template_path = os.path.join(template_dir, 'scoring_prompts.csv')
            elif template_name.endswith('_reference') or template_name == 'scoring_reference':
                template_path = os.path.join(template_dir, 'scoring_reference_prompts.csv')
            elif template_name.endswith('_followup') or template_name == 'followup_question':
                template_path = os.path.join(template_dir, 'followup_prompts.csv')
            elif template_name.endswith('_report') or template_name == 'summary_report':
                template_path = os.path.join(template_dir, 'report_prompts.csv')
                
            if not os.path.exists(template_path):
                print(f"警告: 提示模板文件 {template_path} 不存在，使用默认模板路径 {self.prompt_template_path}")
                template_path = self.prompt_template_path
                
            found_template = False
            try:
                df = pd.read_csv(template_path)
                
                if 'name' in df.columns and template_name in df['name'].values:
                    template_row = df[df['name'] == template_name]
                    if 'content' in df.columns and not template_row['content'].isnull().all():
                        template_content = template_row['content'].iloc[0]
                        if template_content and not pd.isna(template_content):
                            found_template = True
                            return template_content
                
                if not found_template and template_path != self.prompt_template_path:
                    try:
                        df_main = pd.read_csv(self.prompt_template_path)
                        if 'name' in df_main.columns and template_name in df_main['name'].values:
                            template_row = df_main[df_main['name'] == template_name]
                            if 'content' in df_main.columns and not template_row['content'].isnull().all():
                                template_content = template_row['content'].iloc[0]
                                if template_content and not pd.isna(template_content):
                                    found_template = True
                                    return template_content
                    except Exception as e:
                        print(f"读取主提示文件时出错: {e}")
            except Exception as e:
                print(f"读取提示模板文件时出错: {e}")
            
            if not found_template:
                print(f"警告: 未找到名为 '{template_name}' 的提示模板，使用默认提示")
                
                default_templates = {
                    'flexibility_prompt': "你是一位创新思维评估专家，专注于评估学生回答的灵活性。灵活性是指能从多个角度思考问题，提出多种不同解决方案的能力。请对学生的回答进行评分（0-10分）并提供详细反馈。",
                    'fluency_prompt': "你是一位创新思维评估专家，专注于评估学生回答的流畅性。流畅性是指能够快速、大量产生相关想法的能力。请对学生的回答进行评分（0-10分）并提供详细反馈。",
                    'effectiveness_prompt': "你是一位创新思维评估专家，专注于评估学生回答的有效性。有效性是指解决方案的实用性、可行性和价值。请对学生的回答进行评分（0-10分）并提供详细反馈。",
                    'scoring_reference': "请为{dimension}维度的问题'{question}'设计评分标准，分别对应2分、4分、6分和8分水平。",
                    'exploratory_evaluation': "请根据以下评分参考标准，评估学生在{dimension}维度的回答:\n\n问题: {question}\n\n学生回答: {response}\n\n评分参考标准:\n{scoring_references}\n\n请给出分数(0-10分)和详细反馈。",
                    'followup_question': "请基于学生的回答，生成一个针对{dimension}维度的跟进问题，以进一步评估学生的创新思维能力。",
                    'guidance_followup': "学生在{dimension}维度的表现需要引导。请基于以下对话历史，设计一个引导性的问题，帮助学生展示更多的创新思维能力:\n\n{dialogue_history}",
                    'summary_report': "请根据学生ID:{student_id}在灵活性({flexibility_score}分)、流畅性({fluency_score}分)和有效性({effectiveness_score}分)三个维度的表现，生成一份全面的评估报告。",
                    'feedback_analysis': "请分析学生ID:{student_id}在创新思维评估中的表现，特别关注成功和失败反馈记录，以生成针对性的学习建议。"
                }
                
                if template_name in default_templates:
                    return default_templates[template_name]
                elif 'followup' in template_name:
                    return default_templates['followup_question']
                elif 'reference' in template_name:
                    return default_templates['scoring_reference']
                elif 'report' in template_name:
                    return default_templates['summary_report']
                else:
                    return f"请评估学生在{{{template_name.split('_')[0]}}}维度的表现，并给出分数(0-10分)和反馈。"
                
        except Exception as e:
            print(f"加载提示模板时发生错误: {e}")
            return "请评估学生的回答并给出分数和反馈。"
    
    def select_exploratory_question(self, dimension):
        """
        从题库中选择适中难度的试探性试题
        选择难度值在4-6之间的题目，或返回默认问题
        
        参数:
            dimension (str): 需要评估的维度
            
        返回:
            str/dict: 问题内容或包含问题内容的字典
        """
        try:
            candidates = self.question_bank.get(dimension, [])
            
            if not candidates:
                print(f"警告: {dimension}维度没有可用问题，使用默认问题")
                default_questions = {
                    'flexibility': '如何解决城市交通拥堵问题？请提出至少三种不同的解决方案。',
                    'fluency': '请列出尽可能多的圆形物体。',
                    'effectiveness': '设计一个解决方案来减少塑料污染，考虑其可行性和经济效益。'
                }
                return default_questions.get(dimension, '请提出一个创新的解决方案来解决日常生活中的问题。')
                
            medium_difficulty_questions = []
            for q in candidates:
                if isinstance(q, dict) and 'difficulty' in q and 4 <= q['difficulty'] <= 6:
                    medium_difficulty_questions.append(q)
                elif isinstance(q, dict) and 'question' in q:
                    medium_difficulty_questions.append(q)
            
            if not medium_difficulty_questions and candidates:
                medium_difficulty_questions = candidates
            
            if not medium_difficulty_questions:
                print(f"警告: 在{dimension}维度中未找到适用问题，使用默认问题。")
                default_questions = {
                    'flexibility': '如何解决城市交通拥堵问题？请提出至少三种不同的解决方案。',
                    'fluency': '请列出尽可能多的圆形物体。',
                    'effectiveness': '设计一个解决方案来减少塑料污染，考虑其可行性和经济效益。'
                }
                return default_questions.get(dimension, '请提出一个创新的解决方案来解决日常生活中的问题。')
            
            selected = random.choice(medium_difficulty_questions)
            
            if isinstance(selected, dict):
                if 'question' in selected:
                    return selected['question']
                return selected
            return selected
            
        except Exception as e:
            print(f"选择问题时出错: {e}")
            default_questions = {
                'flexibility': '如何解决城市交通拥堵问题？请提出至少三种不同的解决方案。',
                'fluency': '请列出尽可能多的圆形物体。',
                'effectiveness': '设计一个解决方案来减少塑料污染，考虑其可行性和经济效益。'
            }
            return default_questions.get(dimension, '请提出一个创新的解决方案来解决日常生活中的问题。')

    def generate_scoring_reference(self, dimension, question):
        """为开放性问题生成评分参考标准"""
        template = self.load_prompt_template("scoring_reference")
        
        prompt = self.format_prompt(template,
            dimension=dimension,
            question=question["question"]
        )
        
        try:
            response = self.llm.query(prompt)
            
            score_standards = {}
            
            score_2_match = re.search(r"分数2标准[：:](.*?)(?=分数4标准[：:]|$)", response, re.DOTALL)
            if score_2_match:
                score_standards[2] = score_2_match.group(1).strip()
            
            score_4_match = re.search(r"分数4标准[：:](.*?)(?=分数6标准[：:]|$)", response, re.DOTALL)
            if score_4_match:
                score_standards[4] = score_4_match.group(1).strip()
            
            score_6_match = re.search(r"分数6标准[：:](.*?)(?=分数8标准[：:]|$)", response, re.DOTALL)
            if score_6_match:
                score_standards[6] = score_6_match.group(1).strip()
            
            score_8_match = re.search(r"分数8标准[：:](.*?)(?=$)", response, re.DOTALL)
            if score_8_match:
                score_standards[8] = score_8_match.group(1).strip()
            
            if len(score_standards) < 4:
                print("警告：未能提取完整的评分标准，使用默认值补充")
                if 2 not in score_standards:
                    score_standards[2] = "回答简单，缺乏创新性，未能充分回应问题"
                if 4 not in score_standards:
                    score_standards[4] = "回答基本合理，有一定创新性，但缺乏深度思考"
                if 6 not in score_standards:
                    score_standards[6] = "回答创新性良好，思路清晰，有一定实用价值"
                if 8 not in score_standards:
                    score_standards[8] = "回答高度创新，思路深入，具有很强的实用价值和可行性"
            
            return score_standards
        except Exception as e:
            print(f"生成评分参考标准时出错: {e}")
            return {
                2: "回答简单，缺乏创新性，未能充分回应问题",
                4: "回答基本合理，有一定创新性，但缺乏深度思考",
                6: "回答创新性良好，思路清晰，有一定实用价值",
                8: "回答高度创新，思路深入，具有很强的实用价值和可行性"
            }

    def format_prompt(self, prompt_template, **kwargs):
        """
        格式化提示模板，将变量替换为具体值
        处理可能出现的空值或格式错误
        """
        try:
            for key, value in kwargs.items():
                if value is None:
                    kwargs[key] = ""  
                elif not isinstance(value, str):
                    kwargs[key] = str(value)  
            
            if not prompt_template or not isinstance(prompt_template, str):
                print("警告: 提示模板无效或为空，使用默认提示")
                dimension = kwargs.get('dimension', '创新思维')
                if 'question' in kwargs and 'response' in kwargs:
                    return f"请评估学生在{dimension}维度的回答。\n问题:{kwargs.get('question', '')}\n回答:{kwargs.get('response', '')}"
                else:
                    return f"请评估学生在{dimension}维度的表现，并给出分数和反馈。"
            
            try:
                return prompt_template.format(**kwargs)
            except KeyError as e:
                missing_key = str(e).strip("'")
                print(f"警告: 格式化提示时缺少变量 {missing_key}，使用空字符串替代")
                kwargs[missing_key] = ""  
                return prompt_template.format(**kwargs)  
        except Exception as e:
            print(f"格式化提示时发生错误: {e}")
            dimension = kwargs.get('dimension', '创新思维')
            return f"请评估学生在{dimension}维度的表现，并给出分数和反馈。"

    def extract_score_and_feedback(self, evaluation_text):
        """
        从评估文本中提取分数和反馈
        
        参数:
            evaluation_text (str): 包含评估结果的文本
            
        返回:
            tuple: (分数, 反馈)
        """
        try:
            score_patterns = [
                r'分数[:：]\s*(\d+(?:\.\d+)?)',
                r'得分[:：]\s*(\d+(?:\.\d+)?)',
                r'评分[:：]\s*(\d+(?:\.\d+)?)',
                r'score[:：]\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*[/／]\s*10',
                r'(\d+(?:\.\d+)?)\s*分'
            ]
            
            score = None
            for pattern in score_patterns:
                match = re.search(pattern, evaluation_text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        break
                    except ValueError:
                        continue
            
            if score is None:
                if re.search(r'非常差|极差|terrible', evaluation_text, re.IGNORECASE):
                    score = 1.0
                elif re.search(r'差|poor', evaluation_text, re.IGNORECASE):
                    score = 3.0
                elif re.search(r'一般|中等|average', evaluation_text, re.IGNORECASE):
                    score = 5.0
                elif re.search(r'良好|good', evaluation_text, re.IGNORECASE):
                    score = 7.0
                elif re.search(r'优秀|excellent', evaluation_text, re.IGNORECASE):
                    score = 9.0
                else:
                    score = 5.0
                    
            score = max(0, min(10, score))
            
            feedback_patterns = [
                r'反馈[:：](.*)',
                r'feedback[:：](.*)',
                r'评价[:：](.*)',
                r'分析[:：](.*)'
            ]
            
            feedback = None
            for pattern in feedback_patterns:
                match = re.search(pattern, evaluation_text, re.DOTALL)
                if match:
                    feedback = match.group(1).strip()
                    break
            
            if feedback is None:
                lines = evaluation_text.split('\n')
                feedback_lines = []
                for line in lines:
                    if not any(re.search(p, line) for p in score_patterns):
                        feedback_lines.append(line)
                feedback = '\n'.join(feedback_lines).strip()
            
            return score, feedback
        except Exception as e:
            print(f"提取分数和反馈时出错: {e}")
            return 5.0, "无法提取有效反馈"
            
    def save_evaluation_results(self, results):
        """
        保存单个维度的评估结果
        
        参数:
            results (dict): 评估结果
        """
        try:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
            os.makedirs(results_dir, exist_ok=True)
            
            filename = f"{results['dimension']}_{results['student_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            print(f"已保存评估结果: {filepath}")
        except Exception as e:
            print(f"保存评估结果时出错: {e}")
            
    def save_final_results(self, results):
        """
        保存最终的评估结果
        
        参数:
            results (dict): 完整评估结果
        """
        try:
            report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_reports")
            os.makedirs(report_dir, exist_ok=True)
            
            filename = f"full_report_{results['student_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(report_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            print(f"已保存完整评估报告: {filepath}")
            
            txt_filename = f"report_{results['student_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            txt_filepath = os.path.join(report_dir, txt_filename)
            
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(f"创新思维能力评估报告\n")
                f.write(f"====================\n\n")
                f.write(f"学生ID: {results['student_id']}\n")
                f.write(f"评估时间: {results['timestamp']}\n")
                f.write(f"总体得分: {results['overall_score']:.2f}/10\n\n")
                
                f.write("各维度得分\n----------\n")
                for dimension, result in results['dimensions'].items():
                    score = result.get('score', 0)
                    f.write(f"{dimension}: {score:.2f}/10\n")
                
                f.write("\n详细评估报告\n============\n\n")
                f.write(results['report'])
                
            print(f"已生成用户友好报告: {txt_filepath}")
        except Exception as e:
            print(f"保存最终结果时出错: {e}")
            
    def generate_report(self, evaluation_results):
        """
        生成整体评估报告
        
        参数:
            evaluation_results (dict): 完整评估结果
            
        返回:
            str: 评估报告文本
        """
        try:

            student_id = evaluation_results['student_id']
            
            dimension_scores = {}
            for dimension, result in evaluation_results['dimensions'].items():
                dimension_scores[dimension] = result.get('score', 0)
            
            report_prompt_name = "summary_report"
            report_prompt_template = self.load_prompt_template(report_prompt_name)
            
            prompt_vars = {
                'student_id': student_id,
                'overall_score': evaluation_results['overall_score'],
                'flexibility_score': dimension_scores.get('flexibility', 0),
                'fluency_score': dimension_scores.get('fluency', 0),
                'effectiveness_score': dimension_scores.get('effectiveness', 0),
            }
            
            for dimension, result in evaluation_results['dimensions'].items():
                prompt_vars[f'{dimension}_feedback'] = result.get('feedback', '无反馈数据')
            
            report_prompt = self.format_prompt(report_prompt_template, **prompt_vars)
            
            try:
                report = self.llm.query(report_prompt)
                return report
            except Exception as e:
                print(f"生成报告时出错: {e}")
                
                simple_report = f"""学生{student_id}的创新思维评估报告
                
总体得分: {evaluation_results['overall_score']:.2f}/10

各维度得分:
"""
                for dimension, score in dimension_scores.items():
                    simple_report += f"- {dimension}: {score:.2f}/10\n"
                    
                simple_report += "\n详细反馈:\n"
                for dimension, result in evaluation_results['dimensions'].items():
                    simple_report += f"\n{dimension}维度:\n"
                    simple_report += f"{result.get('feedback', '无反馈数据')}\n"
                
                return simple_report
        except Exception as e:
            print(f"生成报告过程中出错: {e}")
            return f"无法生成评估报告。错误: {str(e)}"

    def load_question_bank(self):

        try:
            question_bank = {
                'flexibility': [],
                'fluency': [],
                'effectiveness': []
            }
            
            if os.path.exists(self.question_bank_path):
                print(f"加载主问题库: {self.question_bank_path}")
                file_ext = os.path.splitext(self.question_bank_path)[1].lower()
                
                try:
                    if file_ext == '.json':
                        with open(self.question_bank_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict):
                            for dimension in ['flexibility', 'fluency', 'effectiveness']:
                                if dimension in data and isinstance(data[dimension], list):
                                    question_bank[dimension] = data[dimension]
                                    print(f"从主题库加载{dimension}维度问题: {len(data[dimension])}题")
                    elif file_ext == '.csv':
                        df = pd.read_csv(self.question_bank_path)
                        if 'dimension' in df.columns and 'question' in df.columns:
                            for dimension in ['flexibility', 'fluency', 'effectiveness']:
                                questions = df[df['dimension'] == dimension]['question'].tolist()
                                if questions:
                                    question_bank[dimension] = [{'question': q, 'dimension': dimension} for q in questions]
                                    print(f"从主题库加载{dimension}维度问题: {len(questions)}题")
                except Exception as e:
                    print(f"加载主题库时出错: {str(e)}")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            dimension_files = {
                'flexibility': 'question_flex.json',
                'fluency': 'question_flu.json',
                'effectiveness': 'question_eff.json'
            }
            
            for dimension, filename in dimension_files.items():
                file_path = os.path.join(current_dir, filename)
                if os.path.exists(file_path):
                    try:
                        print(f"加载{dimension}维度问题库: {file_path}")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict):
                            if dimension in data and isinstance(data[dimension], list):
                                question_bank[dimension] = data[dimension]
                                print(f"从独立题库加载{dimension}维度问题: {len(data[dimension])}题")
                            elif len(data) > 0 and isinstance(list(data.values())[0], list):
                                first_key = list(data.keys())[0]
                                question_bank[dimension] = data[first_key]
                                print(f"从独立题库加载{dimension}维度问题: {len(data[first_key])}题")
                        elif isinstance(data, list):
                            question_bank[dimension] = data
                            print(f"从独立题库加载{dimension}维度问题: {len(data)}题")
                    except Exception as e:
                        print(f"加载{dimension}维度独立题库时出错: {str(e)}")
            
            for dimension in ['flexibility', 'fluency', 'effectiveness']:
                if not question_bank[dimension]:
                    print(f"警告: 在问题库中未找到 {dimension} 维度的问题。使用默认问题。")
                    question_bank[dimension] = self._create_default_questions()[dimension]
            
            return question_bank
        except Exception as e:
            print(f"加载问题库时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return self._create_default_questions()
    
    def _create_default_questions(self):
        """创建默认问题，当问题库加载失败时使用"""
        return {
            'flexibility': [
                {
                    'question': '如何解决城市交通拥堵问题？请提出至少三种不同的解决方案。',
                    'difficulty': 5,
                    'dimension': 'flexibility'
                },
                {
                    'question': '一个空塑料瓶可以有哪些创新的重复使用方法？请尽可能多地列举。',
                    'difficulty': 4,
                    'dimension': 'flexibility'
                }
            ],
            'fluency': [
                {
                    'question': '请在一分钟内列出尽可能多的圆形物体。',
                    'difficulty': 4,
                    'dimension': 'fluency'
                },
                {
                    'question': '如果互联网突然永久消失，会产生哪些后果？请尽可能详细地描述。',
                    'difficulty': 6,
                    'dimension': 'fluency'
                }
            ],
            'effectiveness': [
                {
                    'question': '设计一个解决方案来减少塑料污染，考虑其可行性和经济效益。',
                    'difficulty': 7,
                    'dimension': 'effectiveness'
                },
                {
                    'question': '如何提高远程教育的效果？请提出一个详细、可行的方案。',
                    'difficulty': 5,
                    'dimension': 'effectiveness'
                }
            ]
        }

    def add_msg_to_dialog_history(self, dimension, message):
        """
        向特定维度的对话历史添加消息
        
        参数:
            dimension (str): 维度名称 ('flexibility', 'fluency', 'effectiveness')
            message (dict): 消息字典，包含'role'和'content'键
        """
        if dimension in self.dialog_history_by_dimension:
            if len(self.dialog_history_by_dimension[dimension]) >= self.K * 2:  # 每轮对话有用户和助手两条消息
                self.dialog_history_by_dimension[dimension] = self.dialog_history_by_dimension[dimension][2:]
            
            self.dialog_history_by_dimension[dimension].append(message)
        else:
            print(f"警告: 维度 {dimension} 不存在于对话历史记录中")

# 将CreativeThinkingEvaluator作为Evaluator导出
Evaluator = CreativeThinkingEvaluator