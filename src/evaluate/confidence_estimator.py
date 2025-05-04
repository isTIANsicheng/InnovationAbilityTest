import numpy as np
from scipy.stats import norm

class ConfidenceEstimator:
    """可信度估计器，用于动态评估分数的可靠性"""
    
    def __init__(self, 
                initial_confidence=0.5,  # 初始可信度
                min_confidence=0.3,      # 最小可信度
                max_confidence=0.95,     # 最大可信度
                consistency_weight=0.6,  # 一致性权重
                response_quality_weight=0.4,  # 回答质量权重
                confidence_threshold=0.75  # 可信度阈值
                ):
        self.initial_confidence = initial_confidence
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.consistency_weight = consistency_weight
        self.response_quality_weight = response_quality_weight
        self.confidence_threshold = confidence_threshold
        
        # 每个维度的可信度记录
        self.dimension_confidence = {
            'flexibility': initial_confidence,
            'fluency': initial_confidence,
            'effectiveness': initial_confidence,
            'overall': initial_confidence
        }
        
        # 历史评分及其可信度
        self.score_history = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
        
        # 可信度历史
        self.confidence_history = {
            'flexibility': [],
            'fluency': [],
            'effectiveness': []
        }
    
    def update_confidence(self, dimension, new_score, response_quality, llm_uncertainty=None):
        """
        更新可信度
        
        参数:
            dimension: 评估维度
            new_score: 新的评分
            response_quality: LLM评估的回答质量 (0-1)
            llm_uncertainty: LLM自我评估的不确定性 (可选)
        """
        # 将新分数添加到历史记录
        self.score_history[dimension].append(new_score)
        
        # 1. 基于一致性的可信度计算
        consistency_confidence = self._calculate_consistency_confidence(dimension)
        
        # 2. 基于回答质量的可信度
        quality_confidence = response_quality
        
        # 3. 考虑LLM自我报告的不确定性（如果提供）
        if llm_uncertainty is not None:
            llm_confidence = 1.0 - llm_uncertainty
        else:
            llm_confidence = 0.7  # 默认值
        
        # 综合计算新的可信度
        new_confidence = (
            self.consistency_weight * consistency_confidence +
            self.response_quality_weight * quality_confidence * llm_confidence
        )
        
        # 平滑更新，避免突变
        current_confidence = self.dimension_confidence[dimension]
        updated_confidence = 0.7 * current_confidence + 0.3 * new_confidence
        
        # 确保可信度在合理范围内
        updated_confidence = max(self.min_confidence, min(self.max_confidence, updated_confidence))
        
        # 更新可信度
        self.dimension_confidence[dimension] = updated_confidence
        self.confidence_history[dimension].append(updated_confidence)
        
        return updated_confidence
    
    def _calculate_consistency_confidence(self, dimension):
        """计算基于历史分数一致性的可信度"""
        scores = self.score_history[dimension]
        
        if len(scores) < 3:
            # 数据点太少，返回中等可信度
            return 0.5
        
        # 计算趋势的一致性
        recent_scores = scores[-min(5, len(scores)):]  # 最近5个或更少
        
        # 计算方差 - 方差越小，一致性越高
        variance = np.var(recent_scores)
        
        # 线性趋势一致性
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        # 计算拟合优度
        if len(recent_scores) > 2:
            correlation = np.abs(np.corrcoef(x, y)[0, 1])
            # 相关性低说明数据波动，可信度低
            correlation_factor = correlation if not np.isnan(correlation) else 0.5
        else:
            correlation_factor = 0.5
            
        # 方差反映一致性，方差越小，一致性越高
        variance_factor = np.exp(-2 * variance)  # 转换为0-1范围
        
        # 综合因素
        consistency = 0.6 * variance_factor + 0.4 * correlation_factor
        
        return consistency
    
    def is_confidence_sufficient(self, dimension):
        """判断可信度是否足够高"""
        return self.dimension_confidence[dimension] >= self.confidence_threshold
    
    def adjust_score_by_confidence(self, dimension, raw_score):
        """根据可信度调整得分，可信度低时倾向于中间值"""
        confidence = self.dimension_confidence[dimension]
        
        # 可信度低时，分数向5分靠拢
        adjusted_score = confidence * raw_score + (1 - confidence) * 5.0
        
        return adjusted_score, confidence
    
    def update_overall_confidence(self):
        """更新总体可信度"""
        # 按维度重要性加权计算总体可信度
        self.dimension_confidence['overall'] = (
            0.33 * self.dimension_confidence['flexibility'] +
            0.33 * self.dimension_confidence['fluency'] +
            0.34 * self.dimension_confidence['effectiveness']
        )
        return self.dimension_confidence['overall']
    
    def get_confidence_assessment(self, dimension):
        """获取可信度评估报告"""
        confidence = self.dimension_confidence[dimension]
        
        if confidence < 0.4:
            return "低可信度", f"该维度的评估可信度较低({confidence:.2f})，建议增加更多评估轮次以提高可靠性。"
        elif confidence < 0.7:
            return "中等可信度", f"该维度的评估可信度中等({confidence:.2f})，结果具有一定参考价值。"
        else:
            return "高可信度", f"该维度的评估可信度较高({confidence:.2f})，结果可靠。"