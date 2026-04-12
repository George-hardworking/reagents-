"""
Analysis module for evaluating experimental results.
This module loads data from record files and performs statistical analysis.
"""

from core.record import load_record
from typing import List, Dict, Any
import os
from collections import defaultdict


class ExperimentAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    def load_all_samples(self, sample_ids: List[str] = None) -> List[Dict]:
        """加载所有或指定的样本数据"""
        if sample_ids:
            samples = []
            for sample_id in sample_ids:
                sample_data = load_record(
                    folder=self.results_dir,
                    savename=sample_id,
                    from_block=True
                )
                if sample_data:
                    samples.append(sample_data)
            return samples
        else:
            # 加载所有样本数据
            samples = []
            block_files = [f for f in os.listdir(self.results_dir) if f.startswith("block_results_")]
            for block_file in block_files:
                block_path = os.path.join(self.results_dir, block_file)
                block_data = load_record(folder=self.results_dir, savename=block_file.replace(".json", ""), from_block=True)
                if block_data:
                    samples.append(block_data)
            return samples

    def calculate_accuracy(self, samples: List[Dict], model_name: str = "agent1") -> float:
        """计算指定模型的准确率（基于overview中的正确性信息）"""
        correct_count = 0
        total_count = 0
        
        for sample in samples:
            for sample_id, sample_data in sample.items():
                # 从overview中获取正确性信息
                correct_info = sample_data.get("overview", {}).get("correct", {})
                
                if model_name == "agent1":
                    is_correct = correct_info.get("agent1", False)
                elif model_name == "agent2":
                    is_correct = correct_info.get("agent2", False)
                elif model_name == "model":
                    is_correct = correct_info.get("model_correct", False)
                else:
                    # 如果overview中没有正确性信息，回退到原来的方法
                    final_answer = self._extract_final_answer(sample_data, model_name)
                    gold_answer = sample_data["overview"]["gold_answer"]
                    is_correct = self._is_answer_correct(final_answer, gold_answer)
                
                if is_correct:
                    correct_count += 1
                total_count += 1
            
        return correct_count / total_count if total_count > 0 else 0

    def _extract_final_answer(self, sample_data: Dict, model_name: str) -> str:
        """从样本数据中提取最终答案"""
        # 从最后一轮的 details 中提取答案
        details = sample_data["details"]
        last_round = f"round_{len(details)}"
        if last_round in details:
            round_data = details[last_round]
            return round_data.get(f"{model_name}_answer", "")
        return ""

    def _is_answer_correct(self, pred_answer: str, gold_answer: str) -> bool:
        """判断答案是否正确（调用您现有的评估逻辑）"""
        from evaluation.math_normalize import normalize_answer
        from evaluation.grader import grade_answer
        from evaluation.eval_math import parse_answer
        
        pred_normalized = normalize_answer(parse_answer(pred_answer))
        gold_normalized = normalize_answer(parse_answer(gold_answer))
        return grade_answer(pred_normalized, gold_normalized)

    def generate_report(self, samples: List[Dict]) -> Dict[str, Any]:
        """生成完整的实验报告"""
        total_samples = sum(len(sample) for sample in samples)
        
        report = {
            "total_samples": total_samples,
            "agent1_accuracy": self.calculate_accuracy(samples, "agent1"),
            "agent2_accuracy": self.calculate_accuracy(samples, "agent2"),
            "model_accuracy": self.calculate_accuracy(samples, "model"),
            "token_usage_summary": self._calculate_token_summary(samples),
            "runtime_summary": self._calculate_runtime_summary(samples)
        }
        return report

    def _calculate_token_summary(self, samples: List[Dict]) -> Dict:
        """计算 token 使用汇总"""
        total_input_tokens = {"agent1": 0, "agent2": 0}
        total_output_tokens = {"agent1": 0, "agent2": 0}
        
        for sample in samples:
            for sample_id, sample_data in sample.items():
                for round_key, round_data in sample_data.get("details", {}).items():
                    total_input_tokens["agent1"] += round_data.get("input_tokens", {}).get("agent1", 0)
                    total_input_tokens["agent2"] += round_data.get("input_tokens", {}).get("agent2", 0)
                    total_output_tokens["agent1"] += round_data.get("output_tokens", {}).get("agent1", 0)
                    total_output_tokens["agent2"] += round_data.get("output_tokens", {}).get("agent2", 0)
        
        return {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": {
                "agent1": total_input_tokens["agent1"] + total_output_tokens["agent1"],
                "agent2": total_input_tokens["agent2"] + total_output_tokens["agent2"]
            }
        }

    def _calculate_runtime_summary(self, samples: List[Dict]) -> Dict:
        """计算运行时间汇总"""
        total_runtime = 0
        total_count = 0
        
        for sample in samples:
            for sample_id, sample_data in sample.items():
                total_runtime += sample_data.get("overview", {}).get("total_runtime", 0)
                total_count += 1
        
        return {
            "total_runtime": total_runtime,
            "average_runtime": total_runtime / total_count if total_count > 0 else 0
        }


def analyze_experiment_results(results_dir: str = "results", sample_ids: List[str] = None):
    """生成分析报告"""
    analyzer = ExperimentAnalyzer(results_dir)
    samples = analyzer.load_all_samples(sample_ids)
    return analyzer.generate_report(samples)