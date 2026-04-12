"""
File: statistics.py
      Statistical Analysis Module for Agent Experiment Results
File Manager: Kaibiao ZHU
Description: This module provides unified statistical analysis for both single-agent
             and multi-agent debate experiments, calculating accuracy rates, runtime
             statistics, and token usage metrics.
####################################################################################
"""

import os
import json
import glob
import tiktoken
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

from evaluation.grader import grade_answer
from evaluation.eval_math import parse_answer
from evaluation.math_normalize import normalize_answer


class ExperimentStatistics:
    """统一的实验统计分析类"""

    def __init__(self, results_directory: str, output_directory: str = None):
        """
        初始化统计分析器

        Args:
            results_directory: 结果文件所在目录
            output_directory: 输出目录（可选，默认为结果目录下的 statistic_summary）
        """
        self.results_directory = results_directory
        self.output_directory = output_directory or os.path.join(
            os.path.dirname(results_directory), "statistic_summary"
        )
        self.samples = {}
        self.experiment_type = None
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    def load_data(self) -> None:
        """加载所有 JSON 数据文件"""
        json_files = glob.glob(os.path.join(self.results_directory, "*.json"))
        print(f"Found {len(json_files)} JSON files in {self.results_directory}")

        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_data.append(data)
                    print(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # 提取所有样本
        for data in all_data:
            for key, value in data.items():
                if key.startswith("sample_"):
                    self.samples[key] = value

        print(f"Total samples extracted: {len(self.samples)}")

        # 检测实验类型
        self._detect_experiment_type()

    def _detect_experiment_type(self) -> None:
        """检测实验类型（单智能体或多智能体）"""
        if not self.samples:
            self.experiment_type = "multi"
            return

        first_sample = next(iter(self.samples.values()))
        details = first_sample.get("details", {})

        # 检查是否有 round_1
        if "round_1" in details:
            # 进一步检查是否有 agent2
            if "agent2_content" in details["round_1"]:
                self.experiment_type = "multi"
            else:
                self.experiment_type = "single"
        else:
            self.experiment_type = "multi"

        print(
            f"Detected experiment type: {'Single-Agent' if self.experiment_type == 'single' else 'Multi-Agent'}"
        )

    def calculate_accuracy(self) -> Dict[str, float]:
        """计算准确率统计"""
        total_samples = len(self.samples)
        agent1_correct_count = 0
        agent2_correct_count = 0
        both_correct_count = 0

        print("Evaluating sample correctness...")

        for sample_id, sample in self.samples.items():
            agent1_correct, agent2_correct = self._evaluate_sample(sample)

            if agent1_correct:
                agent1_correct_count += 1
            if agent2_correct:
                agent2_correct_count += 1
            if agent1_correct and agent2_correct:
                both_correct_count += 1

        # 单智能体和多智能体都返回相同格式
        return {
            "total_samples": total_samples,
            "agent1_correct": agent1_correct_count,
            "model_correct": both_correct_count,
            "agent1_accuracy": (
                agent1_correct_count / total_samples if total_samples > 0 else 0
            ),
            "model_accuracy": (
                both_correct_count / total_samples if total_samples > 0 else 0
            ),
        }

    def calculate_accuracy_by_rounds(self) -> Dict[str, Dict[str, float]]:
        """计算各轮次的准确率"""
        print("Evaluating sample correctness by rounds...")

        round_stats = {}
        for round_num in [1, 2, 3]:
            round_key = f"round_{round_num}"
            round_stats[round_key] = {
                "total_samples": len(self.samples),
                "agent1_correct": 0,
                "model_correct": 0,
            }

        for sample_id, sample in self.samples.items():
            for round_num in [1, 2, 3]:
                agent1_correct, agent2_correct = self._evaluate_sample_by_round(
                    sample, round_num
                )
                round_key = f"round_{round_num}"

                if agent1_correct:
                    round_stats[round_key]["agent1_correct"] += 1
                if agent1_correct and agent2_correct:
                    round_stats[round_key]["model_correct"] += 1

        # 计算百分比
        for round_key in round_stats:
            stats = round_stats[round_key]
            total = stats["total_samples"]
            if total > 0:
                stats["agent1_accuracy"] = stats["agent1_correct"] / total
                stats["model_accuracy"] = stats["model_correct"] / total
            else:
                stats["agent1_accuracy"] = 0
                stats["model_accuracy"] = 0

        return round_stats

    def calculate_runtime_statistics(self) -> Dict[str, float]:
        """计算运行时间统计"""
        total_overview_runtime = 0
        total_agent1_runtime = 0
        total_agent2_runtime = 0
        total_rounds = 0
        total_agent1_round1_runtime = 0
        total_agent1_round2_runtime = 0
        total_agent1_round3_runtime = 0
        total_agent2_round1_runtime = 0
        total_agent2_round2_runtime = 0
        total_agent2_round3_runtime = 0

        print("Calculating runtime statistics...")

        for sample_id, sample in self.samples.items():
            overview_runtime = sample["overview"].get("total_runtime", 0)
            total_overview_runtime += overview_runtime

            details = sample["details"]

            for round_key, round_data in details.items():
                if round_key.startswith("round_"):
                    total_rounds += 1

                    if self.experiment_type == "single":
                        # 单智能体只有一个 agent，兼容两种数据结构
                        if "agent1_content" in round_data:
                            agent_runtime = round_data["agent1_content"].get(
                                "runtime", 0
                            )
                        elif "content" in round_data:
                            agent_runtime = round_data["content"].get("runtime", 0)
                        else:
                            agent_runtime = 0

                        total_agent1_runtime += agent_runtime
                        if round_key == "round_1":
                            total_agent1_round1_runtime += agent_runtime
                    else:
                        # 多智能体有两个 agent
                        agent1_runtime = round_data["agent1_content"].get("runtime", 0)
                        agent2_runtime = round_data["agent2_content"].get("runtime", 0)
                        total_agent1_runtime += agent1_runtime
                        total_agent2_runtime += agent2_runtime

                        if round_key == "round_1":
                            total_agent1_round1_runtime += agent1_runtime
                            total_agent2_round1_runtime += agent2_runtime
                        elif round_key == "round_2":
                            total_agent1_round2_runtime += agent1_runtime
                            total_agent2_round2_runtime += agent2_runtime
                        elif round_key == "round_3":
                            total_agent1_round3_runtime += agent1_runtime
                            total_agent2_round3_runtime += agent2_runtime

        # 单智能体和多智能体都返回相同格式
        return {
            "total_overview_runtime": total_overview_runtime,
            "total_agent1_runtime": total_agent1_runtime,
            "total_rounds": total_rounds,
            "total_agent1_round1_runtime": total_agent1_round1_runtime,
            "total_agent1_round2_runtime": total_agent1_round2_runtime,
            "total_agent1_round3_runtime": total_agent1_round3_runtime,
        }

    def calculate_token_statistics(self) -> Dict[str, int]:
        """计算 token 使用统计"""
        total_agent1_tokens = 0
        total_agent2_tokens = 0
        total_agent1_round1_tokens = 0
        total_agent1_round2_tokens = 0
        total_agent1_round3_tokens = 0
        total_agent2_round1_tokens = 0
        total_agent2_round2_tokens = 0
        total_agent2_round3_tokens = 0

        print("Calculating token statistics...")

        for sample_id, sample in self.samples.items():
            details = sample["details"]

            for round_key, round_data in details.items():
                if round_key.startswith("round_"):
                    # 兼容两种数据结构
                    if "agent1_content" in round_data:
                        agent1_response = round_data["agent1_content"].get(
                            "response", ""
                        )
                    elif "content" in round_data:
                        agent1_response = round_data["content"].get("response", "")
                    else:
                        agent1_response = ""

                    agent1_tokens = len(self.encoding.encode(agent1_response))
                    total_agent1_tokens += agent1_tokens

                    if self.experiment_type == "multi":
                        agent2_response = round_data["agent2_content"].get(
                            "response", ""
                        )
                        agent2_tokens = len(self.encoding.encode(agent2_response))
                        total_agent2_tokens += agent2_tokens
                    else:
                        agent2_tokens = 0

                    if round_key == "round_1":
                        total_agent1_round1_tokens += agent1_tokens
                        total_agent2_round1_tokens += agent2_tokens
                    elif round_key == "round_2":
                        total_agent1_round2_tokens += agent1_tokens
                        total_agent2_round2_tokens += agent2_tokens
                    elif round_key == "round_3":
                        total_agent1_round3_tokens += agent1_tokens
                        total_agent2_round3_tokens += agent2_tokens

        # 单智能体和多智能体都返回相同格式
        return {
            "total_agent1_tokens": total_agent1_tokens,
            "total_tokens": total_agent1_tokens,
            "total_agent1_round1_tokens": total_agent1_round1_tokens,
            "total_agent1_round2_tokens": total_agent1_round2_tokens,
            "total_agent1_round3_tokens": total_agent1_round3_tokens,
        }

    def generate_report(
        self,
        accuracy_stats: Dict[str, float],
        runtime_stats: Dict[str, float],
        round_accuracy_stats: Dict[str, Dict[str, float]],
        token_stats: Dict[str, int],
    ) -> str:
        """生成统计报告"""
        title = (
            "Single-Agent" if self.experiment_type == "single" else "Multi-Agent Debate"
        )

        report = f"""
{'='*80}
{title} Results Statistical Analysis
{'='*80}

ACCURACY STATISTICS:
{'─'*40}
Total Samples: {accuracy_stats['total_samples']}
"""

        if self.experiment_type == "single":
            report += f"""
Agent Performance:
  • Correct Answers: {accuracy_stats['agent1_correct']} ({accuracy_stats['agent1_accuracy']:.4f})
  • Accuracy Rate: {accuracy_stats['agent1_accuracy']*100:.2f}%
"""
        else:
            report += f"""
Agent Performance:
  • Agent1 Correct: {accuracy_stats['agent1_correct']} ({accuracy_stats['agent1_accuracy']:.4f})
  • Agent2 Correct: {accuracy_stats['agent2_correct']} ({accuracy_stats['agent2_accuracy']:.4f})
  • Both Agents Correct: {accuracy_stats['model_correct']} ({accuracy_stats['model_accuracy']:.4f})

Individual Accuracy Rates:
  • Agent1 Accuracy: {accuracy_stats['agent1_accuracy']*100:.2f}%
  • Agent2 Accuracy: {accuracy_stats['agent2_accuracy']*100:.2f}%
  • Both Agents Accuracy: {accuracy_stats['model_accuracy']*100:.2f}%
"""

        # 多智能体添加各轮统计
        if self.experiment_type == "multi" and round_accuracy_stats:
            report += f"""
ACCURACY BY ROUNDS:
{'─'*40}"""
            for round_key in ["round_1", "round_2", "round_3"]:
                if round_key in round_accuracy_stats:
                    stats = round_accuracy_stats[round_key]
                    if stats["total_samples"] > 0:
                        report += f"""
{round_key.upper()} Statistics (Total: {stats['total_samples']} samples):
  • Agent1 Correct: {stats['agent1_correct']} ({stats['agent1_accuracy']:.4f}) - {stats['agent1_accuracy']*100:.2f}%
  • Agent2 Correct: {stats['agent2_correct']} ({stats['agent2_accuracy']:.4f}) - {stats['agent2_accuracy']*100:.2f}%
  • Both Agents Correct: {stats['model_correct']} ({stats['model_accuracy']:.4f}) - {stats['model_accuracy']*100:.2f}%"""

        report += f"""

RUNTIME STATISTICS:
{'─'*40}
Total Runtime Statistics:
  • Total Overview Runtime: {runtime_stats['total_overview_runtime']:.2f} seconds
"""

        if self.experiment_type == "single":
            avg_time = (
                runtime_stats["total_agent1_runtime"] / runtime_stats["total_rounds"]
                if runtime_stats["total_rounds"] > 0
                else 0
            )
            report += f"""  • Agent Runtime: {runtime_stats['total_agent1_runtime']:.2f} seconds
  • Total Samples: {runtime_stats['total_rounds']}
  • Average Time per Sample: {avg_time:.2f} seconds

Time Conversion:
  • Total Overview Runtime: {runtime_stats['total_overview_runtime']/3600:.2f} hours
  • Total Agent Runtime: {runtime_stats['total_agent1_runtime']/3600:.2f} hours
"""
        else:
            report += f"""  • Total Agent1 Runtime: {runtime_stats['total_agent1_runtime']:.2f} seconds
  • Total Agent2 Runtime: {runtime_stats['total_agent2_runtime']:.2f} seconds
  • Total Rounds: {runtime_stats['total_rounds']}

Agent1 Runtime by Rounds:
  • Total Agent1 Round1 Runtime: {runtime_stats['total_agent1_round1_runtime']:.2f} seconds
  • Total Agent1 Round2 Runtime: {runtime_stats['total_agent1_round2_runtime']:.2f} seconds
  • Total Agent1 Round3 Runtime: {runtime_stats['total_agent1_round3_runtime']:.2f} seconds

Agent2 Runtime by Rounds:
  • Total Agent2 Round1 Runtime: {runtime_stats['total_agent2_round1_runtime']:.2f} seconds
  • Total Agent2 Round2 Runtime: {runtime_stats['total_agent2_round2_runtime']:.2f} seconds
  • Total Agent2 Round3 Runtime: {runtime_stats['total_agent2_round3_runtime']:.2f} seconds

Time Conversion:
  • Total Overview Runtime: {runtime_stats['total_overview_runtime']/3600:.2f} hours
  • Total Agent1 Runtime: {runtime_stats['total_agent1_runtime']/3600:.2f} hours
  • Total Agent2 Runtime: {runtime_stats['total_agent2_runtime']/3600:.2f} hours
"""

        report += f"""
TOKEN USAGE STATISTICS:
{'─'*40}
"""

        if self.experiment_type == "single":
            avg_tokens = (
                token_stats["total_agent1_tokens"] / accuracy_stats["total_samples"]
                if accuracy_stats["total_samples"] > 0
                else 0
            )
            report += f"""Total Token Usage:
  • Agent Tokens: {token_stats['total_agent1_tokens']:,}
  • Average Tokens per Sample: {avg_tokens:.0f}
"""
        else:
            report += f"""Total Token Usage:
  • Total Agent1 Tokens: {token_stats['total_agent1_tokens']:,}
  • Total Agent2 Tokens: {token_stats['total_agent2_tokens']:,}
  • Total Combined Tokens: {token_stats['total_tokens']:,}

Agent1 Token Usage by Rounds:
  • Agent1 Round1 Tokens: {token_stats['total_agent1_round1_tokens']:,}
  • Agent1 Round2 Tokens: {token_stats['total_agent1_round2_tokens']:,}
  • Agent1 Round3 Tokens: {token_stats['total_agent1_round3_tokens']:,}

Agent2 Token Usage by Rounds:
  • Agent2 Round1 Tokens: {token_stats['total_agent2_round1_tokens']:,}
  • Agent2 Round2 Tokens: {token_stats['total_agent2_round2_tokens']:,}
  • Agent2 Round3 Tokens: {token_stats['total_agent2_round3_tokens']:,}
"""

        report += f"""
{'='*80}
"""
        return report

    def save_results(
        self,
        accuracy_stats: Dict[str, float],
        runtime_stats: Dict[str, float],
        round_accuracy_stats: Dict[str, Dict[str, float]],
        token_stats: Dict[str, int],
    ) -> str:
        """保存详细结果到 JSON 文件，严格遵循标准格式"""
        os.makedirs(self.output_directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}.json"
        output_path = os.path.join(self.output_directory, filename)

        # 按照标准格式构建结果字典（顺序很重要）
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 单智能体和多智能体都使用相同的字段名
        results["final_round_accuracy_statistics"] = accuracy_stats
        if round_accuracy_stats:
            results["round_by_round_accuracy_statistics"] = round_accuracy_stats

        results["runtime_statistics"] = runtime_stats
        results["token_usage_statistics"] = token_stats

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path

    def _evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, bool]:
        """评估样本的正确性"""
        try:
            gold_answer = sample["overview"]["gold_answer"]
            details = sample["details"]

            # 获取最后一轮
            total_rounds = sample["overview"]["total_rounds"]
            final_round_key = f"round_{total_rounds}"

            if final_round_key not in details:
                return False, False

            final_round = details[final_round_key]

            # 兼容两种数据结构：
            # 单智能体旧格式：round_1 -> content -> response
            # 单智能体新格式：round_1 -> agent1_content -> response
            # 多智能体格式：round_1 -> agent1_content/agent2_content -> response

            # 获取 agent1 答案
            if "agent1_content" in final_round:
                agent1_answer = final_round["agent1_content"]["response"]
            elif "content" in final_round:
                # 旧的单智能体格式
                agent1_answer = final_round["content"]["response"]
            else:
                return False, False

            # 获取 agent2 答案（如果是多智能体）
            if self.experiment_type == "multi":
                agent2_answer = final_round["agent2_content"]["response"]
            else:
                agent2_answer = agent1_answer  # 单智能体，agent2 答案与 agent1 相同

            # 标准化答案
            gold_parsed = parse_answer(gold_answer)
            gold_normalized = (
                normalize_answer(gold_parsed)
                if gold_parsed
                else normalize_answer(gold_answer)
            )

            agent1_parsed = parse_answer(agent1_answer)
            agent1_normalized = (
                normalize_answer(agent1_parsed)
                if agent1_parsed
                else normalize_answer(agent1_answer)
            )

            agent2_parsed = parse_answer(agent2_answer)
            agent2_normalized = (
                normalize_answer(agent2_parsed)
                if agent2_parsed
                else normalize_answer(agent2_answer)
            )

            # 评分
            agent1_correct = grade_answer(agent1_normalized, gold_normalized)
            agent2_correct = grade_answer(agent2_normalized, gold_normalized)

            return agent1_correct, agent2_correct

        except Exception as e:
            print(f"Error evaluating sample correctness: {e}")
            return False, False

    def _evaluate_sample_by_round(
        self, sample: Dict[str, Any], round_num: int
    ) -> Tuple[bool, bool]:
        """评估样本在特定轮次的正确性"""
        try:
            gold_answer = sample["overview"]["gold_answer"]
            details = sample["details"]
            round_key = f"round_{round_num}"

            if round_key not in details:
                return False, False

            round_data = details[round_key]

            # 兼容两种数据结构: 旧的 "content" 和新的 "agent1_content"
            if "agent1_content" in round_data:
                agent1_answer = round_data["agent1_content"]["response"]
                # 对于单智能体，agent2 也使用 agent1 的数据
                if "agent2_content" in round_data:
                    agent2_answer = round_data["agent2_content"]["response"]
                else:
                    agent2_answer = agent1_answer
            elif "content" in round_data:
                agent1_answer = round_data["content"]["response"]
                agent2_answer = agent1_answer  # 单智能体使用相同的答案
            else:
                return False, False

            gold_parsed = parse_answer(gold_answer)
            gold_normalized = (
                normalize_answer(gold_parsed)
                if gold_parsed
                else normalize_answer(gold_answer)
            )

            agent1_parsed = parse_answer(agent1_answer)
            agent1_normalized = (
                normalize_answer(agent1_parsed)
                if agent1_parsed
                else normalize_answer(agent1_answer)
            )

            agent2_parsed = parse_answer(agent2_answer)
            agent2_normalized = (
                normalize_answer(agent2_parsed)
                if agent2_parsed
                else normalize_answer(agent2_answer)
            )

            agent1_correct = grade_answer(agent1_normalized, gold_normalized)
            agent2_correct = grade_answer(agent2_normalized, gold_normalized)

            return agent1_correct, agent2_correct

        except Exception as e:
            print(f"Error evaluating sample correctness for round {round_num}: {e}")
            return False, False

    def run_analysis(self) -> Dict[str, Any]:
        """运行完整的统计分析"""
        print(f"\n{'='*50}")
        print(f"Statistical Analysis for Agent Experiment Results")
        print(f"{'='*50}\n")

        print(f"1. Loading JSON files from: {self.results_directory}")
        self.load_data()

        print(f"\n2. Calculating accuracy statistics...")
        accuracy_stats = self.calculate_accuracy()

        print(f"\n3. Calculating round-by-round accuracy statistics...")
        round_accuracy_stats = self.calculate_accuracy_by_rounds()

        print(f"\n4. Calculating runtime statistics...")
        runtime_stats = self.calculate_runtime_statistics()

        print(f"\n5. Calculating token usage statistics...")
        token_stats = self.calculate_token_statistics()

        print(f"\n6. Generating report...")
        report = self.generate_report(
            accuracy_stats, runtime_stats, round_accuracy_stats, token_stats
        )
        print(report)

        print(f"\n7. Saving detailed results...")
        output_file = self.save_results(
            accuracy_stats, runtime_stats, round_accuracy_stats, token_stats
        )
        print(f"Detailed results saved to: {output_file}")

        print("\nAnalysis complete!")

        return {
            "accuracy_stats": accuracy_stats,
            "runtime_stats": runtime_stats,
            "round_accuracy_stats": round_accuracy_stats,
            "token_stats": token_stats,
            "output_file": output_file,
        }
