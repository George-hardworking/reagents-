"""
File: analyze_debate_results.py
      Statistical Analysis Script for Multi-Agent Debate Results
File Manager: Kaibiao ZHU
Description: This script analyzes JSON result files from multi-agent debate experiments, calculating accuracy rates, runtime statistics, and other performance metrics.
####################################################################################
"""

import os
import json
import glob
import tiktoken

from pathlib import Path
from evaluation.grader import grade_answer
from typing import Dict, List, Any, Tuple
from evaluation.eval_math import parse_answer
from evaluation.math_normalize import normalize_answer


def load_json_files(directory: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the specified directory.

    Args:
        directory: Path to directory containing JSON files

    Returns:
        List of loaded JSON data
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Found {len(json_files)} JSON files in {directory}")

    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.append(data)
                print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return all_data


def extract_all_samples(json_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract all samples from the loaded JSON files.

    Args:
        json_data_list: List of JSON data dictionaries

    Returns:
        Dictionary with all samples combined
    """
    all_samples = {}

    for data in json_data_list:
        for key, value in data.items():
            if key.startswith("sample_"):
                all_samples[key] = value

    print(f"Total samples extracted: {len(all_samples)}")
    return all_samples


def evaluate_sample_correctness_by_round(
    sample: Dict[str, Any], round_num: int
) -> Tuple[bool, bool]:
    """
    Evaluate if a sample's agents answered correctly in a specific round.

    Args:
        sample: Sample data dictionary
        round_num: Round number to evaluate (1, 2, 3, etc.)

    Returns:
        Tuple of (agent1_correct, agent2_correct)
    """
    try:
        gold_answer = sample["overview"]["gold_answer"]

        rounds = sample["details"]
        round_key = f"round_{round_num}"

        if round_key not in rounds:
            return False, False

        round_data = rounds[round_key]

        agent1_answer = round_data["agent1_content"]["response"]
        agent2_answer = round_data["agent2_content"]["response"]

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


def evaluate_sample_correctness(sample: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluate if a sample's agents answered correctly.

    Args:
        sample: Sample data dictionary

    Returns:
        Tuple of (agent1_correct, agent2_correct)
    """
    try:

        gold_answer = sample["overview"]["gold_answer"]

        rounds = sample["details"]
        final_round_key = f"round_{sample['overview']['total_rounds']}"
        final_round = rounds[final_round_key]

        agent1_answer = final_round["agent1_content"]["response"]
        agent2_answer = final_round["agent2_content"]["response"]

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
        print(f"Error evaluating sample correctness: {e}")
        return False, False


def calculate_accuracy_by_rounds(
    samples: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy rates for both agents by rounds.

    Args:
        samples: Dictionary of all samples

    Returns:
        Dictionary with accuracy statistics for each round
    """
    print("Evaluating sample correctness by rounds...")

    round_stats = {}
    for round_num in [1, 2, 3]:
        round_stats[f"round_{round_num}"] = {
            "total_samples": 0,
            "agent1_correct": 0,
            "agent2_correct": 0,
            "model_correct": 0,
        }

    for sample_id, sample in samples.items():
        total_rounds = sample["overview"]["total_rounds"]

        for round_num in [1, 2, 3]:
            if round_num <= total_rounds:
                agent1_correct, agent2_correct = evaluate_sample_correctness_by_round(
                    sample, round_num
                )

                round_key = f"round_{round_num}"
                round_stats[round_key]["total_samples"] += 1

                if agent1_correct:
                    round_stats[round_key]["agent1_correct"] += 1
                if agent2_correct:
                    round_stats[round_key]["agent2_correct"] += 1
                if agent1_correct and agent2_correct:
                    round_stats[round_key]["model_correct"] += 1

    for round_key in round_stats:
        stats = round_stats[round_key]
        total = stats["total_samples"]
        if total > 0:
            stats["agent1_accuracy"] = stats["agent1_correct"] / total
            stats["agent2_accuracy"] = stats["agent2_correct"] / total
            stats["model_accuracy"] = stats["model_correct"] / total
        else:
            stats["agent1_accuracy"] = 0
            stats["agent2_accuracy"] = 0
            stats["model_accuracy"] = 0

    return round_stats


def calculate_accuracy(samples: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate accuracy rates for model agents.

    Args:
        samples: Dictionary of all samples

    Returns:
        Dictionary with accuracy statistics
    """
    total_samples = len(samples)
    agent1_correct_count = 0
    agent2_correct_count = 0
    model_correct_count = 0

    print("Evaluating sample correctness...")

    for sample_id, sample in samples.items():
        agent1_correct, agent2_correct = evaluate_sample_correctness(sample)

        if agent1_correct:
            agent1_correct_count += 1
        if agent2_correct:
            agent2_correct_count += 1
        if agent1_correct and agent2_correct:
            model_correct_count += 1

    return {
        "total_samples": total_samples,
        "agent1_correct": agent1_correct_count,
        "agent2_correct": agent2_correct_count,
        "model_correct": model_correct_count,
        "agent1_accuracy": (
            agent1_correct_count / total_samples if total_samples > 0 else 0
        ),
        "agent2_accuracy": (
            agent2_correct_count / total_samples if total_samples > 0 else 0
        ),
        "model_accuracy": (
            model_correct_count / total_samples if total_samples > 0 else 0
        ),
    }


def calculate_token_statistics(samples: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculate token statistics using OpenAI's tiktoken.

    Args:
        samples: Dictionary of all samples

    Returns:
        Dictionary with token statistics
    """
    # Initialize tiktoken encoder for GPT-4
    encoding = tiktoken.encoding_for_model("gpt-4")

    total_agent1_tokens = 0
    total_agent2_tokens = 0
    total_agent1_round1_tokens = 0
    total_agent1_round2_tokens = 0
    total_agent1_round3_tokens = 0
    total_agent2_round1_tokens = 0
    total_agent2_round2_tokens = 0
    total_agent2_round3_tokens = 0

    print("Calculating token statistics...")

    for sample_id, sample in samples.items():
        rounds = sample["details"]
        for round_key, round_data in rounds.items():
            if round_key.startswith("round_"):
                agent1_response = round_data["agent1_content"].get("response", "")
                agent2_response = round_data["agent2_content"].get("response", "")

                agent1_tokens = len(encoding.encode(agent1_response))
                agent2_tokens = len(encoding.encode(agent2_response))

                total_agent1_tokens += agent1_tokens
                total_agent2_tokens += agent2_tokens

                if round_key == "round_1":
                    total_agent1_round1_tokens += agent1_tokens
                    total_agent2_round1_tokens += agent2_tokens
                elif round_key == "round_2":
                    total_agent1_round2_tokens += agent1_tokens
                    total_agent2_round2_tokens += agent2_tokens
                elif round_key == "round_3":
                    total_agent1_round3_tokens += agent1_tokens
                    total_agent2_round3_tokens += agent2_tokens

    return {
        "total_agent1_tokens": total_agent1_tokens,
        "total_agent2_tokens": total_agent2_tokens,
        "total_tokens": total_agent1_tokens + total_agent2_tokens,
        "total_agent1_round1_tokens": total_agent1_round1_tokens,
        "total_agent1_round2_tokens": total_agent1_round2_tokens,
        "total_agent1_round3_tokens": total_agent1_round3_tokens,
        "total_agent2_round1_tokens": total_agent2_round1_tokens,
        "total_agent2_round2_tokens": total_agent2_round2_tokens,
        "total_agent2_round3_tokens": total_agent2_round3_tokens,
    }


def calculate_runtime_statistics(samples: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate runtime statistics.

    Args:
        samples: Dictionary of all samples

    Returns:
        Dictionary with runtime statistics
    """
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

    for sample_id, sample in samples.items():
        overview_runtime = sample["overview"].get("total_runtime", 0)
        total_overview_runtime += overview_runtime

        rounds = sample["details"]
        for round_key, round_data in rounds.items():
            if round_key.startswith("round_"):
                total_rounds += 1
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

    return {
        "total_overview_runtime": total_overview_runtime,
        "total_agent1_runtime": total_agent1_runtime,
        "total_agent2_runtime": total_agent2_runtime,
        "total_rounds": total_rounds,
        "total_agent1_round1_runtime": total_agent1_round1_runtime,
        "total_agent1_round2_runtime": total_agent1_round2_runtime,
        "total_agent1_round3_runtime": total_agent1_round3_runtime,
        "total_agent2_round1_runtime": total_agent2_round1_runtime,
        "total_agent2_round2_runtime": total_agent2_round2_runtime,
        "total_agent2_round3_runtime": total_agent2_round3_runtime,
    }


def generate_report(
    accuracy_stats: Dict[str, float],
    runtime_stats: Dict[str, float],
    round_accuracy_stats: Dict[str, Dict[str, float]],
    token_stats: Dict[str, int],
) -> str:
    """
    Generate a comprehensive statistics report.

    Args:
        accuracy_stats: Final round accuracy statistics
        runtime_stats: Runtime statistics
        round_accuracy_stats: Accuracy statistics by rounds
        token_stats: Token usage statistics

    Returns:
        Formatted report string
    """
    report = f"""
{'='*80}
Multi-Agent Debate Results Statistical Analysis
{'='*80}

FINAL ROUND ACCURACY STATISTICS:
{'─'*40}
Total Samples: {accuracy_stats['total_samples']}

Agent Performance:
  • Agent1 Correct: {accuracy_stats['agent1_correct']} ({accuracy_stats['agent1_accuracy']:.4f})
  • Agent2 Correct: {accuracy_stats['agent2_correct']} ({accuracy_stats['agent2_accuracy']:.4f})
  • model Correct: {accuracy_stats['model_correct']} ({accuracy_stats['model_accuracy']:.4f})

Individual Accuracy Rates:
  • Agent1 Accuracy: {accuracy_stats['agent1_accuracy']*100:.2f}%
  • Agent2 Accuracy: {accuracy_stats['agent2_accuracy']*100:.2f}%
  • model Agents Accuracy: {accuracy_stats['model_accuracy']*100:.2f}%

ACCURACY BY ROUNDS:
{'─'*40}"""

    # Add round-by-round accuracy
    for round_key in ["round_1", "round_2", "round_3"]:
        if round_key in round_accuracy_stats:
            stats = round_accuracy_stats[round_key]
            if stats["total_samples"] > 0:
                report += f"""
{round_key.upper()} Statistics (Total: {stats['total_samples']} samples):
  • Agent1 Correct: {stats['agent1_correct']} ({stats['agent1_accuracy']:.4f}) - {stats['agent1_accuracy']*100:.2f}%
  • Agent2 Correct: {stats['agent2_correct']} ({stats['agent2_accuracy']:.4f}) - {stats['agent2_accuracy']*100:.2f}%
  • model Correct: {stats['model_correct']} ({stats['model_accuracy']:.4f}) - {stats['model_accuracy']*100:.2f}%"""

    report += f"""

RUNTIME STATISTICS:
{'─'*40}
Total Runtime Statistics:
  • Total Overview Runtime: {runtime_stats['total_overview_runtime']:.2f} seconds
  • Total Agent1 Runtime: {runtime_stats['total_agent1_runtime']:.2f} seconds
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

TOKEN USAGE STATISTICS:
{'─'*40}
Total Token Usage:
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

{'='*80}
"""
    return report


def save_detailed_results(
    accuracy_stats: Dict[str, float],
    runtime_stats: Dict[str, float],
    round_accuracy_stats: Dict[str, Dict[str, float]],
    token_stats: Dict[str, int],
    output_dir: str = None,
) -> str:
    """
    Save detailed results to JSON file.

    Args:
        accuracy_stats: Final round accuracy statistics
        runtime_stats: Runtime statistics
        round_accuracy_stats: Accuracy statistics by rounds
        token_stats: Token usage statistics
        output_dir: Output directory for saving results

    Returns:
        Path to saved file
    """
    from datetime import datetime

    if output_dir is None:
        output_dir = os.getcwd()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_round_accuracy_statistics": accuracy_stats,
        "round_by_round_accuracy_statistics": round_accuracy_stats,
        "runtime_statistics": runtime_stats,
        "token_usage_statistics": token_stats,
    }

    output_file = os.path.join(output_dir, f"summary_math_multi_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return output_file


def main():
    """
    Main function to run the statistical analysis.
    """
    # =========================== 路径配置 ===========================
    # 服务器路径配置 (Linux server paths)
    # results_directory = (
    #    "/home/kaibiaozhu/reagents/results/2_results_multi_math_temp0/merged"
    # )
    # output_directory = (
    #    "/home/kaibiaozhu/reagents/results/2_results_multi_math_temp0/statistic_summary"
    # )

    # 本地Mac电脑路径配置 (Local Mac paths) - 使用时请注释掉上面的服务器路径
    results_directory = (
        "/Users/kaibiaozhu/Documents/GitHub/reagents/results/4_single_math_dou/merged"
    )
    output_directory = "/Users/kaibiaozhu/Documents/GitHub/reagents/results/4_single_math_dou/statistic_summary"

    print("Multi-Agent Debate Results Statistical Analysis")
    print("=" * 50)

    print(f"\n1. Loading JSON files from: {results_directory}")
    json_data_list = load_json_files(results_directory)

    if not json_data_list:
        print("No JSON files found. Exiting.")
        return

    print("\n2. Extracting samples...")
    all_samples = extract_all_samples(json_data_list)

    print("\n3. Calculating final round accuracy statistics...")
    accuracy_stats = calculate_accuracy(all_samples)

    print("\n4. Calculating round-by-round accuracy statistics...")
    round_accuracy_stats = calculate_accuracy_by_rounds(all_samples)

    print("\n5. Calculating runtime statistics...")
    runtime_stats = calculate_runtime_statistics(all_samples)

    print("\n6. Calculating token usage statistics...")
    token_stats = calculate_token_statistics(all_samples)

    print("\n7. Generating report...")
    report = generate_report(
        accuracy_stats, runtime_stats, round_accuracy_stats, token_stats
    )
    print(report)

    print("\n8. Saving detailed results...")
    output_file = save_detailed_results(
        accuracy_stats,
        runtime_stats,
        round_accuracy_stats,
        token_stats,
        output_directory,
    )
    print(f"Detailed results saved to: {output_file}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
