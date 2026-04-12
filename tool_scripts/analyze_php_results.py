"""
File: analyze_php_results.py
      Statistical Analysis Script for PHP (Progressive-Hint Prompting) Results
File Manager: Kaibiao ZHU
Description: This script analyzes JSON result files from PHP self-reflection experiments,
             calculating accuracy rates, runtime statistics, and token usage.
####################################################################################
"""

import os
import json
import glob
import tiktoken

from pathlib import Path
from agentvs.grader import grade_answer
from typing import Dict, List, Any, Tuple
from agentvs.eval_math import parse_answer
from agentvs.math_normalize import normalize_answer


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
) -> bool:
    """
    Evaluate if a sample's answer was correct in a specific round.

    Args:
        sample: Sample data dictionary
        round_num: Round number to evaluate (1, 2, 3, etc.)

    Returns:
        Boolean indicating if the answer was correct
    """
    try:
        gold_answer = sample["overview"]["gold_answer"]

        rounds = sample["details"]
        round_key = f"round_{round_num}"

        if round_key not in rounds:
            return False

        round_data = rounds[round_key]

        # PHP结构中，agent1是主要的答案生成者
        agent_answer = round_data["agent1_content"]["response"]

        gold_parsed = parse_answer(gold_answer)
        gold_normalized = (
            normalize_answer(gold_parsed)
            if gold_parsed
            else normalize_answer(gold_answer)
        )

        agent_parsed = parse_answer(agent_answer)
        agent_normalized = (
            normalize_answer(agent_parsed)
            if agent_parsed
            else normalize_answer(agent_answer)
        )

        is_correct = grade_answer(agent_normalized, gold_normalized)

        return is_correct

    except Exception as e:
        print(f"Error evaluating sample correctness for round {round_num}: {e}")
        return False


def evaluate_sample_correctness(sample: Dict[str, Any]) -> bool:
    """
    Evaluate if a sample's final answer was correct.

    Args:
        sample: Sample data dictionary

    Returns:
        Boolean indicating if the final answer was correct
    """
    try:
        gold_answer = sample["overview"]["gold_answer"]

        rounds = sample["details"]
        final_round_key = f"round_{sample['overview']['total_rounds']}"
        final_round = rounds[final_round_key]

        # PHP结构中，agent1是主要的答案生成者
        agent_answer = final_round["agent1_content"]["response"]

        gold_parsed = parse_answer(gold_answer)
        gold_normalized = (
            normalize_answer(gold_parsed)
            if gold_parsed
            else normalize_answer(gold_answer)
        )

        agent_parsed = parse_answer(agent_answer)
        agent_normalized = (
            normalize_answer(agent_parsed)
            if agent_parsed
            else normalize_answer(agent_answer)
        )

        is_correct = grade_answer(agent_normalized, gold_normalized)

        return is_correct

    except Exception as e:
        print(f"Error evaluating sample correctness: {e}")
        return False


def calculate_accuracy_by_rounds(
    samples: Dict[str, Any], max_rounds: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy rates by rounds.

    Args:
        samples: Dictionary of all samples
        max_rounds: Maximum number of rounds to analyze

    Returns:
        Dictionary with accuracy statistics for each round
    """
    print("Evaluating sample correctness by rounds...")

    round_stats = {}
    for round_num in range(1, max_rounds + 1):
        round_stats[f"round_{round_num}"] = {
            "total_samples": 0,
            "agent1_correct": 0,
            "model_correct": 0,
        }

    for sample_id, sample in samples.items():
        total_rounds = sample["overview"]["total_rounds"]

        for round_num in range(1, max_rounds + 1):
            if round_num <= total_rounds:
                is_correct = evaluate_sample_correctness_by_round(sample, round_num)

                round_key = f"round_{round_num}"
                round_stats[round_key]["total_samples"] += 1

                if is_correct:
                    round_stats[round_key]["agent1_correct"] += 1
                    round_stats[round_key]["model_correct"] += 1

    # Calculate accuracy rates
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


def calculate_accuracy(samples: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate final accuracy rates.

    Args:
        samples: Dictionary of all samples

    Returns:
        Dictionary with accuracy statistics
    """
    total_samples = len(samples)
    correct_count = 0

    print("Evaluating sample correctness...")

    for sample_id, sample in samples.items():
        is_correct = evaluate_sample_correctness(sample)

        if is_correct:
            correct_count += 1

    return {
        "total_samples": total_samples,
        "agent1_correct": correct_count,
        "model_correct": correct_count,
        "agent1_accuracy": correct_count / total_samples if total_samples > 0 else 0,
        "model_accuracy": correct_count / total_samples if total_samples > 0 else 0,
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
    round_tokens = {}  # {round_1: tokens, round_2: tokens, ...}

    print("Calculating token statistics...")

    for sample_id, sample in samples.items():
        rounds = sample["details"]
        for round_key, round_data in rounds.items():
            if round_key.startswith("round_"):
                agent_response = round_data["agent1_content"].get("response", "")
                agent_tokens = len(encoding.encode(agent_response))

                total_agent1_tokens += agent_tokens

                if round_key not in round_tokens:
                    round_tokens[round_key] = 0
                round_tokens[round_key] += agent_tokens

    # Build the result dictionary
    result = {
        "total_tokens": total_agent1_tokens,
        "total_agent1_tokens": total_agent1_tokens,
    }

    # Add round-specific token counts
    for i in range(1, 4):  # rounds 1-3
        round_key = f"round_{i}"
        result[f"total_agent1_{round_key}_tokens"] = round_tokens.get(round_key, 0)

    return result


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
    total_rounds = 0
    round_runtimes = {}  # {round_1: runtime, round_2: runtime, ...}

    print("Calculating runtime statistics...")

    for sample_id, sample in samples.items():
        overview_runtime = sample["overview"].get("total_runtime", 0)
        total_overview_runtime += overview_runtime

        rounds = sample["details"]
        for round_key, round_data in rounds.items():
            if round_key.startswith("round_"):
                total_rounds += 1
                agent_runtime = round_data["agent1_content"].get("runtime", 0)
                total_agent1_runtime += agent_runtime

                if round_key not in round_runtimes:
                    round_runtimes[round_key] = 0
                round_runtimes[round_key] += agent_runtime

    # Build the result dictionary
    result = {
        "total_overview_runtime": total_overview_runtime,
        "total_agent1_runtime": total_agent1_runtime,
        "total_rounds": total_rounds,
    }

    # Add round-specific runtime counts
    for i in range(1, 4):  # rounds 1-3
        round_key = f"round_{i}"
        result[f"total_agent1_{round_key}_runtime"] = round_runtimes.get(round_key, 0)

    return result


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
PHP (Progressive-Hint Prompting) Results Statistical Analysis
{'='*80}

FINAL ROUND ACCURACY STATISTICS:
{'─'*40}
Total Samples: {accuracy_stats['total_samples']}

Performance:
  • Correct Answers: {accuracy_stats['agent1_correct']}
  • Accuracy Rate: {accuracy_stats['agent1_accuracy']*100:.2f}%

ACCURACY BY ROUNDS:
{'─'*40}"""

    # Add round-by-round accuracy
    for i in range(1, 4):
        round_key = f"round_{i}"
        if round_key in round_accuracy_stats:
            stats = round_accuracy_stats[round_key]
            if stats["total_samples"] > 0:
                report += f"""
{round_key.upper()} Statistics (Total: {stats['total_samples']} samples):
  • Correct: {stats['agent1_correct']} ({stats['agent1_accuracy']:.4f}) - {stats['agent1_accuracy']*100:.2f}%"""

    report += f"""

RUNTIME STATISTICS:
{'─'*40}
Total Runtime Statistics:
  • Total Overview Runtime: {runtime_stats['total_overview_runtime']:.2f} seconds
  • Total Agent Runtime: {runtime_stats['total_agent1_runtime']:.2f} seconds
  • Total Rounds: {runtime_stats['total_rounds']}

Runtime by Rounds:"""

    for i in range(1, 4):
        round_key = f"round_{i}"
        runtime_key = f"total_agent1_{round_key}_runtime"
        if runtime_key in runtime_stats:
            report += f"""
  • Total {round_key.upper()} Runtime: {runtime_stats[runtime_key]:.2f} seconds"""

    report += f"""

Time Conversion:
  • Total Overview Runtime: {runtime_stats['total_overview_runtime']/3600:.2f} hours
  • Total Agent Runtime: {runtime_stats['total_agent1_runtime']/3600:.2f} hours

TOKEN USAGE STATISTICS:
{'─'*40}
Total Token Usage:
  • Total Tokens: {token_stats['total_tokens']:,}

Token Usage by Rounds:"""

    for i in range(1, 4):
        round_key = f"round_{i}"
        token_key = f"total_agent1_{round_key}_tokens"
        if token_key in token_stats:
            report += f"""
  • {round_key.upper()} Tokens: {token_stats[token_key]:,}"""

    report += f"""

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

    output_file = os.path.join(output_dir, f"summary_math_php_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return output_file


def main():
    """
    Main function to run the statistical analysis.
    """
    # =========================== 路径配置 ===========================
    # 服务器路径配置 (Linux server paths)
    results_directory = (
        "/home/kaibiaozhu/reagents/results/12_singleself_gsm8k_qwen14b/merged"
    )
    output_directory = "/home/kaibiaozhu/reagents/results/12_singleself_gsm8k_qwen14b/statistic_summary"

    # 本地Mac电脑路径配置 (Local Mac paths) - 使用时请注释掉上面的服务器路径
    # results_directory = "/Users/kaibiaozhu/Documents/GitHub/reagents/results/6_singleself_math_qwen/merged"
    # output_directory = "/Users/kaibiaozhu/Documents/GitHub/reagents/results/6_singleself_math_qwen/statistic_summary"

    print("PHP (Progressive-Hint Prompting) Results Statistical Analysis")
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
    round_accuracy_stats = calculate_accuracy_by_rounds(all_samples, max_rounds=3)

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
