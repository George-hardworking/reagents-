###################################################################################
# File: compare_all_experiments_en.py
# File Manager: Kaibiao ZHU
# Description: Comprehensive analysis and comparison of all MATH dataset experiments
#              Generates visualizations and comparative tables for experiments 2-6
#              All outputs in English
####################################################################################

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set matplotlib backend and style
plt.switch_backend("Agg")
sns.set_style("whitegrid")
sns.set_palette("husl")


class ExperimentConfig:
    """Experiment configuration information"""

    def __init__(
        self,
        exp_id: str,
        name: str,
        method: str,
        agent1_model: str,
        agent2_model: Optional[str] = None,
        temperature: float = 0.0,
        description: str = "",
    ):
        self.exp_id = exp_id
        self.name = name
        self.method = method  # "single", "multi", "php"
        self.agent1_model = agent1_model
        self.agent2_model = agent2_model
        self.temperature = temperature
        self.description = description


# Experiment configuration mapping
EXPERIMENT_CONFIGS = {
    "1_multi_math_qwen_qwen": ExperimentConfig(
        exp_id="1",
        name="Multi-Agent Debate(qwen2.5-7B & qwen2.5-7B)",
        method="multi",
        agent1_model="Qwen2.5-7B",
        agent2_model="Qwen2.5-7B",
        temperature=0.0,
        description="Debate(Qwen7b_Qwen7b)",
    ),
    "2_multi_math_qwen_dou": ExperimentConfig(
        exp_id="2",
        name="Multi-Agent Debate(qwen2.5-7B & Doubao-1.6)",
        method="multi",
        agent1_model="Qwen2.5-7B",
        agent2_model="Doubao-1.6",
        temperature=0.0,
        description="Debate(Qwen7b_Dou)",
    ),
    "3_multi_math_dou_dou": ExperimentConfig(
        exp_id="3",
        name="Multi-Agent Debate(Doubao-1.6 & Doubao-1.6)",
        method="multi",
        agent1_model="Doubao-1.6",
        agent2_model="Doubao-1.6",
        temperature=0.0,
        description="Debate(Dou_Dou)",
    ),
    "4_single_math_qwen": ExperimentConfig(
        exp_id="4",
        name="Single-Agent(qwen2.5-7B)",
        method="single",
        agent1_model="Qwen2.5-7B",
        temperature=0.0,
        description="Single(Qwen7b)",
    ),
    "5_single_math_qwen14b": ExperimentConfig(
        exp_id="5",
        name="Single-Agent(Qwen2.5-14b)",
        method="single",
        agent1_model="Qwen2.5-14b",
        temperature=0.0,
        description="Single(Qwen14b)",
    ),
    "6_single_math_dou": ExperimentConfig(
        exp_id="6",
        name="Single-Agent(Doubao-1.6)",
        method="single",
        agent1_model="Doubao-1.6",
        temperature=0.0,
        description="Single(Doubao)",
    ),
    "7_singleself_math_qwen": ExperimentConfig(
        exp_id="7",
        name="Single Self-Reflection(qwen2.5-7B)",
        method="php",
        agent1_model="Qwen2.5-7B",
        temperature=0.0,
        description="SingleSelf(Qwen7b)",
    ),
    "8_singleself_math_qwen14b": ExperimentConfig(
        exp_id="8",
        name="Single Self-Reflection(Qwen-14b)",
        method="php",
        agent1_model="Qwen-14b",
        temperature=0.0,
        description="SingleSelf(Qwen14b)",
    ),
    "9_singleself_math_dou": ExperimentConfig(
        exp_id="9",
        name="Single Self-Reflection(Doubao-1.6)",
        method="php",
        agent1_model="Doubao-1.6",
        temperature=0.0,
        description="SingleSelf(Dou)",
    ),
}


def find_latest_summary(summary_dir: Path) -> Optional[Path]:
    """Find the latest summary file"""
    if not summary_dir.exists():
        return None

    json_files = sorted(summary_dir.glob("summary_*.json"))
    return json_files[-1] if json_files else None


def load_experiment_data(results_dir: Path) -> Dict[str, Any]:
    """Load data from all experiments"""
    experiments_data = {}

    for exp_folder in sorted(results_dir.iterdir()):
        if not exp_folder.is_dir():
            continue

        folder_name = exp_folder.name
        if folder_name not in EXPERIMENT_CONFIGS:
            continue

        summary_dir = exp_folder / "statistic_summary"
        summary_file = find_latest_summary(summary_dir)

        if not summary_file:
            print(f"⚠️  Summary file not found: {folder_name}")
            continue

        with open(summary_file) as f:
            data = json.load(f)

        config = EXPERIMENT_CONFIGS[folder_name]
        experiments_data[config.exp_id] = {
            "config": config,
            "data": data,
            "summary_file": summary_file.name,
        }

        print(
            f"✓ Loaded Experiment {config.exp_id}: {config.name} ({summary_file.name})"
        )

    return experiments_data


def create_comparison_table(experiments_data: Dict[str, Any]) -> pd.DataFrame:
    """Create comparison table"""
    rows = []

    for exp_id in sorted(experiments_data.keys(), key=int):
        exp = experiments_data[exp_id]
        config = exp["config"]
        data = exp["data"]

        final_stats = data.get("final_round_accuracy_statistics", {})
        runtime_stats = data.get("runtime_statistics", {})
        token_stats = data.get("token_usage_statistics", {})

        # Calculate average runtime and tokens
        total_samples = final_stats.get("total_samples", 0)
        total_runtime = runtime_stats.get("total_overview_runtime", 0)
        total_tokens = token_stats.get("total_tokens", 0)

        avg_runtime_per_sample = (
            total_runtime / total_samples if total_samples > 0 else 0
        )
        avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0

        # Get accuracy
        if config.method == "multi":
            agent1_acc = final_stats.get("agent1_accuracy", 0) * 100
            agent2_acc = final_stats.get("agent2_accuracy", 0) * 100
            model_acc = final_stats.get("model_accuracy", 0) * 100
            final_acc = model_acc  # Use model's result as final
        else:
            agent1_acc = final_stats.get("agent1_accuracy", 0) * 100
            agent2_acc = None
            model_acc = final_stats.get("model_accuracy", 0) * 100
            final_acc = agent1_acc

        row = {
            "Exp ID": f"Exp-{exp_id}",
            "Experiment Name": config.name,
            "Method": config.method.upper(),
            "Agent1 Model": config.agent1_model,
            "Agent2 Model": config.agent2_model if config.agent2_model else "-",
            "Temperature": config.temperature,
            "Samples": total_samples,
            "Agent1 Acc(%)": f"{agent1_acc:.2f}",
            "Agent2 Acc(%)": f"{agent2_acc:.2f}" if agent2_acc else "-",
            "Final Acc(%)": f"{final_acc:.2f}",
            "Total Runtime(h)": f"{total_runtime / 3600:.2f}",
            "Avg Time/Sample(s)": f"{avg_runtime_per_sample:.2f}",
            "Total Tokens": f"{total_tokens:,}",
            "Avg Tokens/Sample": f"{avg_tokens_per_sample:.0f}",
            "Description": config.description,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_accuracy_comparison(experiments_data: Dict[str, Any], output_dir: Path):
    """Plot accuracy comparison charts"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare data
    exp_ids = []
    exp_names = []
    exp_labels = []  # For legend with full description
    final_accuracies = []
    round1_accuracies = []
    round2_accuracies = []
    round3_accuracies = []

    for exp_id in sorted(experiments_data.keys(), key=int):
        exp = experiments_data[exp_id]
        config = exp["config"]
        data = exp["data"]

        exp_ids.append(f"Exp-{exp_id}")
        exp_names.append(config.name)
        exp_labels.append(f"Exp-{exp_id}: {config.name}")

        final_stats = data.get("final_round_accuracy_statistics", {})
        round_stats = data.get("round_by_round_accuracy_statistics", {})

        # Use final accuracy
        if config.method == "multi":
            final_acc = final_stats.get("model_accuracy", 0) * 100
        else:
            final_acc = final_stats.get("agent1_accuracy", 0) * 100

        final_accuracies.append(final_acc)

        # Round-by-round accuracy
        r1_acc = (
            round_stats.get("round_1", {}).get("model_accuracy", 0) * 100
            if config.method == "multi"
            else round_stats.get("round_1", {}).get("agent1_accuracy", 0) * 100
        )

        # For single agent, only round 1 has data; set round 2 and 3 to None
        if config.method == "single":
            r2_acc = None  # No round 2 for single agent
            r3_acc = None  # No round 3 for single agent
        else:
            r2_acc = (
                round_stats.get("round_2", {}).get("model_accuracy", 0) * 100
                if config.method == "multi"
                else round_stats.get("round_2", {}).get("agent1_accuracy", 0) * 100
            )
            r3_acc = (
                round_stats.get("round_3", {}).get("model_accuracy", 0) * 100
                if config.method == "multi"
                else round_stats.get("round_3", {}).get("agent1_accuracy", 0) * 100
            )

        round1_accuracies.append(r1_acc)
        round2_accuracies.append(r2_acc)
        round3_accuracies.append(r3_acc)

    # Chart 1: Final Accuracy Comparison
    colors = sns.color_palette("husl", len(exp_ids))
    bars = axes[0].bar(
        exp_ids, final_accuracies, color=colors, alpha=0.8, edgecolor="black"
    )
    axes[0].set_xlabel("Experiment", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Final Accuracy Comparison Across Experiments", fontsize=14, fontweight="bold"
    )
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis="y", alpha=0.3)

    # Add legend below the plot area
    axes[0].legend(
        bars,
        exp_labels,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        framealpha=0.9,
    )

    # Add values on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Chart 2: Round-by-Round Accuracy Changes
    x = np.arange(len(exp_ids))
    width = 0.25

    # Define round colors (different shades for different rounds)
    round1_color = "#E74C3C"  # Red
    round2_color = "#3498DB"  # Blue
    round3_color = "#2ECC71"  # Green

    # Plot Round 1 bars for all experiments using experiment-specific colors
    bars_r1 = []
    for i in range(len(exp_ids)):
        bar = axes[1].bar(
            x[i] - width,
            round1_accuracies[i],
            width,
            color=round1_color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        bars_r1.append(bar)

    # Plot Round 2 bars only for experiments that have round 2 data
    for i, r2 in enumerate(round2_accuracies):
        if r2 is not None:
            axes[1].bar(
                x[i],
                r2,
                width,
                color=round2_color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

    # Plot Round 3 bars only for experiments that have round 3 data
    for i, r3 in enumerate(round3_accuracies):
        if r3 is not None:
            axes[1].bar(
                x[i] + width,
                r3,
                width,
                color=round3_color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

    # Add legend items manually
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=round1_color, alpha=0.8, edgecolor="black", label="Round 1"),
        Patch(facecolor=round2_color, alpha=0.8, edgecolor="black", label="Round 2"),
        Patch(facecolor=round3_color, alpha=0.8, edgecolor="black", label="Round 3"),
    ]

    axes[1].set_xlabel("Experiment", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_title("Round-by-Round Accuracy Changes", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_ids)
    axes[1].set_ylim([0, 100])
    # Place legend below the plot area
    axes[1].legend(
        handles=legend_elements,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        framealpha=0.9,
    )
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "accuracy_comparison.png"
    # Use bbox_inches='tight' to include the legend outside the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved chart: {output_file.name}")
    plt.close()


def plot_performance_comparison(experiments_data: Dict[str, Any], output_dir: Path):
    """Plot performance comparison (runtime and token usage)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    exp_ids = []
    exp_labels = []  # For legend with full description
    avg_runtimes = []
    avg_tokens = []

    for exp_id in sorted(experiments_data.keys(), key=int):
        exp = experiments_data[exp_id]
        config = exp["config"]
        data = exp["data"]

        exp_ids.append(f"Exp-{exp_id}")
        exp_labels.append(f"Exp-{exp_id}: {config.name}")

        final_stats = data.get("final_round_accuracy_statistics", {})
        runtime_stats = data.get("runtime_statistics", {})
        token_stats = data.get("token_usage_statistics", {})

        total_samples = final_stats.get("total_samples", 1)
        total_runtime = runtime_stats.get("total_overview_runtime", 0)
        total_tokens = token_stats.get("total_tokens", 0)

        avg_runtimes.append(total_runtime / total_samples)
        avg_tokens.append(total_tokens / total_samples)

    # Chart 1: Average Runtime
    colors = sns.color_palette("husl", len(exp_ids))
    bars = axes[0].bar(
        exp_ids, avg_runtimes, color=colors, alpha=0.8, edgecolor="black"
    )
    axes[0].set_xlabel("Experiment", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Avg Runtime (seconds/sample)", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Average Runtime Comparison Across Experiments", fontsize=14, fontweight="bold"
    )
    axes[0].grid(axis="y", alpha=0.3)

    # Add legend below the plot area
    axes[0].legend(
        bars,
        exp_labels,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        framealpha=0.9,
    )

    for bar, runtime in zip(bars, avg_runtimes):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{runtime:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # Chart 2: Average Token Usage
    bars = axes[1].bar(exp_ids, avg_tokens, color=colors, alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("Experiment", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Avg Tokens (tokens/sample)", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Average Token Usage Comparison Across Experiments",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].grid(axis="y", alpha=0.3)

    # Add legend below the plot area
    axes[1].legend(
        bars,
        exp_labels,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        framealpha=0.9,
    )

    for bar, tokens in zip(bars, avg_tokens):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 20,
            f"{tokens:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    output_file = output_dir / "performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved chart: {output_file.name}")
    plt.close()


def plot_round_improvement(experiments_data: Dict[str, Any], output_dir: Path):
    """Plot round-by-round improvement trend"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for exp_id in sorted(experiments_data.keys(), key=int):
        exp = experiments_data[exp_id]
        config = exp["config"]
        data = exp["data"]

        round_stats = data.get("round_by_round_accuracy_statistics", {})

        rounds = []
        accuracies = []

        # For single agent, only plot Round 1 (the only actual round)
        if config.method == "single":
            round_data = round_stats.get("round_1", {})
            acc = round_data.get("agent1_accuracy", 0) * 100
            rounds.append("Round 1")
            accuracies.append(acc)

            # Plot single point for single-agent experiments
            ax.plot(
                rounds,
                accuracies,
                marker="o",
                linewidth=0,  # No line, just a point
                markersize=10,
                label=f"Exp-{exp_id}: {config.name}",
                alpha=0.8,
            )
        else:
            # For multi-agent and PHP, plot all three rounds
            for round_num in ["round_1", "round_2", "round_3"]:
                round_data = round_stats.get(round_num, {})
                if config.method == "multi":
                    acc = round_data.get("model_accuracy", 0) * 100
                else:
                    acc = round_data.get("agent1_accuracy", 0) * 100

                rounds.append(round_num.replace("round_", "Round "))
                accuracies.append(acc)

            # Plot line with markers for multi-round experiments
            ax.plot(
                rounds,
                accuracies,
                marker="o",
                linewidth=2,
                markersize=8,
                label=f"Exp-{exp_id}: {config.name}",
                alpha=0.8,
            )

    ax.set_xlabel("Interaction Round", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Round-by-Round Accuracy Improvement Trend", fontsize=14, fontweight="bold"
    )
    ax.set_ylim([60, 90])
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "round_improvement_trend.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved chart: {output_file.name}")
    plt.close()


def generate_detailed_report(experiments_data: Dict[str, Any], output_dir: Path):
    """Generate detailed text report"""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("DETAILED EXPERIMENT COMPARISON REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 100)
    report_lines.append("")

    for exp_id in sorted(experiments_data.keys(), key=int):
        exp = experiments_data[exp_id]
        config = exp["config"]
        data = exp["data"]

        final_stats = data.get("final_round_accuracy_statistics", {})
        round_stats = data.get("round_by_round_accuracy_statistics", {})
        runtime_stats = data.get("runtime_statistics", {})
        token_stats = data.get("token_usage_statistics", {})

        report_lines.append(f"【Experiment {exp_id}】{config.name}")
        report_lines.append("-" * 100)
        report_lines.append(f"  Method Type: {config.method.upper()}")
        report_lines.append(f"  Agent1 Model: {config.agent1_model}")
        if config.agent2_model:
            report_lines.append(f"  Agent2 Model: {config.agent2_model}")
        report_lines.append(f"  Temperature: {config.temperature}")
        report_lines.append(f"  Description: {config.description}")
        report_lines.append(f"  Data File: {exp['summary_file']}")
        report_lines.append("")

        # Accuracy Statistics
        report_lines.append("  【Accuracy Statistics】")
        total_samples = final_stats.get("total_samples", 0)

        if config.method == "multi":
            agent1_correct = final_stats.get("agent1_correct", 0)
            agent2_correct = final_stats.get("agent2_correct", 0)
            model_correct = final_stats.get("model_correct", 0)

            report_lines.append(f"    Total Samples: {total_samples}")
            report_lines.append(
                f"    Agent1 Correct: {agent1_correct} ({final_stats.get('agent1_accuracy', 0) * 100:.2f}%)"
            )
            report_lines.append(
                f"    Agent2 Correct: {agent2_correct} ({final_stats.get('agent2_accuracy', 0) * 100:.2f}%)"
            )
            report_lines.append(
                f"    Final Correct: {model_correct} ({final_stats.get('model_accuracy', 0) * 100:.2f}%)"
            )
        else:
            agent1_correct = final_stats.get("agent1_correct", 0)
            report_lines.append(f"    Total Samples: {total_samples}")
            report_lines.append(
                f"    Correct: {agent1_correct} ({final_stats.get('agent1_accuracy', 0) * 100:.2f}%)"
            )

        report_lines.append("")

        # Round-by-Round Statistics
        report_lines.append("  【Round-by-Round Accuracy】")

        # For single agent, only show Round 1 since it's the only actual round
        if config.method == "single":
            rounds_to_show = ["round_1"]
        else:
            rounds_to_show = ["round_1", "round_2", "round_3"]

        for round_num in rounds_to_show:
            round_data = round_stats.get(round_num, {})
            round_label = round_num.replace("_", " ").title()

            # Add "(Final)" suffix for single agent
            if config.method == "single":
                round_label += " (Final)"

            if config.method == "multi":
                acc = round_data.get("model_accuracy", 0) * 100
                correct = round_data.get("model_correct", 0)
            else:
                acc = round_data.get("agent1_accuracy", 0) * 100
                correct = round_data.get("agent1_correct", 0)

            report_lines.append(
                f"    {round_label}: {correct}/{total_samples} ({acc:.2f}%)"
            )

        report_lines.append("")

        # Performance Statistics
        total_runtime = runtime_stats.get("total_overview_runtime", 0)
        total_tokens = token_stats.get("total_tokens", 0)
        avg_runtime = total_runtime / total_samples if total_samples > 0 else 0
        avg_tokens = total_tokens / total_samples if total_samples > 0 else 0

        report_lines.append("  【Performance Statistics】")
        report_lines.append(f"    Total Runtime: {total_runtime / 3600:.2f} hours")
        report_lines.append(f"    Avg Time/Sample: {avg_runtime:.2f} seconds")
        report_lines.append(f"    Total Tokens: {total_tokens:,}")
        report_lines.append(f"    Avg Tokens/Sample: {avg_tokens:.0f}")
        report_lines.append("")
        report_lines.append("")

    # Save report
    report_file = output_dir / "detailed_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  ✓ Saved detailed report: {report_file.name}")


def main():
    """Main function"""
    print("=" * 80)
    print("EXPERIMENT COMPARISON ANALYSIS TOOL")
    print("=" * 80)
    print()

    # Setup paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    output_dir = project_root / "results" / "comparison_analysis"
    output_dir.mkdir(exist_ok=True)

    print(f"Results Directory: {results_dir}")
    print(f"Output Directory: {output_dir}")
    print()

    # Load data
    print("Loading experiment data...")
    experiments_data = load_experiment_data(results_dir)

    if not experiments_data:
        print("❌ No experiment data found")
        return

    print(f"\nSuccessfully loaded {len(experiments_data)} experiments")
    print()

    # Create comparison table
    print("Generating comparison table...")
    df = create_comparison_table(experiments_data)

    # Save table
    csv_file = output_dir / "experiments_comparison.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved CSV table: {csv_file.name}")

    # Try to save Excel (if openpyxl is available)
    try:
        excel_file = output_dir / "experiments_comparison.xlsx"
        df.to_excel(excel_file, index=False, engine="openpyxl")
        print(f"  ✓ Saved Excel table: {excel_file.name}")
    except ImportError:
        print(f"  ⚠️  Skipped Excel export (requires openpyxl)")

    # Print table in terminal
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON OVERVIEW")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Generate visualizations
    print("Generating visualization charts...")
    plot_accuracy_comparison(experiments_data, output_dir)
    plot_performance_comparison(experiments_data, output_dir)
    plot_round_improvement(experiments_data, output_dir)
    print()

    # Generate detailed report
    print("Generating detailed text report...")
    generate_detailed_report(experiments_data, output_dir)
    print()

    # Summary
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"All output files saved to: {output_dir}")
    print("=" * 80)
    print("\nOutput Files:")
    for file in sorted(output_dir.iterdir()):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
