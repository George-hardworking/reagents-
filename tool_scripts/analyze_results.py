"""
File: analyze_results.py
      Unified Analysis Script for Agent Experiment Results
File Manager: Kaibiao ZHU
Description: This script provides a unified interface to analyze both single-agent
             and multi-agent debate experiment results. It automatically detects the
             experiment type and generates comprehensive statistical reports.
Usage:
    python tool_scripts/analyze_results.py --results_dir <path_to_results>
    python tool_scripts/analyze_results.py --results_dir <path_to_results> --output_dir <path_to_output>
####################################################################################
"""

# 在terminal中运行以下命令
# python tool_scripts/analyze_results.py \
#     --results_dir results/5_results_single_math_temp0/merged \
#     --output_dir results/5_results_single_math_temp0/statistic_summary


import argparse
import os
import sys

# Add parent directory to path to import agentvs modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvs.statistics import ExperimentStatistics


def main():
    """主函数：解析命令行参数并运行分析"""
    parser = argparse.ArgumentParser(
        description="Unified statistical analysis for agent experiment results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing result JSON files (merged folder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory for saving analysis results (optional)",
    )

    args = parser.parse_args()

    # 验证结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory does not exist: {args.results_dir}")
        sys.exit(1)

    # 创建统计分析器并运行分析
    analyzer = ExperimentStatistics(
        results_directory=args.results_dir, output_directory=args.output_dir
    )

    try:
        analyzer.run_analysis()
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
