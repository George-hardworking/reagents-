###################################################################################
# File: dataset_test_unified.py
# File Manager: Kaibiao ZHU & Jiajun Lin
# Description: This file is to test both multi-agent debate and single-agent models on datasets like GSM8K and MATH,
#              evaluating accuracy using agentvs standards with configurable mode selection.
####################################################################################

import argparse
import datetime
import pathlib
import random
import json
import sys
import os
from agentvs import run_multi_agent_debate_for_dataset, run_single_agent_for_dataset
from data.dataset_loader import DatasetLoader
from core.record import save_record
from dotenv import load_dotenv
from typing import Optional
import math

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()


def _safe_float(x: Optional[str]) -> Optional[float]:
    """Convert string to float safely"""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _custom_judge(
    pred_norm: Optional[str], gold_norm: Optional[str], tol: float = 1e-6
) -> bool:
    """Custom judge: string equality + numeric approximation"""
    if pred_norm is None or gold_norm is None:
        return False
    if pred_norm == gold_norm:
        return True
    f1, f2 = _safe_float(pred_norm), _safe_float(gold_norm)
    if f1 is not None and f2 is not None:
        return math.isclose(f1, f2, rel_tol=tol, abs_tol=tol)
    return False


def get_test_samples(
    dataset_name="gsm8k",
    n_samples=-1,
    seed=42,
    start_index=0,
    chunk_size=-1,
    random_mode=False,
):
    """Enhanced sampling with support for sharding and random mode"""
    loader = DatasetLoader(dataset_name)
    questions, answers = loader.load_data()
    total = len(questions)

    if total == 0:
        return []

    if random_mode:
        random.seed(seed)
        if n_samples == -1 or n_samples >= total:
            selected_indices = list(range(total))
        else:
            selected_indices = random.sample(list(range(total)), n_samples)
            selected_indices.sort()
    else:
        # 分片逻辑（支持并行处理）
        start_index = max(0, start_index)
        if start_index >= total:
            return []

        if chunk_size == -1:
            end_index = total
        else:
            end_index = min(total, start_index + chunk_size)

        if n_samples != -1:
            end_index = min(end_index, start_index + n_samples)

        selected_indices = list(range(start_index, end_index))

    return [
        {
            "question": questions[idx],
            "answer": answers[idx],
            "index": idx,  # 保留索引信息
        }
        for idx in selected_indices
    ]


def test_dataset_debate(
    test_samples, model1_name, model2_name, dataset_name, save_results, results_dir
):
    """use debate model to answer dataset questions and evaluate accuracy using agentvs standards"""

    chunk_dir_path = pathlib.Path(results_dir)
    if test_samples and save_results:
        indices = [sample.get("index", idx) for idx, sample in enumerate(test_samples)]
        indices.sort()
        chunk_folder_name = f"chunk_{indices[0]:05d}-{indices[-1]:05d}"
        chunk_dir_path = chunk_dir_path / "chunks" / chunk_folder_name

    for i, sample in enumerate(test_samples):
        sample_global_index = sample.get("index", i)
        print(
            f"\n=== Test Sample {i+1}/{len(test_samples)} (global #{sample_global_index}) ==="
        )
        print(f"Question: {sample['question']}")
        print(f"Correct Answer (raw): {sample['answer']}")

        result = run_multi_agent_debate_for_dataset(
            question=sample["question"],
            gold_answer=sample["answer"],
            model_name_agent1=model1_name,
            model_name_agent2=model2_name,
            save_result=save_results,
            dataset_name=dataset_name,
            sample_id=f"sample_{sample_global_index:05d}",
            results_dir=str(chunk_dir_path),
        )

        print(f"Agent1 (Extracted): {result.agent1_final_extracted}")
        print(f"Agent2 (Extracted): {result.agent2_final_extracted}")
        print(f"Agent1 Full Answer: {result.agent1_raw}")
        print(f"Agent2 Full Answer: {result.agent2_raw}")
        print(f"Gold Answer: {result.gold_answer}")


def test_dataset_single(
    test_samples,
    model_name,
    dataset_name,
    save_results,
    results_dir,
):
    """Run single agent for provided samples and optionally persist results (parallel style)."""

    chunk_dir_path = pathlib.Path(results_dir)
    if test_samples and save_results:
        indices = [sample.get("index", idx) for idx, sample in enumerate(test_samples)]
        indices.sort()
        chunk_folder_name = f"chunk_{indices[0]:05d}-{indices[-1]:05d}"
        chunk_dir_path = chunk_dir_path / "chunks" / chunk_folder_name

    for i, sample in enumerate(test_samples):
        sample_global_index = sample.get("index", i)
        print(
            f"\n=== Test Sample {i+1}/{len(test_samples)} "
            f"(global #{sample_global_index}) ==="
        )
        print(f"Question: {sample['question']}")
        print(f"Correct Answer (raw): {sample['answer']}")

        # 调用单智能体函数
        result = run_single_agent_for_dataset(
            question=sample["question"],
            gold_answer=sample["answer"],
            model_name=model_name,
            prompt_type="cot",
            save_result=save_results,
            dataset_name=dataset_name,
            sample_id=f"sample_{sample_global_index:05d}",
            results_dir=str(chunk_dir_path),
        )

        print(f"Model Full Answer: {result.answer}")
        print(f"Gold Answer: {sample['answer']}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset test runner supporting both single and multi-agent modes"
    )

    # 基础参数
    parser.add_argument(
        "--dataset", default="math", choices=["math", "gsm8k"], help="选择数据集"
    )
    parser.add_argument(
        "--mode",
        default="single_agent",
        choices=["single_agent", "multi_agent"],
        help="选择模式",
    )

    # 采样参数
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="样本数量，-1表示全部"
    )
    parser.add_argument("--start", type=int, default=0, help="起始索引（支持分片）")
    parser.add_argument("--chunk-size", type=int, default=-1, help="块大小（并行模式）")
    parser.add_argument("--random", action="store_true", help="启用随机采样")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 模型参数
    parser.add_argument("--model1", default="qwen", help="Agent1模型（多智能体模式）")
    parser.add_argument(
        "--model2", default="volc_ark", help="Agent2模型（多智能体模式）"
    )
    parser.add_argument("--single-model", default="qwen", help="单智能体模型")

    # 保存参数
    parser.add_argument(
        "--save-results", action="store_true", default=True, help="保存结果"
    )
    parser.add_argument("--results-dir", default="results_unified", help="结果目录")

    args = parser.parse_args()

    # 从参数获取配置
    dataset_name = args.dataset
    mode = args.mode
    model1_name = args.model1
    model2_name = args.model2
    single_model_name = args.single_model
    seed = args.seed
    n_samples = args.n_samples
    save_results = args.save_results
    results_dir = args.results_dir

    print("=" * 100)
    print(f"Dataset={dataset_name.upper()}  Mode={mode.upper()}")
    if mode == "multi_agent":
        print(f"Model1={model1_name}  Model2={model2_name}")
    else:
        print(f"Model={single_model_name}")
    print(f"Samples: {n_samples if n_samples != -1 else 'ALL'}, Start: {args.start}")
    if args.chunk_size != -1:
        print(f"Chunk-size: {args.chunk_size}")
    print("=" * 100)

    test_samples = get_test_samples(
        dataset_name=dataset_name,
        n_samples=n_samples,
        seed=seed,
        start_index=args.start,
        chunk_size=args.chunk_size,
        random_mode=args.random,
    )

    if not test_samples:
        print("⚠️ 未获取到任何测试样本，请检查参数设置")
        return

    if mode == "multi_agent":
        test_dataset_debate(
            test_samples,
            model1_name,
            model2_name,
            dataset_name,
            save_results,
            results_dir,
        )
    elif mode == "single_agent":
        test_dataset_single(
            test_samples,
            single_model_name,
            dataset_name,
            save_results,
            results_dir,
        )
    else:
        print(f"错误: 未知的模式 '{mode}'，请选择 'single_agent' 或 'multi_agent'")


if __name__ == "__main__":
    main()
