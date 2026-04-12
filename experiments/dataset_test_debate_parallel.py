###################################################################################
# File: dataset_test_debate_parallel.py
# File Manager: Kaibiao ZHU
# Description: Enhanced runner supporting dataset sharding & CLI options for
#              parallel evaluation of multi-agent debate models on GSM8K/MATH.
####################################################################################

import argparse
import pathlib
import random
import sys
from typing import List, Dict

from dotenv import load_dotenv

from agentvs import run_multi_agent_debate_for_dataset
from agentvs.dataset_loader import DatasetLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()


def get_test_samples(
    dataset_name: str = "gsm8k",
    n_samples: int = -1,
    seed: int = 42,
    start_index: int = 0,
    chunk_size: int = -1,
    random_mode: bool = False,
) -> List[Dict[str, str]]:
    """Load dataset and return selected samples with optional sharding.

    * random_mode=True  -> random sampling (backwards compatible).
    * random_mode=False -> deterministic slicing, ideal for multi-process runs.
    """

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
            "index": idx,
        }
        for idx in selected_indices
    ]


def run_debate_batch(
    samples: List[Dict[str, str]],
    model1_name: str,
    model2_name: str,
    dataset_name: str,
    save_results: bool,
    results_dir: str,
) -> None:

    # 添加模型名称映射
    model_name_map = {
        "qwen": "qwen2.5-7b-instruct",
        "volc": "Doubao-Seed-1.6-flash",
    }

    # 应用映射
    display_model1 = model_name_map.get(model1_name, model1_name)
    display_model2 = model_name_map.get(model2_name, model2_name)

    # 然后在打印时使用 display_model1 和 display_model2
    print(f"Agent1 Model: {display_model1}")
    print(f"Agent2 Model: {display_model2}")
    """Run debate for provided samples and optionally persist results."""

    chunk_dir_path = pathlib.Path(results_dir)
    if samples and save_results:
        indices = [sample.get("index", idx) for idx, sample in enumerate(samples)]
        indices.sort()
        chunk_folder_name = f"chunk_{indices[0]:05d}-{indices[-1]:05d}"
        chunk_dir_path = chunk_dir_path / "chunks" / chunk_folder_name

    for i, sample in enumerate(samples):
        sample_global_index = sample.get("index", i)
        print(
            f"\n=== Test Sample {i+1}/{len(samples)} "
            f"(global #{sample_global_index}) ==="
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


def main() -> None:
    # === 配置说明 ===
    # 根据需要直接在下面 parser.add_argument 的 default 参数中修改默认值。
    # 常用调整：
    #   - 指定数据集: 把 default="math" 改成 "gsm8k" 即可。
    #   - 指定模型: 修改 --model1 / --model2 的 default 值。
    #   - 顺序分片: 修改 --offset 和 --chunk-size 的 default，示例：
    #       default=0  表示从第 0 条样本开始
    #       default=200 表示每个脚本处理 200 条题目
    #   - 随机采样: 如果想延续旧逻辑，设置 random_mode 默认 True，或者运行时加 --random
    #   - 写入结果: 将 --save-results 的 default 改成 True，运行时就不用再传参
    #   - 输出目录: 更新 --results-dir 的 default，比如改成 "results/math_parallel"
    #
    parser = argparse.ArgumentParser(
        description="Parallel-friendly runner for multi-agent debate datasets."
    )
    parser.add_argument(
        "--dataset", default="math", choices=["math", "gsm8k"], help="选择评测数据集"
    )
    parser.add_argument("--model1", default="qwen", help="Agent1 模型标识")
    parser.add_argument("--model2", default="qwen", help="Agent2 模型标识")
    parser.add_argument("--seed", type=int, default=42, help="随机模式下的种子")
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="采样数量，-1 表示全部/块大小"
    )
    parser.add_argument("--offset", type=int, default=0, help="顺序模式下的起始索引")
    parser.add_argument(
        "--chunk-size", type=int, default=200, help="顺序模式下一次处理的最大样本量"
    )
    parser.add_argument("--random", action="store_true", help="启用随机采样模式")
    parser.add_argument(
        "--save-results",
        default=True,
        action="store_true",
        help="写出 results/{sample}.json",
    )
    parser.add_argument(
        "--results-dir", default="results_temperature0", help="结果保存目录"
    )

    args = parser.parse_args()

    print("=" * 100)
    print(
        f"Dataset={args.dataset.upper()}  Model1={args.model1}"
        f"  Model2={args.model2}  Mode={'RANDOM' if args.random else 'SEQUENTIAL'}"
    )
    if args.random:
        print(
            f"Random Samples: {args.n_samples if args.n_samples != -1 else 'ALL'}"
            f"  Seed: {args.seed}"
        )
    else:
        chunk_desc = (
            args.chunk_size
            if args.chunk_size != -1
            else (args.n_samples if args.n_samples != -1 else "ALL-REMAINING")
        )
        print(f"Range Offset: {args.offset}  Chunk Size: {chunk_desc}")
    print("=" * 100)

    samples = get_test_samples(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        seed=args.seed,
        start_index=args.offset,
        chunk_size=args.chunk_size,
        random_mode=args.random,
    )

    if not samples:
        print(
            "⚠️ 未获取到任何测试样本，请检查 offset/chunk-size/n-samples 是否超出范围。"
        )
        return

    print(f"Loaded {len(samples)} samples (dataset total may be larger).")

    run_debate_batch(
        samples,
        args.model1,
        args.model2,
        args.dataset,
        args.save_results,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
