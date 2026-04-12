###################################################################################
# Dataset Testing Script: Single-Agent Model Using the Self-Reflection Function of the PHP System
# File Manager: Kaibiao ZHU
# Description: Testing the math dataset using the PHP method
###################################################################################


import sys
import os
import random
import datetime
from typing import Dict, Any

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import DatasetLoader
from agentvs.single_refletion import solve_with_php, PHPResult
from core.record import save_record
from core.prompts import get_prompt, get_php_continuation_prompt
from typing import Literal, Optional


# ============================== User parameter configuration area ==============================
# 在这里填写想要覆盖的参数。设置为 None 表示继续使用命令行参数或默认值。
# 可配置项说明：
#   num_samples     -> int    : 控制采样数量，例如 20、999，设置 -1 代表使用 chunk 或数据集剩余数量
#   random_seed     -> int    : 固定随机种子，确保可复现实验
#   dataset_name    -> str    : 数据集名称，例如 "math"、"gsm8k"
#   model_name      -> str    : 模型名称，例如 "qwen"、"volc_ark"
#   base_prompt_type-> str    : 基础提示模板，可选 "standard"/"cot"/complex_cot"
#   max_interactions-> int    : 每个样本允许的最大交互轮数
#   temperature     -> float  : 模型温度，控制生成多样性
#   save_results    -> bool   : 是否把结果写入磁盘
#   results_dir     -> str    : 保存结果的目录，例如 "results/6_results_self_math_temp0"
#   start_index     -> int    : 顺序模式下的起始索引，配合 chunk_size 使用
#   chunk_size      -> int    : 顺序模式下每批处理的样本数，-1 表示不限
#   random_mode     -> bool   : True 表示随机采样，False 表示顺序采样
#   block_size      -> int    : 保存结果时单个 block 文件包含的样本数
USER_RUN_CONFIG: Dict[str, Optional[Any]] = {
    "num_samples": None,
    "random_seed": None,
    "dataset_name": None,
    "model_name": None,
    "base_prompt_type": None,
    "max_interactions": None,
    "temperature": None,
    "save_results": True,
    "results_dir": None,
    "start_index": None,
    "chunk_size": None,
    "random_mode": False,
    "block_size": None,
}
# ============================ User parameter configuration area ============================


def run_php_for_dataset(
    question: str,
    gold_answer: str,
    model_name: str = "qwen",
    base_prompt_type: Literal["standard", "cot", "complex_cot"] = "complex_cot",
    max_interactions: int = 3,
    temperature: float = 0.0,
    save_result: bool = False,
    dataset_name: str = "math",
    sample_id: Optional[str] = None,
    results_dir: str = "results/6_singleself_math_qwen",
    block_size: int = 1000,
) -> PHPResult:
    """
    用于数据集评估的PHP运行函数（带保存功能）

    Args:
        question: 问题文本
        gold_answer: 标准答案
        model_name: 模型名称
        base_prompt_type: 基础提示类型 (standard/cot/complex_cot)
        max_interactions: 最大交互次数
        temperature: 温度参数
        save_result: 是否保存结果
        dataset_name: 数据集名称
        sample_id: 样本ID
        results_dir: 结果保存目录
    block_size: 保存结果时单个 block 文件包含的样本数

    Returns:
        PHPResult对象
    """
    # 记录开始时间
    start_time = datetime.datetime.now()

    # 调用PHP方法求解
    result = solve_with_php(
        question=question,
        model_name=model_name,
        base_prompt_type=base_prompt_type,
        max_interactions=max_interactions,
        temperature=temperature,
    )

    # 计算运行时间
    end_time = datetime.datetime.now()
    runtime = (end_time - start_time).total_seconds()

    # 如果需要保存结果，按照统一的数据结构保存
    if save_result and sample_id:
        # 构建详细的交互记录
        rounds_data: Dict[str, Any] = {}
        rounds = list(zip(result.all_answers, result.all_responses))
        round_runtime = (
            runtime / result.interaction_count if result.interaction_count else runtime
        )

        for idx, (answer, response) in enumerate(rounds, 1):
            previous_answers = result.all_answers[: idx - 1]
            if idx == 1:
                system_prompt = get_prompt("", f"php_base_{result.base_prompt_type}")
                human_message = question
                hints = []
            else:
                system_prompt, human_message = get_php_continuation_prompt(
                    question, previous_answers, result.base_prompt_type
                )
                hints = previous_answers

            rounds_data[f"round_{idx}"] = {
                "system_status": {
                    "base_prompt_type": result.base_prompt_type,
                    "max_interactions": result.max_interactions,
                },
                "agent1_config": {
                    "model": result.model,
                    "temperature": result.model_config.get("temperature"),
                    "top_p": result.model_config.get("top_p"),
                    "frequency_penalty": result.model_config.get("frequency_penalty"),
                    "presence_penalty": result.model_config.get("presence_penalty"),
                    "repetition_penalty": result.model_config.get("repetition_penalty"),
                },
                "agent1_content": {
                    "prompt": {
                        "system_prompt": system_prompt,
                        "human_message": human_message,
                        "hints": hints,
                    },
                    "response": response,
                    "extracted_answer": answer,
                    "runtime": round_runtime,
                },
            }

        # 构建统一的数据结构
        sample_data = {
            "overview": {
                "dataset": dataset_name,
                "question": question,
                "gold_answer": gold_answer,
                "total_runtime": runtime,
                "total_rounds": result.interaction_count,
                "termination_reason": result.termination_reason,
                "timestamp": start_time.isoformat(),
            },
            "details": rounds_data,
        }

        # 保存结果
        save_record(
            data=sample_data,
            folder=results_dir,
            savename=sample_id,
            file_format="json",
            as_block=True,
            block_size=block_size,
        )

    return result


def test_php_on_math_dataset(
    num_samples: int = 50,
    random_seed: int = 42,
    model_name: str = "qwen",
    base_prompt_type: Literal["standard", "cot", "complex_cot"] = "cot",
    max_interactions: int = 5,
    temperature: float = 0.0,
    save_results: bool = True,
    results_dir: str = "results/6_singleself_math_qwen",
    dataset_name: str = "math",
    start_index: int = 0,
    chunk_size: int = -1,
    random_mode: bool = False,
    block_size: int = 1000,
):
    """
    在指定数据集上测试PHP方法

    Args:
        num_samples: 测试样本数量
        random_seed: 随机种子
        model_name: 模型名称
        base_prompt_type: 基础提示类型
        max_interactions: 最大交互次数
        temperature: 温度参数
        save_results: 是否保存结果
        results_dir: 结果保存目录
        dataset_name: 数据集名称
    start_index: 顺序模式起始索引
    chunk_size: 顺序模式每批样本数
    random_mode: 是否启用随机采样
    block_size: 保存结果时单个 block 文件包含的样本数
    """
    print("=" * 80)
    print("Start dataset test: Single-Agent Model with Self Reflection")
    print("=" * 80)
    print(f"\n📋 Test Parameter:")
    print(f"  - Dataset: {dataset_name.upper()}")
    print(f"  - Number of Samples: {num_samples}")
    print(f"  - Random Seed: {random_seed}")
    print(f"  - Model: {model_name}")
    print(f"  - Base Prompt Type: {base_prompt_type}")
    print(f"  - Max Interactions: {max_interactions}")
    print(f"  - Temperature: {temperature}")
    print(f"  - Save Results: {save_results}")
    print(f"  - Results Directory: {results_dir}")
    mode_label = "RANDOM" if random_mode else "SEQUENTIAL"
    print(f"  - Sampling Mode: {mode_label}")
    if not random_mode:
        print(f"  - Start Index: {start_index}")
        print(f"  - Chunk Size: {chunk_size if chunk_size != -1 else 'AUTO'}")
    print(f"  - Record Block Size: {block_size}")

    # 设置随机种子
    random.seed(random_seed)

    # 加载数据集
    print(f"\n📚 Loading {dataset_name.upper()} dataset...")
    loader = DatasetLoader(dataset_name)
    questions, answers = loader.load_data()
    print(f"✅ Dataset loaded with {len(questions)} samples")

    total_samples = len(questions)

    if total_samples == 0:
        print("⚠️ Dataset is empty. Abort.")
        return

    # 选择样本
    if random_mode:
        if num_samples == -1 or num_samples >= total_samples:
            selected_indices = list(range(total_samples))
        else:
            selected_indices = random.sample(range(total_samples), num_samples)
            selected_indices.sort()
        print(f"✅ Randomly selected {len(selected_indices)} samples")
    else:
        start_index = max(0, start_index)
        if start_index >= total_samples:
            print(
                f"⚠️ Start index {start_index} is out of range (dataset size {total_samples}). Abort."
            )
            return

        if chunk_size == -1:
            end_index = total_samples
        else:
            end_index = min(total_samples, start_index + max(0, chunk_size))

        if num_samples != -1:
            end_index = min(end_index, start_index + max(0, num_samples))

        selected_indices = list(range(start_index, end_index))

        if not selected_indices:
            print("⚠️ No samples selected for the specified range. Abort.")
            return

        print(
            f"✅ Sequentially selected {len(selected_indices)} samples from index {selected_indices[0]} to {selected_indices[-1]}"
        )

    effective_samples = len(selected_indices)
    selected_samples = [
        {"question": questions[i], "answer": answers[i]} for i in selected_indices
    ]

    # 创建结果目录
    if save_results:
        if random_mode:
            target_results_dir = results_dir
        else:
            chunk_start = selected_indices[0]
            chunk_end = selected_indices[-1]
            target_results_dir = os.path.join(
                results_dir,
                "chunks",
                f"chunk_{chunk_start:05d}-{chunk_end:05d}",
            )
        os.makedirs(target_results_dir, exist_ok=True)
        print(f"✅ Results directory ready: {target_results_dir}")
    else:
        target_results_dir = results_dir

    # 统计变量
    total_interactions = 0

    # 逐个测试样本
    print(f"\n🚀 Start test...\n")
    start_time = datetime.datetime.now()

    for idx, sample in enumerate(selected_samples, 1):
        question = sample["question"]
        gold_answer = sample["answer"]
        sample_index = selected_indices[idx - 1]
        sample_id = f"sample_{sample_index:05d}"

        print(f"\n{'='*80}")
        print(f"Sample {idx}/{num_samples} (ID: {sample_id})")
        print(f"{'='*80}")
        print(
            f"Question: {question[:100]}..."
            if len(question) > 100
            else f"Question: {question}"
        )
        print(f"Gold Answer: {gold_answer}")

        try:
            # 运行PHP方法
            result = run_php_for_dataset(
                question=question,
                gold_answer=gold_answer,
                model_name=model_name,
                base_prompt_type=base_prompt_type,
                max_interactions=max_interactions,
                temperature=temperature,
                save_result=save_results,
                dataset_name=dataset_name,
                sample_id=sample_id,
                results_dir=target_results_dir,
                block_size=block_size,
            )

            # 更新统计
            total_interactions += result.interaction_count

            # 打印结果
            print(f"\n✅ Completed!")
            print(f"  - Model Answer: {result.final_answer}")
            print(f"  - Interaction Count: {result.interaction_count}")
            print(f"  - Termination Reason: {result.termination_reason}")
            print(f"  - Answer Evolution: {' → '.join(result.all_answers)}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
            continue

        # 打印进度
        print(f"\n📊 Current Statistics:")
        print(f"  - Completed: {idx}/{effective_samples}")
        print(f"  - Average Interaction Count: {total_interactions/idx:.2f}")

    # 最终统计
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")
    print(f"\n📊 Final Statistics:")
    print(f"  - Total Samples: {effective_samples}")
    print(f"  - Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if effective_samples:
        print(
            f"  - Average Time per Sample: {total_time/effective_samples:.2f} seconds"
        )
        print(
            f"  - Average Interaction Count: {total_interactions/effective_samples:.2f}"
        )
    if save_results:
        print(f"\n💾 Results saved to: {target_results_dir}")
    else:
        print("\n💾 Results not saved (save_results=False)")
    print(f"{'='*80}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset Testing Script: Single-Agent Model with Self Reflection"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples (default: 50)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--model", type=str, default="qwen", help="Model name (default: qwen)"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="cot",
        choices=["standard", "cot", "complex_cot"],
        help="基础提示类型 (默认: cot)",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=3,
        help="Maximum number of interactions (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature parameter (default: 0.0)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/6_singleself_math_qwen",
        help="Results save directory, (default: results/6_singleself_math_qwen)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        help="Dataset name (default: math)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Sequential mode start index (default: 0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=-1,
        help="Sequential mode chunk size, -1 means auto (default: -1)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Enable random sampling mode (default: sequential)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1000,
        help="Block size used when saving records (default: 1000)",
    )

    args = parser.parse_args()

    config = {
        "num_samples": args.num_samples,
        "random_seed": args.random_seed,
        "model_name": args.model,
        "base_prompt_type": args.prompt_type,
        "max_interactions": args.max_interactions,
        "temperature": args.temperature,
        "save_results": not args.no_save,
        "results_dir": args.results_dir,
        "dataset_name": args.dataset,
        "start_index": args.start_index,
        "chunk_size": args.chunk_size,
        "random_mode": args.random,
        "block_size": args.block_size,
    }

    for key, value in USER_RUN_CONFIG.items():
        if value is not None:
            config[key] = value

    # 运行测试
    test_php_on_math_dataset(**config)


if __name__ == "__main__":
    main()
