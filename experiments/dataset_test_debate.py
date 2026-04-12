###################################################################################
# File: dataset_test_debate.py
# File Manager: Kaibiao ZHU
# Description: This file is to test the multi-agent debate model on datasets like GSM8K and MATH, evaluating accuracy using agentvs standards.
####################################################################################

import argparse
import datetime
import pathlib
import random
import json
import sys
import os
from agentvs.experiment_recorder import ExperimentRecorder
from agentvs import run_multi_agent_debate_for_dataset
from agentvs.dataset_loader import DatasetLoader 
from datasets import load_dataset
from dotenv import load_dotenv
from typing import Optional



ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

def get_test_samples(dataset_name="gsm8k", n_samples=50, seed=42):
    """get test samples using DatasetLoader with random sampling"""
    loader = DatasetLoader(dataset_name)
    questions, answers = loader.load_data()
    random.seed(seed)

    if n_samples == -1 or n_samples >= len(questions):
        selected_indices = list(range(len(questions)))
    else:
        all_indices = list(range(len(questions)))
        selected_indices = random.sample(all_indices, n_samples)
        selected_indices.sort()
    
    selected_questions = [questions[i] for i in selected_indices]
    selected_answers = [answers[i] for i in selected_indices]

    test_samples = [
        {"question": q, "answer": a, "index": idx}
        for idx, q, a in zip(selected_indices, selected_questions, selected_answers)
    ]
    return test_samples

def test_gsm8k_debate(test_samples, model1_name, model2_name, dataset_name, save_results, results_dir):
    """use debate model to answer gsm8k questions and evaluate accuracy using agentvs standards"""
    correct_count_agent1 = 0
    correct_count_agent2 = 0
    correct_count_overall = 0 
    results = []  
    experiment_data_list = [] 

    if save_results and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    recorder = ExperimentRecorder(results_dir=results_dir)

    for i, sample in enumerate(test_samples):
        print(f"\n=== Test Sample {i+1}/{len(test_samples)} ===")
        print(f"Question: {sample['question']}")
        print(f"Correct Answer (raw): {sample['answer']}")

        result = run_multi_agent_debate_for_dataset(
            question=sample["question"],
            gold_answer=sample["answer"],
            model_name_agent1=model1_name,
            model_name_agent2=model2_name,
            save_result=save_results,
            dataset_name=dataset_name,
            sample_id=f"sample_{sample.get('index', i):05d}",
        )

        if result.agent1_correct:
            correct_count_agent1 += 1
        if result.agent2_correct:
            correct_count_agent2 += 1

        if result.agent1_correct and result.agent2_correct:
            correct_count_overall += 1

        acc1 = correct_count_agent1 / (i + 1)
        acc2 = correct_count_agent2 / (i + 1)

        print(f"Agent1 (Extracted): {result.agent1_final_extracted} -> Normalized: {result.agent1_normalized} -> {'✅ CORRECT' if result.agent1_correct else '❌ WRONG'} [Acc: {acc1:.4f}]")
        print(f"Agent2 (Extracted): {result.agent2_final_extracted} -> Normalized: {result.agent2_normalized} -> {'✅ CORRECT' if result.agent2_correct else '❌ WRONG'} [Acc: {acc2:.4f}]")
        print(f"Agent1 Full Answer: {result.agent1_raw}")
        print(f"Agent2 Full Answer: {result.agent2_raw}")
        print(f"Gold Answer: {result.gold_answer}")

        results.append(result)
        experiment_data_list.append(result.experiment_data) 

    total = len(test_samples)
    final_acc1 = correct_count_agent1 / total if total > 0 else 0
    final_acc2 = correct_count_agent2 / total if total > 0 else 0
    final_acc_overall = correct_count_overall / total if total > 0 else 0

    print(f"\n🎯 DATASET EVALUATION COMPLETE")
    print(f"[Agent1 ({model1_name})] Final Acc: {correct_count_agent1}/{total} = {final_acc1:.4f}")
    print(f"[Agent2 ({model2_name})] Final Acc: {correct_count_agent2}/{total} = {final_acc2:.4f}")
    print(f"[Overall Model] Final Acc: {correct_count_overall}/{total} = {final_acc_overall:.4f}")

    if save_results:
        agent1_temperature = None
        agent2_temperature = None
        if experiment_data_list and isinstance(experiment_data_list[0], dict):
            agent1_temperature = experiment_data_list[0].get("agent1_temperature")
            agent2_temperature = experiment_data_list[0].get("agent2_temperature")
        record_data = {
            "record": {
                "total_samples": total,
                "dataset": dataset_name,
                "model1": model1_name,
                "model2": model2_name,
                "agent1_accuracy": final_acc1,
                "agent2_accuracy": final_acc2,
                "overall_accuracy": final_acc_overall,
                "correct_count_agent1": correct_count_agent1,
                "correct_count_agent2": correct_count_agent2,
                "correct_count_overall": correct_count_overall,
                "agent1_temperature": agent1_temperature,
                "agent2_temperature": agent2_temperature
            },
            "detailed_results": experiment_data_list
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_filename = os.path.join(
            results_dir,
            f"record_{dataset_name}_{model1_name}_{model2_name}_{timestamp}.json"
        )   #文件命名格式
        with open(record_filename, 'w', encoding='utf-8') as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {record_filename}")

    return results

def main():
    # ==================== 配置选项 ====================
    dataset_name = "math"  # 选择数据集: "gsm8k" 或 "math"
    model1_name = "qwen"      # Agent1 使用的模型
    model2_name = "volc_ark"  # Agent2 使用的模型
    seed = 42  # 随机种子
    n_samples = 999     # 选择样本数量，=-1则表示全部样本
    save_results = True  # 是否保存结果
    results_dir = "Result"  # 结果保存目录
    # ==================== 配置选项 ====================

    print("=" * 100)
    print(f"Dataset={dataset_name.upper()}  Model1={model1_name}  Model2={model2_name}")
    print(f"Samples: {n_samples if n_samples != -1 else 'ALL'}, Seed: {seed}")
    print("=" * 100)

    test_samples = get_test_samples(dataset_name=dataset_name, n_samples=n_samples, seed=seed)
    test_gsm8k_debate(test_samples, model1_name, model2_name, dataset_name, save_results, results_dir)

if __name__ == "__main__":
    main()