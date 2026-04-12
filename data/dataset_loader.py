# agentvs/dataset_loader.py
from typing import Tuple, List
from datasets import load_dataset
from evaluation.math_normalize import normalize_answer


class DatasetLoader:
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type

    def load_data(self) -> Tuple[List[str], List[str]]:
        """根据 dataset_type 加载数据"""
        if self.dataset_type == "math":
            return self.load_math_data()
        elif self.dataset_type == "gsm8k":
            return self.load_gsm8k_data()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def load_math_data(self) -> Tuple[List[str], List[str]]:
        """加载 MATH 数据集"""
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
        qs, ans = [], []
        for item in ds:
            q = item.get("problem") or item.get("question") or ""
            a = item.get("solution") or item.get("answer") or ""
            qs.append(q)
            ans.append(a)
        return qs, ans

    def load_gsm8k_data(self) -> Tuple[List[str], List[str]]:
        """加载 GSM8K 数据集"""
        ds = load_dataset("openai/gsm8k", "main", split="train")
        qs = [x["question"] for x in ds]
        ans = [x["answer"] for x in ds]  # ⚠ GSM8K 用 answer 字段
        return qs, ans

    def process_answer(self, model_output: str) -> str:
        """调用 math_normalize 进行标准化"""
        return normalize_answer(model_output)
