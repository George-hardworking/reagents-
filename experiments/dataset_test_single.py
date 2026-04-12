# experiments/dataset_test_single.py
import sys
import pathlib
import argparse
import random
import math
from typing import Optional

# import 路径
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from data.dataset_loader import DatasetLoader
from agentvs.single_agent import solve_once
from evaluation.math_normalize import normalize_answer
from evaluation import eval_math


def _extract_and_norm(s: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """返回 (解析后, 归一化后)"""
    if s is None:
        return None, None
    try:
        parsed = eval_math.parse_answer(s)
    except Exception:
        parsed = None
    core = parsed if parsed else s
    try:
        normed = normalize_answer(core)
    except Exception:
        normed = (core or "").strip()
    return parsed, normed


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _judge(pred_norm: Optional[str], gold_norm: Optional[str], tol: float = 1e-6) -> bool:
    """
    不用 grader；先比字符串相等，失败再做数值近似（两边都能转 float 才近似比较）。
    """
    if pred_norm is None or gold_norm is None:
        return False
    if pred_norm == gold_norm:
        return True
    f1, f2 = _safe_float(pred_norm), _safe_float(gold_norm)
    if f1 is not None and f2 is not None:
        return math.isclose(f1, f2, rel_tol=tol, abs_tol=tol)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math", choices=["math", "gsm8k"])
    ap.add_argument("--model", default="qwen", choices=["qwen", "volc_ark"])
    ap.add_argument("--prompt", default="cot")
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--random", action="store_true", help="随机打乱后取样")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    print("=" * 80)
    print(f"Dataset={args.dataset}  Model={args.model}  Prompt={args.prompt}")
    print(f"Range: start={args.start}, max={args.max}, random={args.random}, seed={args.seed}")
    print("Judge: strict equality, fallback to numeric ~=")
    print("=" * 80)

    loader = DatasetLoader(args.dataset)
    questions, golds = loader.load_data()
    n = len(questions)

    idx = list(range(n))
    if args.random:
        rnd = random.Random(args.seed)
        rnd.shuffle(idx)

    end = min(args.start + args.max, n)
    if args.start >= n:
        print("start 超出数据集长度")
        return

    run, correct = 0, 0

    for k in range(args.start, end):
        i = idx[k]
        q, gold_raw = questions[i], golds[i]
        print(f"\n🚀 Q{i+1}/{n}")
        try:
            r = solve_once(q, model_name=args.model, prompt_type=args.prompt)
            pred_raw = getattr(r, "answer", r)

            pred_parsed, pred_norm = _extract_and_norm(pred_raw)
            gold_parsed, gold_norm = _extract_and_norm(gold_raw)

            ok = _judge(pred_norm, gold_norm)
            run += 1
            correct += int(ok)
            acc = correct / run if run else 0.0

            print(f"[Gold-raw ] {str(gold_raw).strip()[:200]}")
            print(f"[Pred-raw ] {str(pred_raw).strip()[:200]}")
            print(f"[Gold->parsed] {gold_parsed}   [->norm] {gold_norm}")
            print(f"[Pred->parsed] {pred_parsed}   [->norm] {pred_norm}")
            print(f"[Final  ] {'✅ CORRECT' if ok else '❌ WRONG'}   [Acc] {correct}/{run} = {acc:.4f}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 80)

    print(f"\n🎯 DONE  Final Acc: {correct}/{run} = {correct/max(1,run):.4f}")


if __name__ == "__main__":
    main()
