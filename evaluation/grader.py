# agentvs/grader.py
import math
try:
    import sympy
except Exception:
    sympy = None


def _safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def grade_answer(pred: str, gold: str, tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    pred, gold = (pred or "").strip(), (gold or "").strip()

    # 1) 字符串完全相等
    if pred == gold:
        return True

    # 2) 浮点近似
    f1, f2 = _safe_float(pred), _safe_float(gold)
    if f1 is not None and f2 is not None:
        return math.isclose(f1, f2, rel_tol=tol, abs_tol=tol)

    # 3) 符号等价（可选）
    if sympy is not None:
        try:
            e1 = sympy.sympify(pred)
            e2 = sympy.sympify(gold)
            return sympy.simplify(e1 - e2) == 0
        except Exception:
            pass

    return False
