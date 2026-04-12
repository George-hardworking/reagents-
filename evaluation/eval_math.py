# agentvs/eval_math.py
import re

# --- 论文版思路：取最后一个 \boxed/\fbox，且支持嵌套大括号 ---
def _last_braced_payload(s: str, opener: str) -> str | None:
    if not s:
        return None
    pat = re.compile(rf"(?:{re.escape(opener)})\{{")
    pos = None
    for m in pat.finditer(s):
        pos = m.end()
    if pos is None:
        return None
    depth, i = 1, pos
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return s[pos:i-1].strip()

def last_boxed_only_string(sample: str) -> str | None:
    return _last_braced_payload(sample, r"\boxed")

def last_fboxed_only_string(sample: str) -> str | None:
    return _last_braced_payload(sample, r"\fbox")

def remove_boxed(s: str) -> str | None:
    # 论文里是直接切头尾；我们保留 payload（上面的函数已返回 payload）
    return s

# --- 我们额外需要的：GSM8K “####” 后抓取 + 末行兜底 + 更强数值/分式模式 ---
_NUM_PAT = re.compile(
    r"""
    (?P<num>
        -?                              # 负号
        (?:
            \d+(?:\.\d+)?               # 整数/小数
            |
            \d+/\d+                     # 纯整数字面分式 a/b
            |
            \\frac\{[^{}]+\}\{[^{}]+\}  # LaTeX 分式 \frac{a}{b}
        )
    )
    """, re.VERBOSE,
)

def _find_after_hashes(text: str) -> str | None:
    m = re.search(r"####\s*(.+)", text, flags=re.DOTALL)
    if not m:
        return None
    tail = m.group(1).strip()
    m2 = _NUM_PAT.search(tail)
    return m2.group("num") if m2 else tail.splitlines()[0].strip() or None

def _last_line_number(text: str) -> str | None:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1].rstrip("。.;,，、")
    m = _NUM_PAT.search(last)
    return m.group("num") if m else (last if last else None)

def parse_answer(text: str) -> str | None:
    if not text:
        return None
    ans = last_boxed_only_string(text)
    if ans is None:
        ans = last_fboxed_only_string(text)
    if ans is not None:
        return ans.strip()
    ans = _find_after_hashes(text)
    if ans is not None:
        return ans.strip()
    ans = _last_line_number(text)
    return ans.strip() if ans else None
