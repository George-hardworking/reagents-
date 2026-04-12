"""
快速测试PHP实现的脚本
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvs.single_refletion import solve_with_php

# 测试问题
test_question = (
    "There are 96 fourth-graders at Small Tree School. "
    "43 of them are girls. On Friday, 5 fourth-grade girls "
    "and 4 fourth-grade boys were absent. "
    "How many fourth grade boys were at Small Tree School on Friday?"
)

print("测试PHP系统")
print("=" * 80)
print(f"问题: {test_question}")
print(f"预期答案: 49")
print("=" * 80)

# 运行PHP
result = solve_with_php(
    question=test_question,
    model_name="qwen",
    base_prompt_type="cot",
    max_interactions=5,
    temperature=0.0,
)

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print(f"最终答案: {result.final_answer}")
print(f"交互次数: {result.interaction_count}")
print(f"终止原因: {result.termination_reason}")
