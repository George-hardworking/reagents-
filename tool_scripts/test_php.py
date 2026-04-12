"""
测试Progressive-Hint Prompting (PHP)系统
"""

from agentvs.single_refletion import solve_with_php, run_example


def test_basic():
    """测试基本功能"""
    print("=" * 80)
    print("测试基本PHP功能")
    print("=" * 80)

    question = "What is 5 + 3?"

    result = solve_with_php(
        question=question,
        model_name="qwen",
        base_prompt_type="cot",
        max_interactions=3,
        temperature=0.0,
    )

    print(f"\n问题: {question}")
    print(f"最终答案: {result.final_answer}")
    print(f"交互次数: {result.interaction_count}")
    print(f"终止原因: {result.termination_reason}")
    print(f"答案历史: {result.all_answers}")


def test_example_from_doc():
    """测试文档中的示例"""
    print("\n" + "=" * 80)
    print("测试文档示例问题")
    print("=" * 80)

    question = (
        "There are 96 fourth-graders at Small Tree School. "
        "43 of them are girls. On Friday, 5 fourth-grade girls "
        "and 4 fourth-grade boys were absent. "
        "How many fourth grade boys were at Small Tree School on Friday?"
    )

    result = solve_with_php(
        question=question,
        model_name="qwen",
        base_prompt_type="cot",
        max_interactions=5,
        temperature=0.0,
    )

    print(f"\n问题: {question}")
    print(f"预期答案: 49")
    print(f"最终答案: {result.final_answer}")
    print(f"交互次数: {result.interaction_count}")
    print(f"答案演化: {' → '.join(result.all_answers)}")


if __name__ == "__main__":
    # 运行测试
    print("\n" + "#" * 80)
    print("# PHP系统测试")
    print("#" * 80 + "\n")

    try:
        # 测试1: 简单问题
        test_basic()

        # 测试2: 文档示例
        test_example_from_doc()

        print("\n" + "#" * 80)
        print("# 所有测试完成！")
        print("#" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
