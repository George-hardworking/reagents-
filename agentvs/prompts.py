def get_prompt(input_data: str, prompt_type: str = "standard") -> str:
    if prompt_type == "standard":
        return f"XXX"

    elif prompt_type == "cot":
        return f"XXX"

    elif prompt_type == "vote":
        return f"XXX"

    elif prompt_type == "compare":
        return f"XXX"

    elif prompt_type == "score":
        return f"XXX"
    # --- 多智能体辩论相关 ---
    elif prompt_type == "multi_agent_initial":
        return """Can you solve the following math problem? <Problem> Explain your reasoning. Your final answer should be a single numerical number;, in the form \\boxed{{answer}}, at the end of your response."""

    elif prompt_type == "multi_agent_debate":
        return """These are the solutions to the problem from other agents: <other agent responses> Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is <Problem>. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

    # --- PHP (Progressive-Hint Prompting) 相关 ---
    elif prompt_type == "php_base_standard":
        return "Answer the following question directly. Provide your final answer clearly. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."

    elif prompt_type == "php_base_cot":
        return "Let's think step by step to answer the following question. Show your reasoning process and provide your final answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."

    elif prompt_type == "php_base_complex_cot":
        return "Let's think step by step to solve this problem. Break down the problem into smaller parts and reason through each step carefully. Provide detailed explanations and your final answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def get_php_continuation_prompt(
    question: str, previous_answers: list, base_prompt_type: str = "cot"
) -> tuple[str, str]:
    """
    生成PHP continuation提示：将历史答案作为hints整合到提示中，引导LLM改进答案。

    提示格式：
    - System: "We know the Answer Hints: {hints}. With the Answer Hints: {hints},
              we will answer the question. {base_instruction}"
    - Human: "{question} (Hint: The answer is near to {hints})."

    Args:
        question: 原始问题
        previous_answers: 历史答案列表
        base_prompt_type: 基础提示类型 (standard/cot/complex_cot)

    Returns:
        (system_prompt, human_prompt) 元组
    """
    # 将历史答案格式化为hints字符串
    hints = ", ".join(previous_answers)

    # 根据base_prompt_type选择基础指令
    base_instructions = {
        "standard": "Answer the question directly.",
        "cot": "Let's think step by step.",
        "complex_cot": "Let's think step by step and break down the problem carefully.",
    }
    base_instruction = base_instructions.get(
        base_prompt_type, "Let's think step by step."
    )

    # 构建PHP格式的system prompt
    system_prompt = (
        f"We know the Answer Hints: {hints}. "
        f"With the Answer Hints: {hints}, we will answer the question. "
        f"{base_instruction}"
    )

    # 构建PHP格式的human prompt
    human_prompt = f"{question} (Hint: The answer is near to {hints})."

    return system_prompt, human_prompt

    # def get_multi_agent_prompts(dataset_type: str = "math"):
    """多智能体辩论相关"""
    if dataset_type == "math":
        return {
            "initial": """Can you solve the following math problem? <Problem> Explain your reasoning. Your final answer should be in the form \\boxed{{answer}} at the end of your response.""",
            "debate": """These are the solutions to the problem from other agents: <other agent responses> Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is <Problem>. Your final answer should be in the from \\boxed{{answer}}, at the end of your response.""",
        }
    elif dataset_type == "gsm8k":
        return {
            "initial": """Can you solve the following math problem? <Prblem> Explain your reasoning. Your final answer should be a single numerical number;, in the form \boxed{{answer}}, at the end of your response.""",
            "debate": """These are the solutions to the problem from other agents: <other agent responses> Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is <Problem>. Your final answer should be a single numerical number, inthe form \boxed{{answer}}, at the end of your response.""",
        }
    else:
        # 默认使用math的prompt
        return get_multi_agent_prompts("math")
