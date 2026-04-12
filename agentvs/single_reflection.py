###################################################################################
# Single-Agent Model Using the Self-Reflection Function of the PHP System
# File Manager: Kaibiao ZHU
# Description: Using LangGraph to implement a single-agent reflection model based on the PHP method
###################################################################################

import re
import os
import sys
import datetime

from dataclasses import dataclass
from core.record import save_record
from core.prompts import get_prompt, get_php_continuation_prompt
from langgraph.graph import StateGraph, END
from .model_manager import ModelManager
from typing_extensions import Annotated
from typing import TypedDict, List, Literal, Optional, Dict, Any, Union, cast
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


###################################################################################
#                           状态定义
###################################################################################


class AgentState(TypedDict):
    question: str  # 原始问题
    previous_answers: List[str]  # 历史答案列表（用作hints）
    current_answer: str  # 当前答案
    base_prompt_type: Literal["standard", "cot", "complex_cot"]  # 基础提示类型
    interaction_count: int  # 当前交互次数
    max_interactions: int  # 最大交互次数限制
    final_answer: str  # 最终答案
    terminated: bool  # 是否已终止
    termination_reason: str  # 终止原因
    raw_responses: List[str]  # 存储每轮的完整响应


@dataclass
class PHPResult:
    """PHP系统运行结果"""

    question: str
    final_answer: str
    all_answers: List[str]
    all_responses: List[str]
    interaction_count: int
    termination_reason: str
    base_prompt_type: str
    model: str
    timestamp: str
    max_interactions: int
    model_config: Dict[str, float]


###################################################################################
#                           辅助函数
###################################################################################


def extract_final_answer(response: str) -> str:
    if not response or not isinstance(response, str):
        return ""

    # 尝试匹配 \\boxed{answer} 格式
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    boxed_match = re.search(boxed_pattern, response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 尝试匹配 "The answer is X" 格式
    answer_is_pattern = r"[Tt]he answer is[:\s]+([^\n\.]+)"
    answer_is_match = re.search(answer_is_pattern, response)
    if answer_is_match:
        return answer_is_match.group(1).strip()

    # 尝试匹配 "Therefore, X" 或 "Thus, X" 格式
    therefore_pattern = r"[Tt]herefore[,\s]+([^\n\.]+)"
    therefore_match = re.search(therefore_pattern, response)
    if therefore_match:
        return therefore_match.group(1).strip()

    thus_pattern = r"[Tt]hus[,\s]+([^\n\.]+)"
    thus_match = re.search(thus_pattern, response)
    if thus_match:
        return thus_match.group(1).strip()

    # 尝试提取最后一行的数字
    lines = response.strip().split("\n")
    for line in reversed(lines):
        # 查找数字（支持整数和小数）
        number_pattern = r"\b\d+\.?\d*\b"
        numbers = re.findall(number_pattern, line)
        if numbers:
            return numbers[-1]

    # 如果都失败，返回最后一行（去除标点）
    if lines:
        last_line = lines[-1].strip()
        # 移除末尾标点
        last_line = re.sub(r"[\.!?;,]+$", "", last_line)
        return last_line

    return response.strip()


def is_valid_answer(answer: str) -> bool:
    if not answer or not isinstance(answer, str):
        return False

    answer_lower = answer.lower().strip()

    # 无效表达列表
    invalid_phrases = [
        "i don't know",
        "i do not know",
        "cannot determine",
        "cannot answer",
        "insufficient information",
        "not enough information",
        "unable to answer",
        "无法确定",
        "不知道",
        "无法回答",
        "信息不足",
    ]

    # 检查是否包含无效表达
    for phrase in invalid_phrases:
        if phrase in answer_lower:
            return False

    # 检查答案长度（太短可能无效）
    if len(answer_lower) < 1:
        return False

    return True


###################################################################################
#                           节点函数（带重试机制）
###################################################################################


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def call_llm_with_retry(
    model_manager: ModelManager, system_prompt: str, human_message: str
) -> str:
    """
    带重试机制的LLM调用

    使用tenacity库实现最多3次重试，指数退避策略。

    Args:
        model_manager: 模型管理器实例
        system_prompt: 系统提示
        human_message: 用户消息

    Returns:
        LLM的响应文本

    Raises:
        Exception: 如果3次重试都失败
    """
    return model_manager.call_model(system_prompt, human_message)


def generate_base_answer(
    state: AgentState, model_manager: ModelManager
) -> Dict[str, Any]:
    """
    节点1: 生成基础答案

    PHP方法的第一阶段，使用基础提示（standard/cot/complex_cot）生成初始答案。
    该答案将作为后续反思的起点。

    工作流程：
    1. 根据base_prompt_type获取相应的系统提示
    2. 调用LLM生成答案（带重试机制）
    3. 提取最终答案
    4. 更新状态：添加到previous_answers，设置current_answer，增加计数

    Args:
        state: 当前状态
        model_manager: 模型管理器

    Returns:
        包含更新字段的状态字典
    """
    print(f"\n{'='*80}")
    print(f"[Round] Generating base answer (using {state['base_prompt_type']} prompt)")
    print(f"{'='*80}")

    question = state["question"]
    prompt_type = state["base_prompt_type"]

    # 从prompts.py获取基础提示
    system_prompt = get_prompt("", f"php_base_{prompt_type}")
    human_message = question

    print(f"\n📝 Question: {question}")
    print(f"\n🔧 System prompt: {system_prompt}")

    try:
        # 调用LLM（带重试）
        response = call_llm_with_retry(model_manager, system_prompt, human_message)
        print(f"\n💬 LLM response:\n{response}")

        # 提取答案
        answer = extract_final_answer(response)
        print(f"\n✅ Extracted Answer: {answer}")

        # 更新状态
        return {
            "previous_answers": [answer],
            "current_answer": answer,
            "interaction_count": 1,
            "raw_responses": [response],
        }

    except Exception as e:
        print(f"\n❌ Error generating base answer: {e}")
        # 返回错误状态
        return {
            "current_answer": f"Error: {str(e)}",
            "previous_answers": [],
            "interaction_count": 1,
            "terminated": True,
            "termination_reason": f"Error in base answer generation: {str(e)}",
            "raw_responses": [],
        }


def check_termination(state: AgentState) -> Dict[str, Any]:
    """
    节点2: 检查终止条件

    判断是否应该停止PHP交互过程。这是PHP方法的关键机制。

    终止条件（按优先级）：
    1. 达到最大交互次数
    2. 当前答案无效（模型无法回答）
    3. 最近三次答案相同（收敛）

    Args:
        state: 当前状态

    Returns:
        包含terminated和termination_reason的状态字典
    """
    print(f"\n{'='*80}")
    print(f"[Check] Checking termination conditions")
    print(f"{'='*80}")

    interaction_count = state["interaction_count"]
    max_interactions = state["max_interactions"]
    current_answer = state["current_answer"]
    previous_answers = state["previous_answers"]

    print(f"\n📊 Current interaction count: {interaction_count}/{max_interactions}")
    print(f"📊 Previous answers: {previous_answers}")
    print(f"📊 Current answer: {current_answer}")

    # 条件1: 检查是否达到最大交互次数
    if interaction_count >= max_interactions:
        print(f"\n🛑 Reached maximum interactions limit")
        return {
            "terminated": True,
            "termination_reason": f"Reached maximum interactions ({max_interactions})",
        }

    # 条件2: 检查当前答案是否有效
    if not is_valid_answer(current_answer):
        print(f"\n🛑 Current answer is invalid")
        return {
            "terminated": True,
            "termination_reason": "Invalid answer (model cannot determine)",
        }

    # 条件3: 检查是否收敛（最近三次答案相同）
    if len(previous_answers) >= 3:
        recent_answers = [
            ans.strip().lower() if isinstance(ans, str) else str(ans)
            for ans in previous_answers[-3:]
        ]
        if len(set(recent_answers)) == 1:
            print(f"\n🎯 Answers have converged (same answer three times)")
            return {
                "terminated": True,
                "termination_reason": "Converged (same answer three times)",
            }

    # 未满足任何终止条件，继续
    print(f"\n▶️ Continuing to next interaction")
    return {"terminated": False, "termination_reason": ""}


def generate_next_answer(
    state: AgentState, model_manager: ModelManager
) -> Dict[str, Any]:
    """
    节点3: 生成下一轮答案

    PHP方法的第二阶段核心：使用历史答案作为hints，生成改进的答案。

    工作流程：
    1. 使用generate_php_prompt构建特殊格式的提示
    2. 调用LLM生成新答案（带重试）
    3. 提取最终答案
    4. 更新状态：添加到previous_answers，更新current_answer，增加计数

    Args:
        state: 当前状态
        model_manager: 模型管理器

    Returns:
        包含更新字段的状态字典
    """
    print(f"\n{'='*80}")
    print(f"[Round] Generating next answer (round {state['interaction_count'] + 1})")
    print(f"{'='*80}")

    question = state["question"]
    previous_answers = state["previous_answers"]
    base_prompt_type = state["base_prompt_type"]

    # 从prompts.py生成PHP continuation提示
    system_prompt, human_prompt = get_php_continuation_prompt(
        question, previous_answers, base_prompt_type
    )

    print(f"\n📝 Question: {question}")
    print(f"\n💡 Hints: {', '.join(previous_answers)}")
    print(f"\n🔧 System prompt: {system_prompt}")
    print(f"\n🔧 User prompt: {human_prompt}")

    try:
        # 调用LLM（带重试）
        response = call_llm_with_retry(model_manager, system_prompt, human_prompt)
        print(f"\n💬 Response:\n{response}")

        # 提取答案
        answer = extract_final_answer(response)
        print(f"\n✅ Extracted Answer: {answer}")
        # 更新状态
        updated_previous_answers = previous_answers + [answer]
        updated_responses = state["raw_responses"] + [response]

        return {
            "previous_answers": updated_previous_answers,
            "current_answer": answer,
            "interaction_count": state["interaction_count"] + 1,
            "raw_responses": updated_responses,
        }

    except Exception as e:
        print(f"\n❌ Error generating next answer: {e}")
        return {
            "current_answer": f"Error: {str(e)}",
            "terminated": True,
            "termination_reason": f"Error in next answer generation: {str(e)}",
        }


def finalize_answer(state: AgentState) -> Dict[str, Any]:
    """
    节点4: 最终化答案

    当系统决定终止时，确定最终答案。通常选择最后一个有效答案。

    工作流程：
    1. 从previous_answers中选择最终答案（最后一个）
    2. 标记terminated为True
    3. 确保termination_reason已记录

    Args:
        state: 当前状态

    Returns:
        包含final_answer和terminated标志的状态字典
    """
    print(f"\n{'='*80}")
    print(f"[Finalize] Finalizing answer")
    print(f"{'='*80}")

    previous_answers = state["previous_answers"]

    # 选择最终答案（通常是最后一个）
    if previous_answers:
        final_answer = previous_answers[-1]
    else:
        final_answer = state.get("current_answer", "No answer generated")

    print(f"\n🎯 Final answer: {final_answer}")
    print(f"\n📊 All previous answers: {previous_answers}")
    print(f"\n🛑 Termination reason: {state.get('termination_reason', 'Unknown')}")

    return {"final_answer": final_answer, "terminated": True}


def route_after_check(state: AgentState) -> str:
    """
    路由函数: 检查后决定下一步

    根据terminated标志决定流程走向：
    - 如果terminated为True，流向finalize节点
    - 否则，流向generate_next节点继续交互

    Args:
        state: 当前状态

    Returns:
        下一个节点的名称
    """
    if state.get("terminated", False):
        return "finalize"
    else:
        return "generate_next"


###################################################################################
#                           工作流构建
###################################################################################


def build_php_workflow(model_manager: ModelManager):
    """
    构建PHP工作流

    工作流结构：
    START → generate_base → check → [route] → generate_next → check → ... → finalize → END
                                      ↓
                                   finalize → END

    节点说明：
    - generate_base: 生成基础答案（第一轮）
    - check: 检查终止条件
    - generate_next: 生成下一轮答案（使用PHP提示）
    - finalize: 最终化答案

    边说明：
    - generate_base → check: 无条件转移
    - check → [route]: 条件转移，根据terminated决定
    - generate_next → check: 形成循环
    - finalize → END: 结束

    Args:
        model_manager: 模型管理器实例

    Returns:
        编译后的StateGraph工作流
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点（使用lambda包装以传递model_manager）
    workflow.add_node(
        "generate_base",
        lambda state: generate_base_answer(cast(AgentState, state), model_manager),
    )
    workflow.add_node("check", check_termination)
    workflow.add_node(
        "generate_next",
        lambda state: generate_next_answer(cast(AgentState, state), model_manager),
    )
    workflow.add_node("finalize", finalize_answer)

    # 设置入口点
    workflow.set_entry_point("generate_base")

    # 添加边
    workflow.add_edge("generate_base", "check")
    workflow.add_conditional_edges(
        "check",
        route_after_check,
        {"generate_next": "generate_next", "finalize": "finalize"},
    )
    workflow.add_edge("generate_next", "check")
    workflow.add_edge("finalize", END)

    # 编译工作流
    return workflow.compile()


###################################################################################
#                           主函数接口
###################################################################################


def solve_with_php(
    question: str,
    model_name: str = "qwen",
    base_prompt_type: Literal["standard", "cot", "complex_cot"] = "cot",
    max_interactions: int = 5,
    temperature: float = 0.0,
) -> PHPResult:
    """
    使用PHP方法求解问题

    这是系统的主要入口函数，封装了整个PHP工作流。

    Args:
        question: 要求解的问题
        model_name: 使用的模型名称（qwen或volc_ark）
        base_prompt_type: 基础提示类型（standard/cot/complex_cot）
        max_interactions: 最大交互次数
        temperature: 模型温度参数

    Returns:
        PHPResult对象，包含完整的求解结果

    Example:
        >>> result = solve_with_php(
        ...     question="What is 2+2?",
        ...     model_name="qwen",
        ...     base_prompt_type="cot",
        ...     max_interactions=3
        ... )
        >>> print(result.final_answer)
        4
    """
    print(f"\n{'#'*80}")
    print(f"# Single-Agent Model with Self-Reflection Function Started")
    print(f"{'#'*80}")
    print(f"\n📋 Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Base prompt type: {base_prompt_type}")
    print(f"  - Max interactions: {max_interactions}")
    print(f"  - Temperature: {temperature}")
    print(f"\n❓ Question: {question}")

    # 初始化模型管理器
    model_manager = ModelManager(
        model_name=model_name, verbose=True, temperature=temperature
    )

    # 构建工作流
    app = build_php_workflow(model_manager)

    # 初始化状态
    initial_state: AgentState = {
        "question": question,
        "previous_answers": [],
        "current_answer": "",
        "base_prompt_type": base_prompt_type,
        "interaction_count": 0,
        "max_interactions": max_interactions,
        "final_answer": "",
        "terminated": False,
        "termination_reason": "",
        "raw_responses": [],
    }

    # 运行工作流
    start_time = datetime.datetime.now()
    final_state = app.invoke(initial_state)
    end_time = datetime.datetime.now()

    # 构建结果
    result = PHPResult(
        question=question,
        final_answer=final_state["final_answer"],
        all_answers=final_state["previous_answers"],
        all_responses=final_state["raw_responses"],
        interaction_count=final_state["interaction_count"],
        termination_reason=final_state["termination_reason"],
        base_prompt_type=base_prompt_type,
        model=model_manager.get_real_model_name(),
        timestamp=start_time.isoformat(),
        max_interactions=max_interactions,
        model_config={
            "temperature": model_manager.temperature,
            "top_p": model_manager.top_p,
            "frequency_penalty": model_manager.frequency_penalty,
            "presence_penalty": model_manager.presence_penalty,
            "repetition_penalty": model_manager.repetition_penalty,
        },
    )

    # 打印总结
    runtime = (end_time - start_time).total_seconds()
    print(f"\n{'#'*80}")
    print(f"# Single-Agent Model with Self-Reflection Function Completed")
    print(f"{'#'*80}")
    print(f"\n📊 Run statistics:")
    print(f"  - Total interactions: {result.interaction_count}")
    print(f"  - Runtime: {runtime:.2f} seconds")
    print(f"  - Termination reason: {result.termination_reason}")
    print(f"\n🎯 Final answer: {result.final_answer}")
    print(f"\n📝 Answer evolution:")
    for i, answer in enumerate(result.all_answers, 1):
        print(f"  Round {i}: {answer}")
    # 打印token统计
    token_stats = model_manager.get_token_stats()
    print(f"\n💰 Token usage statistics:")
    print(f"  - Input tokens: {token_stats['input_tokens']}")
    print(f"  - Output tokens: {token_stats['output_tokens']}")
    print(f"  - Total tokens: {token_stats['total_tokens']}")
    print(f"  - API calls: {token_stats['api_calls']}")

    return result


###################################################################################
#                           示例使用
###################################################################################


def run_example():
    """
    运行示例

    使用文档中提供的示例问题测试系统。
    """
    print("\n" + "=" * 80)
    print("运行PHP示例")
    print("=" * 80)

    # 示例问题-用于测试
    example_question = (
        "There are 96 fourth-graders at Small Tree School. "
        "43 of them are girls. On Friday, 5 fourth-grade girls "
        "and 4 fourth-grade boys were absent. "
        "How many fourth grade boys were at Small Tree School on Friday?"
    )

    print(f"\n示例问题:\n{example_question}")
    print(f"\n预期答案: 49")
    print(f"\n开始求解...\n")

    # 测试不同的base_prompt_type
    prompt_types: List[Literal["standard", "cot", "complex_cot"]] = [
        "standard",
        "cot",
        "complex_cot",
    ]
    for prompt_type in prompt_types:
        print(f"\n{'='*80}")
        print(f"测试 {prompt_type} 提示类型")
        print(f"{'='*80}")

        result = solve_with_php(
            question=example_question,
            model_name="qwen",
            base_prompt_type=prompt_type,
            max_interactions=5,
            temperature=0.0,
        )

        print(f"\n结果: {result.final_answer}")
        print(f"交互次数: {result.interaction_count}")
        print(f"终止原因: {result.termination_reason}")
        print("\n" + "-" * 80 + "\n")


def run_batch_test(questions: List[str], gold_answers: List[str]) -> List[PHPResult]:
    """
    批量测试

    对多个问题运行PHP方法。

    Args:
        questions: 问题列表
        gold_answers: 标准答案列表

    Returns:
        结果列表
    """
    results = []

    for i, (question, gold_answer) in enumerate(zip(questions, gold_answers), 1):
        print(f"\n{'#'*80}")
        print(f"# 问题 {i}/{len(questions)}")
        print(f"{'#'*80}")

        result = solve_with_php(
            question=question,
            model_name="qwen",
            base_prompt_type="cot",
            max_interactions=5,
        )

        results.append(result)

        # 简单的准确性检查（字符串匹配）
        is_correct = (
            result.final_answer.strip().lower() == str(gold_answer).strip().lower()
        )
        print(f"\n✓ 标准答案: {gold_answer}")
        print(f"✓ 模型答案: {result.final_answer}")
        print(f"✓ 是否正确: {'✅' if is_correct else '❌'}")

    return results


# 主程序入口
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Single-Agent Model Using the Self-Reflection Function of the PHP System")
    print("=" * 80)

    # 运行示例
    run_example()

    print("\n\n" + "=" * 80)
    print("Example Run Completed!")
    print("=" * 80)
