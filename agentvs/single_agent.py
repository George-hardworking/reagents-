from dataclasses import dataclass
from typing import Optional, Dict, Any
import datetime
from .model_manager import ModelManager
from core.prompts import get_prompt
from core.record import save_record

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise problem-solver. Answer directly and show minimal necessary working."


@dataclass
class SingleResult:
    question: str
    answer: str
    model: str
    system_prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


def solve_once(
    question: str,
    model_name: Optional[str] = None,
    prompt_type: Optional[str] = "standard",
) -> SingleResult:
    """单次解答"""
    model_manager = ModelManager()
    if model_name:
        model_manager.set_model(model_name)

    sys_prompt = get_prompt(question, prompt_type)  # 从 prompts.py 获取相应的 prompt
    human_message = question  # 用户问题作为消息

    # 调用模型并返回结果
    answer = model_manager.call_model(sys_prompt, human_message)

    return SingleResult(
        question=question,
        answer=answer,
        model=model_manager.current_model,
        system_prompt=sys_prompt,
        temperature=getattr(model_manager, "temperature", 0.0),
        top_p=getattr(model_manager, "top_p", 1.0),
        frequency_penalty=getattr(model_manager, "frequency_penalty", 0.0),
        presence_penalty=getattr(model_manager, "presence_penalty", 0.0),
        repetition_penalty=getattr(model_manager, "repetition_penalty", 1.0),
    )


def solve_batch(
    questions: list,
    model_name: Optional[str] = None,
    prompt_type: Optional[str] = "standard",
) -> list:
    """批量解答"""
    results = []
    for question in questions:
        result = solve_once(question, model_name, prompt_type)
        results.append(result)
    return results


def run_single_agent_for_dataset(
    question: str,
    gold_answer: str,
    model_name: str = "qwen",
    prompt_type: str = "cot",
    save_result: bool = False,
    dataset_name: str = "unknown",
    sample_id: str = None,
    results_dir: str = "results",
) -> SingleResult:
    """用于数据集评估的单智能体运行函数（带保存功能）"""

    # 记录开始时间
    start_time = datetime.datetime.now()

    # 调用单智能体求解
    result = solve_once(question, model_name, prompt_type)

    # 计算运行时间
    end_time = datetime.datetime.now()
    runtime = (end_time - start_time).total_seconds()

    # 如果需要保存结果，按照统一的数据结构保存（与多智能体保持一致）
    if save_result and sample_id:
        # 构建统一的数据结构，使用 round_1 而不是 single_agent
        sample_data = {
            "overview": {
                "dataset": dataset_name,
                "question": question,
                "gold_answer": gold_answer,
                "total_runtime": runtime,
                "total_rounds": 1,  # single agent 只有1轮
                "timestamp": start_time.isoformat(),
            },
            "details": {
                "round_1": {  # 改为 round_1，与多智能体保持一致
                    "agent1_content": {  # 使用 agent1_content，与多智能体保持一致
                        "agent_config": {
                            "model": result.model,
                            "temperature": result.temperature,
                            "top_p": result.top_p,
                            "frequency_penalty": result.frequency_penalty,
                            "presence_penalty": result.presence_penalty,
                            "repetition_penalty": result.repetition_penalty,
                        },
                        "prompt": {
                            "system_prompt": result.system_prompt,
                            "human_message": question,
                        },
                        "response": result.answer,
                        "runtime": runtime,
                    },
                }
            },
        }

        save_record(
            data=sample_data,
            folder=results_dir,
            savename=sample_id,
            file_format="json",
            as_block=True,
            block_size=1000,
        )

    return result
