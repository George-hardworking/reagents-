###################################################################################
# File: magents_gsm8k.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to test the multi-agent debate model on GSM8K dataset.
####################################################################################
import numpy as np
import random
import time
import os
import re
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from typing import Dict, List, Literal, Optional, Tuple, Any, Union
from volcenginesdkarkruntime import Ark
from dataclasses import dataclass, field
from datasets import load_dataset
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

random.seed(42)#要随机记得注释这里
np.random.seed(42)#要随机记得注释这里

ARK_API_KEY = os.getenv("ARK_API_KEY")
VOLC_ENDPOINTID = os.getenv("VOLC_ENDPOINTID")

ark_client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY
)

@dataclass
class DebateState:
    question: str
    agent1_answer: str = "" 
    agent2_answer: str = ""
    round: int = 0
    history: List[Dict] = field(default_factory=list)

def call_llm(system_prompt: str, human_message: str) -> str:
    """火山引擎ARK模型调用"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_message}
    ]
    # 调用API，不使用任何额外参数
    completion = ark_client.chat.completions.create(
        model=VOLC_ENDPOINTID,
        messages=messages,
        temperature=0.7
    )
    
    return completion.choices[0].message.content

system_prompt_initial = """Can you solve the following math problem? <Prblem> Explain your reasoning. Your final answer should be a single numerical number;, in the form \boxed{{answer}}, at the end of your response."""

system_prompt_debate = """These are the solutions to the problem from other agents: <other agent responses> Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is <Problem>. Your final answer should be a single numerical number, inthe form \boxed{{answer}}, at the end of your response."""

def agent_node(state: DebateState, agent_name: str) -> Dict[str, Any]:
    """每个智能体的通用节点函数。根据轮次和智能体名称决定行为。"""
    current_round = state.round
    other_agent_name = "Agent2" if agent_name == "Agent1" else "Agent1"
    other_answer = state.agent2_answer if agent_name == "Agent1" else state.agent1_answer

    if current_round == 0:
        system_prompt = system_prompt_initial
        human_message = state.question
    else:
        system_prompt = system_prompt_debate.replace(
            "<other agent responses>", 
            other_answer
        ).replace(
            "<Problem>", 
            state.question
        )
        human_message = ""

    answer = call_llm(system_prompt, human_message)

    if agent_name == "Agent1":
        return {"agent1_answer": answer}
    else:
        return {"agent2_answer": answer}

def increment_round(state: DebateState) -> Dict[str, Any]:
    """控制辩论轮次递增的函数"""
    return {"round": state.round + 1}

def should_continue(state: DebateState) -> Literal["debate_round", "__end__"]:
    """判断是否继续辩论的条件函数"""
    if state.round < 3: #辩论次数
        return "debate_round"
    else:
        return "__end__"

builder = StateGraph(DebateState)
builder.add_node("Agent1", lambda state: agent_node(state, "Agent1"))
builder.add_node("Agent2", lambda state: agent_node(state, "Agent2"))
builder.add_node("increment_round", increment_round)
builder.set_entry_point("Agent1")
builder.add_edge("Agent1", "Agent2")
builder.add_edge("Agent2", "increment_round")
builder.add_conditional_edges(
    "increment_round",
    should_continue,
    {
        "debate_round": "Agent1",
        "__end__": END
    }
)
builder.add_edge("Agent1", "Agent2")
builder.add_edge("Agent2", "increment_round")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

def run_debate(question: str):
    """运行辩论流程的主函数"""
    initial_state = DebateState(question=question, round=0)
    config = {"configurable": {"thread_id": 1}}
    
    print(f"User Question: {question}\n")
    print("="*50)
    
    final_state = None
    last_round = 0
    agent1_answer = ""
    agent2_answer = ""
    
    for event in graph.stream(initial_state, config, stream_mode="values"):
        current_round = event["round"]
        
        if "agent1_answer" in event:
            agent1_answer = event["agent1_answer"]
        if "agent2_answer" in event:
            agent2_answer = event["agent2_answer"]
        
        if current_round > last_round and agent1_answer and agent2_answer:
            print(f"Round {last_round} Result:")
            print(f"Agent1: {agent1_answer}")
            print(f"Agent2: {agent2_answer}")
            print("-" * 30)

            if final_state is not None:
                if hasattr(final_state,'history'):
                    final_state.history.append({
                        "round": last_round,
                        "agent1_answer": agent1_answer,
                        "agent2_answer": agent2_answer
                    })

            last_round = current_round
            agent1_answer = ""
            agent2_answer = ""
        
        final_state = event

    if agent1_answer and agent2_answer:
        print(f"Round {last_round} Result:")
        print(f"Agent1: {agent1_answer}")
        print(f"Agent2: {agent2_answer}")
        print("-" * 30)

    print(f"final answer - Agent1: {final_state['agent1_answer']}")
    print(f"final answer - Agent2: {final_state['agent2_answer']}")
    print("Debate finished!")
    print("="*50)
    return final_state

def normalize_math_expression(expr: str) -> Optional[str]:
    """标准化数学表达式"""
    if not expr:
        return None
    
    expr = expr.strip()
    
    # 处理LaTeX括号格式：\left( content \right) → (content)
    left_right_pattern = r'\\left\(([^)]+)\\right\)'
    left_right_match = re.search(left_right_pattern, expr)
    if left_right_match:
        content = left_right_match.group(1)
        return f"({content})"
    
    # 处理LaTeX简化分数格式：\frac83 → 8/3
    simplified_frac_pattern = r'\\frac\s*(-?\d+)\s*(-?\d+)'
    simplified_frac_match = re.search(simplified_frac_pattern, expr)
    if simplified_frac_match:
        numerator = simplified_frac_match.group(1)
        denominator = simplified_frac_match.group(2)
        return f"{numerator}/{denominator}"
    
    # 处理标准LaTeX分数格式：\frac{8}{3} → 8/3
    standard_frac_pattern = r'\\frac\{(-?\d+)\}\{(-?\d+)\}'
    standard_frac_match = re.search(standard_frac_pattern, expr)
    if standard_frac_match:
        numerator = standard_frac_match.group(1)
        denominator = standard_frac_match.group(2)
        return f"{numerator}/{denominator}"
    
    # 处理字母答案（选择题）- A-E
    letter_pattern = r'^[A-Ea-e]$'
    if re.match(letter_pattern, expr):
        return expr.upper()  # 统一为大写
    
    # 处理简单分数形式 a/b
    simple_frac_pattern = r'(-?\d+)/(-?\d+)'
    simple_frac_match = re.search(simple_frac_pattern, expr)
    if simple_frac_match:
        numerator = int(simple_frac_match.group(1))
        denominator = int(simple_frac_match.group(2))
        if denominator != 0:
            return f"{numerator}/{denominator}"
    
    # 移除数字序列中的空格
    sequence_pattern = r'(-?\d+)(?:,\s*(-?\d+))+'
    sequence_match = re.search(sequence_pattern, expr)
    if sequence_match:
        # 重新构建无空格的序列
        numbers = re.findall(r'-?\d+', expr)
        return ','.join(numbers)
    
    # 处理其他带空格数学表达式
    expr = re.sub(r'(\d)\s+([\\a-zA-Z])', r'\1\2', expr)  # 移除数字和字母/命令之间的空格
    expr = re.sub(r'([\\a-zA-Z])\s+(\d)', r'\1\2', expr)  # 移除字母/命令和数字之间的空格
    expr = re.sub(r'\s+,\s*', ',', expr)  # 移除逗号前后的空格
    expr = re.sub(r'\s*\(\s*', '(', expr)  # 移除左括号前后的空格
    expr = re.sub(r'\s*\)\s*', ')', expr)  # 移除右括号前后的空格
    return expr

def are_answers_equal(answer1, answer2, tolerance=1e-6) -> bool:
    """比较两个答案是否相等"""
    if answer1 is None or answer2 is None:
        return False
    
    # 标准化两个答案
    norm1 = normalize_math_expression(str(answer1))
    norm2 = normalize_math_expression(str(answer2))
    
    if norm1 is None or norm2 is None:
        return str(answer1).strip().lower() == str(answer2).strip().lower()
    
    # 如果都是字母，直接比较（不区分大小写）
    if re.match(r'^[A-Ea-e]$', norm1) and re.match(r'^[A-Ea-e]$', norm2):
        return norm1.upper() == norm2.upper()
    
    # 如果都是数字序列，比较序列
    if re.match(r'-?\d+(?:,-?\d+)+', norm1) and re.match(r'-?\d+(?:,-?\d+)+', norm2):
        return norm1 == norm2
    
    # 如果都是分数，比较分数值
    if '/' in norm1 and '/' in norm2:
        try:
            # 解析分数
            if '/' in norm1:
                num1, denom1 = map(int, norm1.split('/'))
                value1 = num1 / denom1
            else:
                value1 = float(norm1)
            
            if '/' in norm2:
                num2, denom2 = map(int, norm2.split('/'))
                value2 = num2 / denom2
            else:
                value2 = float(norm2)
            
            return abs(value1 - value2) < tolerance
        except (ValueError, ZeroDivisionError):
            # 如果数值转换失败，进行字符串比较
            pass
    
    # 尝试数值比较
    try:
        value1 = float(norm1) if '/' not in norm1 else float(norm1.split('/')[0]) / float(norm1.split('/')[1])
        value2 = float(norm2) if '/' not in norm2 else float(norm2.split('/')[0]) / float(norm2.split('/')[1])
        return abs(value1 - value2) < tolerance
    except (ValueError, ZeroDivisionError):
        # 如果数值转换失败，进行字符串比较
        return str(norm1).strip().lower() == str(norm2).strip().lower()

def get_test_samples(n_samples=50):
    """get the same test samples from GSM8K dataset"""
    dataset = load_dataset("openai/gsm8k", "main")#gsm8k测试集
    all_indices = list(range(len(dataset['test'])))
    random.seed(42)  #要随机记得注释这里
    selected_indices = random.sample(all_indices, n_samples)
    test_samples = dataset['test'].select(selected_indices)
    return test_samples

def test_gsm8k_debate(test_samples):
    """use debate model to answer gsm8k questions and evaluate accuracy"""
    correct_count = 0
    
    for i, sample in enumerate(test_samples):
        print(f"\n=== Test Sample {i+1} ===")
        print(f"Question: {sample['question']}")
        print(f"Correct Answer: {sample['answer']}")
        
        result = run_debate(sample['question'])
        
        predicted_answer = extract_number_from_text(result['agent1_answer'])
        true_answer = extract_number_from_text(sample['answer'])
        
        is_correct = (predicted_answer == true_answer)
        if is_correct:
            correct_count += 1
            
        print(f"Model Answer: {predicted_answer}, Correct: {is_correct}")
    
    accuracy = correct_count / len(test_samples)
    print(f"\nModel Accuracy Rate: {accuracy:.2%}")
    return accuracy

def extract_number_from_text(text):
    """从文本中提取数字"""
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

if __name__ == "__main__":
    test_samples = get_test_samples(1)
    
    # 测试多智能体辩论
    print("Testing Multi-agents Debate Model...")
    debate_accuracy = test_gsm8k_debate(test_samples)