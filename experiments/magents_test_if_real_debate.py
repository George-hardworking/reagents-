###################################################################################
# File: magents_test_if_real_debate.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to test whether if the model did a real debate.
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
    
    completion = ark_client.chat.completions.create(
        model=VOLC_ENDPOINTID,
        messages=messages,
        temperature=0.7
    )
    
    return completion.choices[0].message.content

system_prompt_initial = """请就以下议题发表您的观点：<Problem> 请详细阐述您的推理过程，并提供有力的论据支持您的立场。"""
system_prompt_debate = """这是其他智能体对此议题的观点：<other agent responses>。请考虑这些观点作为附加信息，给出您更新后的回应，可以同意、反驳或补充对方的论点。"""

def agent_node(state: DebateState, agent_name: str) -> Dict[str, Any]:
    """每个智能体的通用节点函数。根据轮次和智能体名称决定行为。"""
    current_round = state.round
    other_agent_name = "Agent2" if agent_name == "Agent1" else "Agent1"
    other_answer = state.agent2_answer if agent_name == "Agent1" else state.agent1_answer

    if current_round == 0:
        system_prompt = system_prompt_initial.replace("<Problem>", state.question)
        human_message = state.question
    else:
        system_prompt = system_prompt_debate.replace(
            "<other agent responses>", 
            other_answer
        ).replace(
            "<Problem>", 
            state.question
        )
        human_message = "请基于对方的观点给出回应"

    #打印prompt
    #print(f"\nRound {current_round} - {agent_name} Prompt:")
    #print(f"System Prompt: {system_prompt}")
    #print(f"Human Message: {human_message}")
    #print("-" * 40)

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

if __name__ == "__main__":
    question = "人工智能最终会取代人类的大部分工作，这对社会是利大于弊还是弊大于利？"
    
    print("开始辩论")
    result = run_debate(question)