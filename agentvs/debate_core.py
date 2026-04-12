###################################################################################
# File: debate_core_debate.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to implement the core logic of multi-agent debate using different LLM models for each agent.
####################################################################################
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from typing import Dict, List, Literal, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from .model_manager import ModelManager

@dataclass
class DebateState:
    question: str
    agent1_answer: str = ""
    agent2_answer: str = ""
    round: int = 0
    history: List[Dict] = field(default_factory=list)

system_prompt_debate = """These are the solutions to the problem from other agents: <other agent responses>. Using the solutions from other agents as additional information, can you give an updated response...."""

def agent_node(state: DebateState, agent_name: str, model_manager_agent1: ModelManager, model_manager_agent2: ModelManager) -> Dict[str, Any]:
    """每个智能体的通用节点函数。根据轮次和智能体名称决定行为。"""
    current_round = state.round
    other_agent_name = "Agent2" if agent_name == "Agent1" else "Agent1"
    other_answer = state.agent2_answer if agent_name == "Agent1" else state.agent1_answer

    if current_round == 0:
        system_prompt = state.question
        human_message = state.question
    else:
        system_prompt = system_prompt_debate.replace(
            "<other agent responses>", 
            other_answer
        )
        human_message = ""

    # 根据智能体名称在模型管理器中选择不同的模型
    if agent_name == "Agent1":
        print(f"Agent1 使用模型: {model_manager_agent1.current_model}")
        answer = model_manager_agent1.call_model(system_prompt, human_message)
    else:
        print(f"Agent2 使用模型: {model_manager_agent2.current_model}")
        answer = model_manager_agent2.call_model(system_prompt, human_message)

    if agent_name == "Agent1":
        return {"agent1_answer": answer}
    else:
        return {"agent2_answer": answer}  
    # 打印prompt
    #print(f"\nRound {current_round} - {agent_name} Prompt:")
    #print(f"System Prompt: {system_prompt}")
    #print(f"Human Message: {human_message}")
    #print("-" * 40)

def increment_round(state: DebateState) -> Dict[str, Any]:
    """控制辩论轮次递增的函数"""
    return {"round": state.round + 1}

def should_continue(state: DebateState) -> Literal["debate_round", "__end__"]:
    """判断是否继续辩论的条件函数"""
    if state.round < 3: #辩论次数
        return "debate_round"
    else:
        return "__end__"

def create_debate_graph(model_manager_agent1: ModelManager, model_manager_agent2: ModelManager):
    """创建并返回辩论图"""
    builder = StateGraph(DebateState)
    
    # 使用lambda函数传递额外的参数
    builder.add_node("Agent1", lambda state: agent_node(state, "Agent1", model_manager_agent1, model_manager_agent2))
    builder.add_node("Agent2", lambda state: agent_node(state, "Agent2", model_manager_agent1, model_manager_agent2))
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

    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)

def run_debate(question: str, model_name_agent1: str = "volc_ark", model_name_agent2: str = "qwen"):
    """运行辩论流程的主函数，可以为两个智能体指定不同的模型。默认agent1使用volc_ark，agent2使用qwen。"""
    # 为每个agent建立独立的模型管理器并设置不同模型
    model_manager_agent1 = ModelManager(model_name_agent1)
    model_manager_agent2 = ModelManager(model_name_agent2)
    model_manager_agent1.set_model(model_name_agent1)
    model_manager_agent2.set_model(model_name_agent2)
    
    print(f"Agent1 using: {model_name_agent1}")
    print(f"Agent2 using: {model_name_agent2}")

    graph = create_debate_graph(model_manager_agent1, model_manager_agent2)
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