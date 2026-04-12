###################################################################################
# File: multi_agent.py
# File Manager: Kaibiao ZHU
# Description: This file implements a multi-agent debate system using LangGraph, allowing two AI agents to debate and arrive at a final answer for a given question.
####################################################################################
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from typing import Dict, List, Literal, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from .model_manager import ModelManager
from .math_normalize import normalize_answer
from .grader import grade_answer
from .eval_math import parse_answer
import re
import uuid
import time 
from .prompts import get_prompt
from datetime import datetime
from .experiment_recorder import ExperimentRecorder
from .utils.record import save_record

@dataclass
class DebateState:
    question: str
    agent1_answer: str = ""
    agent2_answer: str = ""
    round: int = 0
    history: List[Dict] = field(default_factory=list)

    agent1_system_prompt: str = ""
    agent1_human_message: str = ""
    agent2_system_prompt: str = ""
    agent2_human_message: str = ""

    agent1_runtime: float = 0.0
    agent2_runtime: float = 0.0

@dataclass
class DebateResult:
    question: str
    agent1_raw: str
    agent2_raw: str
    agent1_final_extracted: str
    agent2_final_extracted: str
    agent1_normalized: Optional[str]
    agent2_normalized: Optional[str]
    agent1_correct: bool = False
    agent2_correct: bool = False
    gold_answer: Optional[str] = None
    rounds: int = 0
    model1: str = ""
    model2: str = ""
    experiment_data: Dict[str, Any] = field(default_factory=dict)

class MultiAgentDebate:
    def __init__(self, model_name_agent1: str = "volc_ark", model_name_agent2: str = "qwen", 
                 max_rounds: int = 3, use_unified_prompts: bool = False):
        self.model_manager_agent1 = ModelManager(model_name_agent1, verbose=False)
        self.model_manager_agent2 = ModelManager(model_name_agent2, verbose=False)
        self.max_rounds = max_rounds
        self.system_prompt_initial = get_prompt("", "multi_agent_initial")
        self.system_prompt_debate = get_prompt("", "multi_agent_debate")
        self.graph = self._create_debate_graph()

    def _agent_node_factory(self, agent_name: str):
        """创建智能体节点函数的工厂方法"""
        def agent_node(state: DebateState) -> Dict[str, Any]:
            current_round = state.round

            if current_round == 0:
                system_prompt = self.system_prompt_initial.replace("<Problem>", state.question)
                human_message = state.question
            else:
                if agent_name == "Agent1":
                    other_answer = state.agent2_answer
                else:
                    other_answer = state.agent1_answer
                
                system_prompt = self.system_prompt_debate.replace(
                    "<other agent responses>", 
                    other_answer
                ).replace(
                    "<Problem>", 
                    state.question
                )
                human_message = ""
            start_time = time.time()
            
            if agent_name == "Agent1":
                answer = self.model_manager_agent1.call_model(system_prompt, human_message)
            else:
                answer = self.model_manager_agent2.call_model(system_prompt, human_message)
            
            runtime = time.time() - start_time

            if agent_name == "Agent1":
                return {
                    "agent1_answer": answer,
                    "agent1_system_prompt": system_prompt,
                    "agent1_human_message": human_message,
                    "agent1_runtime": runtime
                }
            else:
                return {
                    "agent2_answer": answer,
                    "agent2_system_prompt": system_prompt,
                    "agent2_human_message": human_message,
                    "agent2_runtime": runtime
                }
        return agent_node
    
    def _increment_round(self, state: DebateState) -> Dict[str, Any]:
        """轮次递增函数"""
        return {"round": state.round + 1}
    
    def _should_continue(self, state: DebateState) -> Literal["debate_round", "__end__"]:
        """判断是否继续辩论的条件函数"""
        if state.round < self.max_rounds:
            return "debate_round"
        else:
            return "__end__"
    
    def _create_debate_graph(self):
        """创建辩论图"""
        builder = StateGraph(DebateState)
        
        builder.add_node("Agent1", self._agent_node_factory("Agent1"))
        builder.add_node("Agent2", self._agent_node_factory("Agent2"))
        builder.add_node("increment_round", self._increment_round)
        builder.set_entry_point("Agent1")
        builder.add_edge("Agent1", "Agent2")
        builder.add_edge("Agent2", "increment_round")
        builder.add_conditional_edges(
            "increment_round",
            self._should_continue,
            {
                "debate_round": "Agent1",
                "__end__": END
            }
        )

        memory = InMemorySaver()
        return builder.compile(checkpointer=memory)
    
    def extract_final_answer(self, text: str) -> str:
        """从模型输出中提取最终responsee- 使用你原来的提取逻辑"""
        if not text:
            return ""
        
        # \boxed{}提取逻辑
        boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
        boxed_match = re.search(boxed_pattern, text) 
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # 如果没有\boxed{}，使用备用提取逻辑
        math_patterns = [
            r'\\frac\{(-?\d+)\}\{(-?\d+)\}',  # \frac{}{}
            r'\\frac\s*(-?\d+)\s*(-?\d+)',    # \frac num den
            r'-?\d+/-?\d+',                   # 简单分数
            r'-?[0-9.]+',                     # 数字
            r'[A-Ea-e]',                      # 选择题答案
            r'\\sqrt\{[^}]+\}',               # 平方根
            r'\\pi',                          # π
            r'-?\d+(?:,\s*-?\d+)+',           # 数字序列
        ]
        
        # 从文本末尾开始反向查找
        for pattern in math_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if pattern in [r'\\frac\{(-?\d+)\}\{(-?\d+)\}', r'\\frac\s*(-?\d+)\s*(-?\d+)']:
                    if len(matches[0]) == 2:
                        return f"{matches[0][0]}/{matches[0][1]}"
                return matches[-1] 
        
        return text  

    def _standardize_and_evaluate(self, model_output: str, gold_answer: Optional[str] = None) -> tuple[str, Optional[str], bool]:
        """标准化流程解析和评估答案"""
        #使用 eval_math 的解析逻辑
        parsed_answer = parse_answer(model_output)
        normalized_answer = normalize_answer(parsed_answer) if parsed_answer else normalize_answer(model_output)
        is_correct = False
        if gold_answer:
            gold_parsed = parse_answer(gold_answer)
            gold_normalized = normalize_answer(gold_parsed) if gold_parsed else normalize_answer(gold_answer)
            is_correct = grade_answer(normalized_answer, gold_normalized)

        return parsed_answer or "", normalized_answer, is_correct

    def run(self, question: str, gold_answer: Optional[str] = None,
            save_result: bool = False, recorder: ExperimentRecorder = None,
            dataset_name: str = "unknown", sample_id: str = None, 
            results_dir: str = "results") -> DebateResult:
        """运行辩论+标准化评估+实验记录"""
        start_time = time.time()
        # 暂时注释掉token计算与统计相关代码
        # self.model_manager_agent1.reset_token_stats()   
        # self.model_manager_agent2.reset_token_stats()

        initial_state = DebateState(question=question, round=0)
        config = {"configurable": {"thread_id": 1}}
        
        print(f"Starting Debate Process...")
        print("="*50)
        
        final_state = None
        last_round = 0
        agent1_answer = ""
        agent2_answer = ""

        round_details = {}

        print(f"Initial Question: {question}")
        print("-" * 30)

        # 用来存储每轮的prompt信息和运行时间
        round_prompts = {}
        round_runtimes = {}
        
        for event in self.graph.stream(initial_state, config, stream_mode="values"):
            current_round = event["round"]
            
            # agent1的回应和prompt
            if "agent1_answer" in event:
                agent1_answer = event["agent1_answer"]
                if current_round not in round_prompts:
                    round_prompts[current_round] = {}
                    round_runtimes[current_round] = {}
                if "agent1_system_prompt" in event:
                    round_prompts[current_round]["agent1_system_prompt"] = event["agent1_system_prompt"]
                if "agent1_human_message" in event:
                    round_prompts[current_round]["agent1_human_message"] = event["agent1_human_message"]
                if "agent1_runtime" in event:
                    round_runtimes[current_round]["agent1_runtime"] = event["agent1_runtime"]
            
            # agent2的回应和prompt
            if "agent2_answer" in event:
                agent2_answer = event["agent2_answer"]
                if current_round not in round_prompts:
                    round_prompts[current_round] = {}
                    round_runtimes[current_round] = {}
                if "agent2_system_prompt" in event:
                    round_prompts[current_round]["agent2_system_prompt"] = event["agent2_system_prompt"]
                if "agent2_human_message" in event:
                    round_prompts[current_round]["agent2_human_message"] = event["agent2_human_message"]
                if "agent2_runtime" in event:
                    round_runtimes[current_round]["agent2_runtime"] = event["agent2_runtime"]
            
            if current_round > last_round and agent1_answer and agent2_answer:
                print(f"Round {current_round} Results:")
                print(f"🤖Agent1: {agent1_answer}") 
                print(f"🤖Agent2: {agent2_answer}")
                print("-" * 30)

                round_key = f"round_{current_round}"
                # 获取当前轮次的prompt信息和运行时间
                current_prompts = round_prompts.get(current_round, {})
                current_runtimes = round_runtimes.get(current_round, {})
                
                # 计算运行时间统计
                agent1_runtime = current_runtimes.get("agent1_runtime", 0.0)
                agent2_runtime = current_runtimes.get("agent2_runtime", 0.0)
                round_total_runtime = agent1_runtime + agent2_runtime
                
                round_details[round_key] = {
                    "agent1_config": {
                        "model": self.model_manager_agent1.get_real_model_name(),
                        "temperature": getattr(self.model_manager_agent1, "temperature", 0.7),
                        "top_p": getattr(self.model_manager_agent1, "top_p", 1.0),
                        "frequency_penalty": getattr(self.model_manager_agent1, "frequency_penalty", 0.0),
                        "presence_penalty": getattr(self.model_manager_agent1, "presence_penalty", 0.0),
                        "repetition_penalty": getattr(self.model_manager_agent1, "repetition_penalty", 1.0)
                    },
                    "agent2_config": {
                        "model": self.model_manager_agent2.get_real_model_name(),
                        "temperature": getattr(self.model_manager_agent2, "temperature", 0.7),
                        "top_p": getattr(self.model_manager_agent2, "top_p", 1.0),
                        "frequency_penalty": getattr(self.model_manager_agent2, "frequency_penalty", 0.0),
                        "presence_penalty": getattr(self.model_manager_agent2, "presence_penalty", 0.0),
                        "repetition_penalty": getattr(self.model_manager_agent2, "repetition_penalty", 1.0)
                    },
                    "agent1_content": {
                        "prompt": {
                            "system_prompt": current_prompts.get("agent1_system_prompt", ""),
                            "human_message": current_prompts.get("agent1_human_message", "")
                        },
                        "response": agent1_answer,
                        "runtime": round(agent1_runtime, 4)
                    },
                    "agent2_content": {
                        "prompt": {
                            "system_prompt": current_prompts.get("agent2_system_prompt", ""),
                            "human_message": current_prompts.get("agent2_human_message", "")
                        },
                        "response": agent2_answer,
                        "runtime": round(agent2_runtime, 4)
                    },
                    "round_total_runtime": round(round_total_runtime, 4)
                }

                if final_state is not None and hasattr(final_state, 'history'):
                    final_state.history.append({
                        "round": current_round,
                        "agent1_answer": agent1_answer,
                        "agent2_answer": agent2_answer
                    })

                last_round = current_round
                agent1_answer = ""
                agent2_answer = ""
            
            final_state = event

        if agent1_answer and agent2_answer:
            print(f"Round {last_round+1} Results:")
            print(f"🤖Agent1: {agent1_answer}")
            print(f"🤖Agent2: {agent2_answer}")
            print("-" * 30)

        print(f"Final Answer - 🤖Agent1: {final_state['agent1_answer']}")
        print(f"Final Answer - 🤖Agent2: {final_state['agent2_answer']}")
        print("Debate ended!")
        print("="*50)
        
        agent1_extracted, agent1_normalized, agent1_correct = self._standardize_and_evaluate(
            final_state['agent1_answer'], gold_answer
        )
        agent2_extracted, agent2_normalized, agent2_correct = self._standardize_and_evaluate(
            final_state['agent2_answer'], gold_answer
        )
        
        end_time = time.time()
        runtime = end_time - start_time

        sample_id = sample_id or f"sample_{str(uuid.uuid4())[:8]}"
        
        # 暂时注释掉token计算与统计相关代码
        # final_token_stats = {
        #     "agent1": self.model_manager_agent1.get_token_stats(),
        #     "agent2": self.model_manager_agent2.get_token_stats()
        # }
        
        # if round_details:
        #     last_round_key = f"round_{final_state['round']}"
        #     if last_round_key in round_details:
        #         round_details[last_round_key]["input_tokens"] = {
        #             "agent1": final_token_stats["agent1"]["input_tokens"],
        #             "agent2": final_token_stats["agent2"]["input_tokens"]
        #         }
        #         round_details[last_round_key]["output_tokens"] = {
        #             "agent1": final_token_stats["agent1"]["output_tokens"],
        #             "agent2": final_token_stats["agent2"]["output_tokens"]
        #         }
        #         round_details[last_round_key]["total_tokens"] = {
        #             "agent1": final_token_stats["agent1"]["total_tokens"],
        #             "agent2": final_token_stats["agent2"]["total_tokens"]
        #         }
        #         round_details[last_round_key]["api_calls"] = {
        #             "agent1": final_token_stats["agent1"]["api_calls"],
        #             "agent2": final_token_stats["agent2"]["api_calls"]
        #         }

        sample_data = {
            sample_id: {
                "overview": {
                    "dataset": dataset_name,
                    "question": question,
                    "gold_answer": gold_answer,
                    "total_runtime": runtime,
                    "total_rounds": final_state['round'],
                    "timestamp": datetime.now().isoformat(),
                    # 暂时注释掉correct部分的代码
                    # "correct": {
                    #     "agent1": agent1_correct,
                    #     "agent2": agent2_correct,
                    #     "model_correct": agent1_correct and agent2_correct
                    # }
                },
                "details": round_details
            }
        }

        if save_result:
            save_record(
                data=sample_data[sample_id],
                folder=results_dir,
                savename=sample_id,
                file_format="json",
                as_block=True,
                block_size=1000
            )

        experiment_data = {
            "id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "agent1_model": self.model_manager_agent1.get_real_model_name(),
            "agent2_model": self.model_manager_agent2.get_real_model_name(),
            "runtime_seconds": runtime,
            "total_rounds": final_state['round'],
            "question": question,
            "gold_answer": gold_answer
        }

        return DebateResult(
            question=question,
            agent1_raw=final_state['agent1_answer'],
            agent2_raw=final_state['agent2_answer'],
            agent1_final_extracted=agent1_extracted,
            agent2_final_extracted=agent2_extracted,
            agent1_normalized=agent1_normalized,
            agent2_normalized=agent2_normalized,
            agent1_correct=agent1_correct, 
            agent2_correct=agent2_correct, 
            gold_answer=gold_answer,
            rounds=final_state['round'],
            model1=self.model_manager_agent1.get_real_model_name(),
            model2=self.model_manager_agent2.get_real_model_name(),
            experiment_data=experiment_data
        )

def run_multi_agent_debate(question: str, model_name_agent1: str = "volc_ark", 
                          model_name_agent2: str = "qwen", max_rounds: int = 3,
                          save_result: bool = False, dataset_name: str = "unknown",
                          sample_id: str = None) -> DebateResult:
    """简化版的运行函数"""
    debate = MultiAgentDebate(model_name_agent1, model_name_agent2, max_rounds)
    return debate.run(question, save_result=save_result, dataset_name=dataset_name, sample_id=sample_id)

def run_multi_agent_debate_for_dataset(question: str, gold_answer: str, 
                                     model_name_agent1: str = "volc_ark", 
                                     model_name_agent2: str = "qwen", 
                                     max_rounds: int = 3,
                                     save_result: bool = False,
                                     dataset_name: str = "unknown",
                                     sample_id: str = None,
                                     results_dir: str = "results") -> DebateResult:
    """用于数据集评估的运行函数"""
    debate = MultiAgentDebate(model_name_agent1, model_name_agent2, max_rounds)
    return debate.run(question, gold_answer, save_result=save_result, 
                     dataset_name=dataset_name, sample_id=sample_id, results_dir=results_dir)