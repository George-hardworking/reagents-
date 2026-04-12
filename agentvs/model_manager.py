###################################################################################
# File: model_manager.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to manage different LLM models and provide a unified interface for model calls.
####################################################################################
from typing import Callable, Dict, Any
import os
import traceback
from volcenginesdkarkruntime import Ark
from langchain_openai import ChatOpenAI
from core.config import (
    ARK_API_KEY, VOLC_ENDPOINTID,
    QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
)

class ModelManager:
    def __init__(self, model_name: str = "volc_ark", verbose: bool = True, 
                 temperature: float = 0.0,        # 控制输出随机性 (0.0-2.0)，越高越随机
                 top_p: float = 1.0,              # 核采样 (0.0-1.0)，控制候选词汇范围
                 frequency_penalty: float = 0.0,  # 频率惩罚 (-2.0-2.0)，惩罚高频词，减少重复
                 presence_penalty: float = 0.0,   # 存在惩罚 (-2.0-2.0)，惩罚已出现的词，鼓励新话题
                 repetition_penalty: float = 1.0  # 重复惩罚 (0.1-2.0)，防止内容重复 >1.0减少重复，<1.0增加重复
                 ):
        self.available_models = {
            "volc_ark": self._call_volc_ark, #火山
            "qwen": self._call_qwen #千问
        }
        self.current_model = model_name
        self.verbose = verbose
        self.temperature = temperature            
        self.top_p = top_p                         
        self.frequency_penalty = frequency_penalty 
        self.presence_penalty = presence_penalty  
        self.repetition_penalty = repetition_penalty
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0
        self._validate_config()  
            
    def _validate_config(self):
        """验证配置是否正确"""
        if self.current_model == "qwen":
            if not QWEN_API_KEY or QWEN_API_KEY == "your_default_qwen_api_key":
                print("警告: QWEN_API_KEY 未正确设置")
            if not QWEN_BASE_URL:
                print("警告: QWEN_BASE_URL 未设置")
            if not QWEN_MODEL:
                print("错误: QWEN_MODEL 未设置")
                raise ValueError("QWEN_MODEL 未设置")
        elif self.current_model == "volc_ark":
            if not ARK_API_KEY or ARK_API_KEY == "your_default_ark_api_key":
                print("警告: ARK_API_KEY 未正确设置")
            if not VOLC_ENDPOINTID:
                print("警告: VOLC_ENDPOINTID 未设置")
    
    def get_real_model_name(self) -> str:
        """获取模型名称"""
        if self.current_model == "qwen":
            return QWEN_MODEL or "qwen-unknown"
        elif self.current_model == "volc_ark":
            return "Doubao-Seed-1.6-flash"
        else:
            return self.current_model
        
    def _call_volc_ark(self, system_prompt: str, human_message: str) -> str:
        """火山引擎ARK模型调用"""
        if self.verbose:
            print(f"调用火山引擎ARK模型，使用模型: Doubao-Seed-1.6-flash")
            print(f"API基础URL: https://ark.cn-beijing.volces.com/api/v3")
            print(f"当前 temperature: {self.temperature}")
        try:
            ark_client = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=ARK_API_KEY
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_message}
            ]
            # 火山引擎API参数
            params = {
                "model": VOLC_ENDPOINTID,
                "messages": messages,
                "temperature": self.temperature, 
                "top_p": self.top_p               
            }
            # 只在非默认值时添加惩罚参数
            if self.frequency_penalty != 0.0:
                params["frequency_penalty"] = self.frequency_penalty 
            if self.presence_penalty != 0.0:
                params["presence_penalty"] = self.presence_penalty  
            if self.repetition_penalty != 1.0:
                params["repetition_penalty"] = self.repetition_penalty
            
            completion = ark_client.chat.completions.create(**params)
            if hasattr(completion, 'usage'): #统计token
                self.total_input_tokens += completion.usage.prompt_tokens
                self.total_output_tokens += completion.usage.completion_tokens
            self.api_calls += 1
            return completion.choices[0].message.content
        except Exception as e:
            print(f"调用火山引擎ARK模型时出错: {e}")
            return f"模型调用错误: {str(e)}"
    
    def _call_qwen(self, system_prompt: str, human_message: str) -> str:
        """Qwen模型调用"""
        if self.verbose:
            print(f"调用Qwen模型，使用模型: {QWEN_MODEL}")
            print(f"API基础URL: {QWEN_BASE_URL}")
            print(f"当前 temperature: {self.temperature}")
        try:
            if not QWEN_MODEL:
                raise ValueError("QWEN_MODEL 未设置或为空")
            
            qwen_params = {
                "api_key": QWEN_API_KEY,
                "base_url": QWEN_BASE_URL,
                "model": QWEN_MODEL,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            # 只在非默认值时添加惩罚参数
            if self.frequency_penalty != 0.0:
                qwen_params["frequency_penalty"] = self.frequency_penalty 
            if self.presence_penalty != 0.0:
                qwen_params["presence_penalty"] = self.presence_penalty
            
            llm = ChatOpenAI(**qwen_params)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_message}
            ]
            response = llm.invoke(messages)
            # 统计token（如果response有token信息）
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                self.total_input_tokens += token_usage.get('prompt_tokens', 0)
                self.total_output_tokens += token_usage.get('completion_tokens', 0)
            elif hasattr(response, 'usage_metadata'):
                # LangChain可能有不同的token统计方式
                usage = response.usage_metadata
                self.total_input_tokens += usage.get('input_tokens', 0)
                self.total_output_tokens += usage.get('output_tokens', 0)
            self.api_calls += 1
            return response.content
        except Exception as e:
            print(f"调用Qwen模型时出错: {e}")
            traceback.print_exc()
            return f"模型调用错误: {str(e)}"
    
    def get_token_stats(self) -> Dict[str, int]:
            """获取token统计信息"""
            return {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "api_calls": self.api_calls
            }
        
    def reset_token_stats(self):
        """重置token统计信息"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0

    def set_model(self, model_name: str):
        """设置当前使用的模型"""
        if model_name not in self.available_models:
            raise ValueError(f"模型 '{model_name}' 不可用。可用模型: {list(self.available_models.keys())}")
        self.current_model = model_name
        if self.verbose:
            print(f"已切换到模型: {model_name}")
    
    def call_model(self, system_prompt: str, human_message: str) -> str:
        """调用当前设置的模型"""
        return self.available_models[self.current_model](system_prompt, human_message)