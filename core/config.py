###################################################################################
# File: config.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to store configuration variables for different LLM.
####################################################################################
import os
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

# 火山引擎ARK配置
ARK_API_KEY = os.getenv("ARK_API_KEY")
VOLC_ENDPOINTID = os.getenv("VOLC_ENDPOINTID")

# Qwen配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = os.getenv("QWEN_MODEL")

# OpenAI配置
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")