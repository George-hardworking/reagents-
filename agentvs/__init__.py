###################################################################################
# File: __init__.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to make the agentvs package a module and expose the run_debate function.
####################################################################################
from .debate_core import run_debate
from .multi_agent import (
    MultiAgentDebate,
    run_multi_agent_debate,
    run_multi_agent_debate_for_dataset,
)
from .single_agent import run_single_agent_for_dataset
from evaluation.experiment_recorder import ExperimentRecorder

# from .dataset_tester import DatasetTester


__all__ = [
    "run_debate",
    "MultiAgentDebate",
    "run_multi_agent_debate",
    "run_multi_agent_debate_for_dataset",
    "run_single_agent_for_dataset",
]

__version__ = "0.1.0"
