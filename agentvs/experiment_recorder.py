###################################################################################
# File: experiment_recorder.py
# File Manager: Kaibiao ZHU
# Description: This file is to record and manage the results of experiments, including saving and loading experiment data.
####################################################################################
import json
import os
from typing import Dict, List, Any
from datetime import datetime
import uuid


# Support that we implemented the experiment on one sample.
sample_result = {
    "sample_id": {
        # Overview present the overall information of the experiment, such as
        # the sample id,
        "overview": {},
        "details": {
            "round_1": {},
            "round_2": {},
            "round_3": {},
            "round_4": {},
        },
    },
}


class ExperimentRecorder:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def save_single_experiment(
        self, experiment_data: Dict[str, Any], filename: str = None
    ) -> str:
        """保存单个实验结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{timestamp}_{uuid.uuid4().hex[:8]}.json"

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

        return filepath

    def save_batch_experiments(
        self, experiments_data: List[Dict[str, Any]], summary_filename: str = None
    ) -> str:
        """保存批量实验结果和汇总"""
        if summary_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"experiment_summary_{timestamp}.json"

        filepath = os.path.join(self.results_dir, summary_filename)

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(experiments_data),
            "experiments": experiments_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        return filepath

    def load_experiment(self, filename: str) -> Dict[str, Any]:
        """加载单个实验结果"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_results(self) -> List[str]:
        """获取所有结果文件列表"""
        return [f for f in os.listdir(self.results_dir) if f.endswith(".json")]
