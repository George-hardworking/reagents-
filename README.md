### Project Description

> This project was carried out under the guidance of **Prof. Sijia Chen**, the project members included **Kaibiao Zhu** and **Jiajun Lin**. 

> To explore whether multi-agent systems are necessarily superior to single-agent systems, this project innovatively questions the efficiency and effectiveness of multi-agent systems. By comparing multi agents systems with single agent systems under rigorous experimental conditions, it captures the precise differences in accuracy among models of the same size when dealing with the same dataset of varying difficulty, and analyzes the differences they have in resource consumption under the same task. This research will set new standards for the application of agents in both practical and academic fields. Eventually，our goal is to present our research findings at the ICML conference.



### Structure

```text
.
├── README.md               # Overview and structure of the project.
├── docs/
│   ├── md-style.md         # 
│   ├── Progress-record.md  # Record of progress and findings.
│   ├── reference.md        # Record of all papers used in the project.
├── configs/                # Configuration files for the project.
├── agentvs/
│   ├── utils
│       ├──record.py
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── debate_core.py
│   ├── eval_math.py
│   ├── experiment_recorder.py
│   ├── grader.py
│   ├── math_normalize.py
│   ├── model_manager.py
│   ├── multi_agent.py
│   ├── prompts.py
│   ├── single_agent.py
├── experiments/
│   ├── dataset_test_parallel
│       ├── 
│   ├── dataset_test_debate.py
│   ├── dataset_test_single.py
│   ├── dataset_test_unified.py
│   ├── magents_gsm8k.py
│   ├── magents_paper.py
│   ├── magents_test_if_real_debate.py
│   ├── test_paper_debate.py
│   ├── test_paper_single.py
├── results/
├── description.txt         # 
└── requirements.txt        # List of dependencies required to run the project.
├── setup.py
.

```

