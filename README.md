### Project Description

> This project was carried out under the guidance of **Prof. Sijia Chen**, the project members included **Kaibiao Zhu** and **Jiajun Lin**. 

> To explore whether multi-agent systems are necessarily superior to single-agent systems, this project innovatively questions the efficiency and effectiveness of multi-agent systems. By comparing multi agents systems with single agent systems under rigorous experimental conditions, it captures the precise differences in accuracy among models of the same size when dealing with the same dataset of varying difficulty, and analyzes the differences they have in resource consumption under the same task. This research will set new standards for the application of agents in both practical and academic fields. Eventually，our goal is to present our research findings at the ICML conference.



### Structure

```text
.
├── README.md                         # Overview and structure of the project.
├── description.txt                   # Brief project description.
├── requirements.txt                  # List of dependencies required to run the project.
├── setup.py                          # Project installation script.
├── LICENSE                           # Project license.
│
├── agentvs/                          # Core agent modules.
│   ├── __init__.py
│   ├── debate_core.py                # Core logic for agent debate.
│   ├── model_manager.py              # Model manager for connecting LLMs.
│   ├── multi_agent.py                # Multi-agent system implementation.
│   ├── single_agent.py               # Single-agent system implementation.
│   └── single_reflection.py          # Single-agent reflection implementation.
│
├── core/                             # Core configuration and common tools.
│   ├── __init__.py
│   ├── config.py                     # Global configuration settings.
│   ├── prompts.py                    # Prompt templates used by agents.
│   └── record.py                     # Logging and record saving tools.
│
├── data/                             # Data processing and loading.
│   ├── __init__.py
│   └── dataset_loader.py             # Dataset loading utilities (GSM8K, MATH, etc.).
│
├── evaluation/                       # Evaluation and normalization modules.
│   ├── __init__.py
│   ├── eval_math.py                  # Math evaluation tools.
│   ├── experiment_recorder.py        # Records experiment meta-data and logs.
│   ├── grader.py                     # Scoring system for evaluating answers.
│   └── math_normalize.py             # Tools to normalize mathematical expressions.
│
├── tool_scripts/                     # Scripts for analysis and project utilities.
│   ├── __init__.py
│   ├── analysis.py                   # General analysis script.
│   ├── analyze_debate_results.py     # Script to analyze debate experiment results.
│   ├── analyze_php_results.py        # Script to analyze PHP (Progressive Hint Prompting) results.
│   ├── analyze_results.py            # Script for basic results analysis.
│   ├── compare_all_experiments.py    # Generates cross-experiment comparison reports.
│   ├── merge_results_chunks.py       # Merges chunked result data.
│   ├── project_structure.py          # Script for generating repository directory trees.
│   ├── statistics.py                 # Statistical analysis tools.
│   └── test_php.py                   # Test utility for PHP scripts.
│
├── experiments/                      # Scripts for running different experiments.
│   ├── dataset_test_debate.py        # Runs multi-agent debate on datasets.
│   ├── dataset_test_debate_parallel.py # Runs debates in parallel.
│   ├── dataset_test_php.py           # Runs PHP tests on datasets.
│   ├── dataset_test_single.py        # Runs single-agent evaluation on datasets.
│   ├── dataset_test_unified_parallel.py # Unified script for running agent benchmarks in parallel.
│   ├── magents_gsm8k.py              # Multi-agent tests specifically for GSM8K.
│   ├── magents_paper.py              # Multi-agent testing reference from research paper.
│   ├── magents_test_if_real_debate.py # Verifies the authenticity and quality of agent debates.
│   ├── test_paper_debate.py          # Debug/test script for debate configurations.
│   ├── test_paper_single.py          # Debug/test script for single-agent configurations.
│   └── test_php_quick.py             # Quick execution script for PHP experiments.
│
├── scripts/                          # Bash scripts for job automation.
│   ├── manage_parallel_experiments.sh
│   ├── run_multi_agent_parallel.sh
│   ├── run_php_parallel.sh
│   ├── run_php_tests.sh
│   └── run_single_agent_parallel.sh
│
├── configs/                          # Configuration files.
│
├── docs/                             # Documentation on the project style and status.
│   └── schedule.md                   # Timeline and schedule planning.
│
└── results/                          # Directory for saving all experiment results and logs.
    ├── comparison_analysis/          # Output folder for merged analyses.
    └── [experiment_folders]/         # Dynamically generated results folders per test.

```

