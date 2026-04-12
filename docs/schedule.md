
##  Schedule


> This file would present to do list and experimental plans.


### To Do List

- [x] 1. Add token calculation function.
- [x] 2. Add a single-agent model with self-reflection function.
- [x] 3. Adjust the size of a single agent to be approximately that of a multi-agent.
- [x] 4. Add multi-round reflection function to a single agent.

---

### Experimental Plan

- [x] Experiment 0: Replicate the paper
- Test the following **multi-agent model** using the **math** dataset:
- [x] Experiment 1: Test the multi-agent model **(qwen7b_qwen7b)**
- [x] Experiment 2: Test the multi-agent model **(qwen7b_Dou)**
- [x] Experiment 3: Test the multi-agent model **(dou_Dou)**
- Test the following **single-agent model** using the **math** dataset:
- [x] Experiment 4: Test the single-agent model **(qwen7b)**
- [x] Experiment 5: Test the single-agent model **(qwen14b)**
- [x] Experiment 6: Test the single-agent model **(Dou)**
- Test the following **single-agent model with self-reflection function** using the **math** dataset:
- [x] Experiment 7: Test the single-agent model with self-reflection function **(qwen7b)**
- [x] Experiment 8: Test the single-agent model with self-reflection function **(qwen14b)**
- [x] Experiment 9: Test the single-agent model with self-reflection function **(Dou)**
---

### Details of Experimental Plan

<details>
<summary><strong> Experiment 0: Replicate the paper</strong></summary>  

> **Target & Details:**  
We aim to establish a multi-agent debate model by replicating the paper "Improving Factuality and Reasoning in Language Models through Multiagent Debate", and at the same time verify this paper as well as our research idea.

> **Main parameter setting:**  
Agent1: qwen2.5-7b-instruct (temp=0.7)  
Agent2: Doubao-Seed-1.6-flash (temp=0.7)

> **Status:**  
Done by Kaibiao ZHU and Jiachun LIN in Sept 22, 2025

</details>

<details>
<summary><strong> Experiment 1: Test the multi-agent model(qwen7b_qwen7b)</strong></summary>  

> **Target & Details:** 
In Experiment 3, we composed a multi-agent model using a high-performance large language model Doubao-Seed-1.6-flash and a low-performance one qwen2.5-7b-instruct. To further compare the advantages and disadvantages between a single agent and a multi-agent system, and to explore the improvement of the overall model's capability brought by the multi-agent debate style, we designed this experiment. In this experiment, we will have both agents use the large language model qwen2.5-7b-instruct to conduct a statistical analysis on the resource consumption (including time and tokens) and accuracy of the single-agent model on the math dataset.

> **Main parameter setting:**  
Agent1: qwen2.5-7b-instruct (temp=0)  
Agent2: qwen2.5-7b-instruct (temp=0)  

> **Status:**  
Done by Kaibiao ZHU in Oct 31, 2025
</details>

<details>
<summary><strong> Experiment 2: Test the multi-agent model(qwen7b_Dou)</strong></summary>  

> **Target & Details:** 
In order to compare which is better between a single agent and a multi-agent system, we need to conduct this experiment to statistically analyze the resource consumption (including time and tokens) and accuracy rate of the multi-agent model when tested on the math dataset.

> **Main parameter setting:**  
Agent1: qwen2.5-7b-instruct (temp=0)  
Agent2: Doubao-Seed-1.6-flash (temp=0)

> **Status:**  
Done by Kaibiao ZHU in Oct 18, 2025

</details>

<details>
<summary><strong> Experiment 3: Test the multi-agent model(Dou_Dou)</strong></summary>  

> **Target & Details:** 
In order to compare which is better between a single agent and a multi-agent system, we need to conduct this experiment to statistically analyze the resource consumption (including time and tokens) and accuracy rate of the multi-agent model when tested on the math dataset.

> **Main parameter setting:**  
Agent1: Doubao-Seed-1.6-flash (temp=0)
Agent2: Doubao-Seed-1.6-flash (temp=0)

> **Status:**  
Done by Kaibiao ZHU in Oct 31, 2025

</details>

<details>
<summary><strong> Experiment 4: Test the single-agent model(qwen7b)</strong></summary>  

> **Target & Details:**  
In order to compare which is better between a single agent and a multi-agent system, we need to conduct this experiment to statistically analyze the resource consumption (including time and tokens) and accuracy rate of the single-agent model when tested on the math dataset.

> **Main parameter setting:**  
Agent1: qwen2.5-7b-instruct (temp=0)  

> **Status:**  
Done by Kaibiao ZHU in Oct 16, 2025

</details>

<details>
<summary><strong> Experiment 5: Test the single-agent model(qwen14b)</strong></summary>  

> **Target & Details:**   
To ensure a fairer comparison between the single-agent model and the multi-agent model, we decided to increase the size of the single-agent model to be as close as possible to that of the multi-agent model. By doing so, we can obtain a more equitable comparison of model function.

> **Main parameter setting:**  
Agent1: qwen2.5-14b-instruct (temp=0)  

> **Status:**  
Not started
</details>


<details>
<summary><strong> Experiment 6: Test the single-agent model(Dou)</strong></summary>  

> **Target & Details:**   
> In Experiment 2, we employed the less performant LLM qwen2.5-7b-instruct. To further compare the advantages and disadvantages between a single agent and a multi-agent system, we set up this experiment. In this experiment, we will use the more performant LLM Doubao-Seed-1.6-flash to conduct statistical analysis on the resource consumption (including time and tokens) and accuracy of the single-agent model on the mathematical dataset.

> **Main parameter setting:**  
Agent1: Doubao-Seed-1.6-flash (temp=0)  

> **Status:**  
Not started
</details>

<details>
<summary><strong> Experiment 7: Test the single-agent model with self-reflection function(qwen7b)</strong></summary>  

> **Target & Details:**   
To ensure a fairer comparison between the single-agent model and the multi-agent model, we have decided to adjust the style of the single-agent model. Specifically, we will have it engage in one round of response followed by two rounds of self-reflection. By doing so, the style of the single-agent model will be more similar to the three-round debate style of the multi-agent model, thereby obtaining a more equitable comparison of their function.

> **Main parameter setting:**  
Agent1: qwen2.5-7b-instruct (temp=0)  

> **Status:**  
Done by Kaibiao ZHU in Nov 12, 2025
</details>

<details>
<summary><strong> Experiment 8: Test the single-agent model with self-reflection function(qwen14b)</strong></summary>  

> **Target & Details:**   
To ensure a fairer comparison between the single-agent model and the multi-agent model, we have decided to adjust the style of the single-agent model. Specifically, we will have it engage in one round of response followed by two rounds of self-reflection. By doing so, the style of the single-agent model will be more similar to the three-round debate style of the multi-agent model, thereby obtaining a more equitable comparison of their function.

> **Main parameter setting:**  
Agent1: qwen2.5-14b-instruct (temp=0)  

> **Status:**  
Done by Kaibiao ZHU in 2025
</details>

<details>
<summary><strong> Experiment 9: Test the single-agent model with self-reflection function(Dou)</strong></summary>  

> **Target & Details:**   
To ensure a fairer comparison between the single-agent model and the multi-agent model, we have decided to adjust the style of the single-agent model. Specifically, we will have it engage in one round of response followed by two rounds of self-reflection. By doing so, the style of the single-agent model will be more similar to the three-round debate style of the multi-agent model, thereby obtaining a more equitable comparison of their function.

> **Main parameter setting:**  
Agent1: Doubao-Seed-1.6-flash (temp=0)  

> **Status:**  
Done by Kaibiao ZHU in 2025
</details>





