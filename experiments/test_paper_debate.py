###################################################################################
# File: test_paper_debate.py
# File Manager: Kaibiao ZHU
# Description: This file is in order to complete the experiment of reproducing the paper using the multi-agent debate model.
####################################################################################
from agentvs.debate_core import run_debate

questions = [
    "A treasure hunter found a buried treasure chest filled with gems. There were 175 diamonds, 35 fewer rubies than diamonds, and twice the number of emeralds than the rubies. How many of the gems were there in the chest?",
    "What is the result of 10+20*23+3-11*18?",
    "What is the result of 3+7*9+19-21*18?",
    "What is the result of 4+23*6+24-24*12?",
    "What is the result of 8+14*15+20-3*26?",
    "Give a bullet point biography of David S. Johnson highlighting their contributions and achievements as a computer scientist, with each fact separated with a new line character.",
    "Six positive integers are written on the faces of a cube. Each vertex is labeled with the product of the three numbers on the faces adjacent to the vertex. If the sum of the numbers on the vertices is equal to 1001, then what is the sum of the numbers written on the faces? A) 18. B) 13. C) 1001. D) 31.",
    "You suspect that your patient has an enlarged submandibular salivary gland. You expect the enlarged gland: A) to be palpable intraorally. B) to be palpable extraorally. C) to be palpable both intra- and extraorally. D) only to be detectable by radiographical examination.",
    "What country was the computer scientist Tomas Lozano-Perez born in?",
    "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?"
]

print("Starting Multi-agents Debate with Different APIs...")

for i, question in enumerate(questions, 1):
    print(f"\n🚀 Starting Question {i}/{len(questions)}")

    #run_debate(question, "volc_ark", "qwen") #agent1, agent2
    run_debate(question, "qwen", "volc_ark") #agent1, agent2
    #run_debate(question, "qwen", "qwen") #agent1, agent2
    #run_debate(question, "volc_ark", "volc_ark") #agent1, agent2

    print(f"\n🏁 Finished Question {i}")
    print("\n" + "="*80 + "\n")