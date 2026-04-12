# experiments/test_paper_single.py
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()  # 加载.env

from agentvs.single_agent import solve_once

# paper的10个问题
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
    "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
]

def main():
    # 仅通过 solve_once 的参数指定模型
    model_name = "qwen"
    prompt_type = "cot"

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Prompt Type: {prompt_type}")
    print("=" * 80)

    total = len(questions)
    for i, question in enumerate(questions, 1):
        print(f"\n🚀 Starting Question {i}/{total}")
        
        try:
            # solve_once 内部会创建 ModelManager
            result = solve_once(question, model_name=model_name, prompt_type=prompt_type)
            print(f"Final answer: {getattr(result, 'answer', result)}")
        except Exception as e:
            print(f"❌ Error on Question {i}: {e}")
        print("-" * 80)
        print(f"🏁 Finished Question {i}/{total}")
        print("=" * 80)

if __name__ == "__main__":
    main()
