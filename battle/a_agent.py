from unsloth import FastLanguageModel
import torch
import json
import re
from collections import Counter

class AAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"🛡️ Loading A-Agent ({model_name})...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,
            load_in_4bit = True,
            gpu_memory_utilization = 0.5,
        )
        FastLanguageModel.for_inference(self.model)

    def _solve_once(self, prompt):
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        text = self.tokenizer.decode(outputs[0])
        match = re.search(r'"final_answer":\s*"([A-D])"', text)
        return match.group(1) if match else "C"

    def solve_with_voting(self, question_json, votes=3):
        """
        Runs the model 3 times and takes the majority answer.
        """
        prompt = f"""You are an expert. Solve this:
{question_json}
Return strict JSON: {{"thought": "...", "final_answer": "A/B/C/D"}}"""
        
        results = []
        print(f"🤔 Thinking (Voting {votes} times)...")
        for i in range(votes):
            ans = self._solve_once(prompt)
            results.append(ans)
        
        # Majority Vote
        final_answer = Counter(results).most_common(1)[0][0]
        print(f"🗳️ Votes: {results} -> Winner: {final_answer}")
        return final_answer