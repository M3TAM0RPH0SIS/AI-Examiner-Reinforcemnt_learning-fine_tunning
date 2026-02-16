from unsloth import FastLanguageModel
import torch
import json

class QAgent:
    def __init__(self, model_path="../models/q_agent_grpo"):
        print(f"😈 Loading Q-Agent from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            load_in_4bit = True,
            gpu_memory_utilization = 0.5, # Low mem for battle
        )
        FastLanguageModel.for_inference(self.model)
        self.prompt = """... (Keep your full prompt here) ... ### Response:\n"""

    def generate_raw(self):
        inputs = self.tokenizer([self.prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.85)
        text = self.tokenizer.decode(outputs[0]).split("Response:")[-1].strip()
        if "<|im_end|>" in text: text = text.split("<|im_end|>")[0]
        return text

    def generate_hardest_trap(self, n=3):
        """
        Generates 3 questions and picks the longest one (Heuristic for complexity).
        """
        print(f"⚔️ Generating {n} candidates to find the killer trap...")
        candidates = []
        for _ in range(n):
            try:
                raw = self.generate_raw()
                # Validate JSON
                obj = json.loads(raw)
                candidates.append(obj)
            except:
                continue
        
        if not candidates: return '{"error": "Failed to gen"}'
        
        # Filter: Pick the one with the longest explanation
        best = max(candidates, key=lambda x: len(x.get("explanation", "")))
        return json.dumps(best)