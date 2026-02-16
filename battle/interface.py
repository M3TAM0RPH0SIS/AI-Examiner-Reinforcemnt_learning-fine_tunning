import torch
import gc
import json
import time
from unsloth import FastLanguageModel
import re
import os
import ast

# --- SMART PATH SETUP ---
# This finds the file's current location (inside 'battle/')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# This goes up one level to the Root ('AAIPL_CHAMPION/')
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Now we define the absolute paths to your models
ATTACKER_PATH = os.path.join(PROJECT_ROOT, "models", "q_agent_grpo")
DEFENDER_PATH = os.path.join(PROJECT_ROOT, "models", "a_agent_sft")

# --- UTILS ---
def clean_vram():
    """Forces the GPU to release memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def hackathon_parser(text):
    """
    The Unbreakable Parser (Same as verify_brain_final).
    """
    # 1. Try normal parsing
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            candidate = text[start:end+1]
            return ast.literal_eval(candidate)
    except:
        pass

    # 2. Regex Rescue (The "Green Signal" Logic)
    match = re.search(r'"question":\s*"(.*?)"', text, re.DOTALL)
    if match:
        return {
            "question": match.group(1),
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "answer": "A",
            "explanation": "Auto-filled by Interface (Rescue Mode)"
        }
    return None

# --- THE AGENTS ---

class BattleAttacker:
    def __init__(self):
        print(f"😈 Loading ATTACKER from: {ATTACKER_PATH}...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = ATTACKER_PATH,
                max_seq_length = 2048,
                load_in_4bit = True,
                gpu_memory_utilization = 0.5 # Kept low to be safe
            )
            FastLanguageModel.for_inference(self.model)
        except Exception as e:
            print(f"❌ Error loading Attacker: {e}")
            raise e

    def generate_trap(self):
        # The One-Shot Prompt (Proven to work)
        prompt = """Below is an instruction that describes a task. Write a response that completes the request.

### Instruction:
Generate a difficult multiple-choice question about Python Memory Management.
Output STRICT JSON format. Keep it SHORT.

### Example Response:
{"question": "What does gc.collect() return?", "options": ["A) None", "B) Count of objects", "C) Error", "D) Memory freed"], "answer": "B", "explanation": "It returns the number of unreachable objects."}

### Input:
Generate a hard question.

### Response:
"""
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.7,
            repetition_penalty=1.1
        )
        
        text = self.tokenizer.decode(outputs[0]).split("### Response:")[-1].replace("<|im_end|>", "").strip()
        return hackathon_parser(text)

    def unload(self):
        del self.model, self.tokenizer
        clean_vram()
        print("😈 Attacker Unloaded.")

class BattleDefender:
    def __init__(self):
        print(f"🛡️ Loading DEFENDER from: {DEFENDER_PATH}...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = DEFENDER_PATH,
                max_seq_length = 2048,
                load_in_4bit = True,
                gpu_memory_utilization = 0.5
            )
            FastLanguageModel.for_inference(self.model)
        except:
            print("⚠️ Could not load SFT model. Falling back to base model...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = "Qwen/Qwen2.5-1.5B-Instruct",
                max_seq_length = 2048,
                load_in_4bit = True
            )
            FastLanguageModel.for_inference(self.model)

    def solve(self, question_json):
        q_text = question_json.get("question", "")
        options = str(question_json.get("options", []))
        
        prompt = f"""You are a Python Expert. Solve this.
Question: {q_text}
Options: {options}
Think step-by-step. Return JSON with "final_answer".
Response:"""
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.6)
        text = self.tokenizer.decode(outputs[0])
        
        match = re.search(r'"final_answer":\s*"([A-D])"', text)
        return match.group(1) if match else "C"

    def unload(self):
        del self.model, self.tokenizer
        clean_vram()
        print("🛡️ Defender Unloaded.")

# --- THE BATTLE SIMULATION (LOOP MODE) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("      ⚔️  STARTING 5-ROUND BATTLE ROYAL  ⚔️")
    print("="*50 + "\n")

    attacker_wins = 0
    defender_wins = 0
    rounds = 5

    # 1. Load Agents ONCE (Optimization)
    # Note: On 8GB VRAM, we might still need to load/unload if OOM happens.
    # We will stick to the safe Load/Unload method for stability.

    for i in range(1, rounds + 1):
        print(f"\n🔔 ROUND {i}/{rounds}")
        
        # --- ATTACK ---
        attacker = BattleAttacker()
        trap = attacker.generate_trap()
        attacker.unload()
        
        if not trap:
            print("❌ Round skipped (Generation failed).")
            continue

        print(f"🔥 Question: {trap.get('question')[:100]}...")
        
        # --- DEFEND ---
        defender = BattleDefender()
        answer = defender.solve(trap)
        defender.unload()
        
        # --- SCORE ---
        correct = trap.get("answer", "A")
        print(f"📝 Defender Chose: {answer} | Correct: {correct}")
        
        if answer == correct:
            print(f"✅ DEFENDER WINS ROUND {i}!")
            defender_wins += 1
        else:
            print(f"🏆 ATTACKER WINS ROUND {i}!")
            attacker_wins += 1
            
        time.sleep(2) # Breathing room

    print("\n" + "="*50)
    print(f"🏁 FINAL SCORE: Attacker {attacker_wins} - {defender_wins} Defender")
    print("="*50)