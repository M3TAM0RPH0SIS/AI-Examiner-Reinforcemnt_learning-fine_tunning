import torch
from unsloth import FastLanguageModel
from rewards import parse_model_output
import os

# --- CONFIG ---
# Validating the model we just trained with RL
MODEL_PATH = "models/q_agent_grpo" 

# Must match the prompt used in 3_train_grpo.py EXACTLY
SYSTEM_PROMPT = """Below is an instruction that describes a task. Write a response that completes the request.
### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.
Output STRICT JSON format.
### Input:
Generate a hard question.
### Response:
"""

def verify_brain():
    print(f"🧠 Loading RLVR-Trained Brain from: {MODEL_PATH}...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_PATH,
            max_seq_length = 2048,
            load_in_4bit = True,
            gpu_memory_utilization = 0.6,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   (Did you run 3_train_grpo.py yet?)")
        return

    print("\n🧪 STARTING STRESS TEST (5 Generations)...")
    print("="*60)

    stats = {"Clean": 0, "Rescued": 0, "Failed": 0}

    # Generate 5 samples to check stability
    for i in range(1, 6):
        print(f"\n[Test {i}/5] Generating...")
        
        inputs = tokenizer([SYSTEM_PROMPT], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.8, # Slightly high to test diversity
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        full_text = tokenizer.decode(outputs[0])
        # Extract just the new text (the response)
        raw_response = full_text.split("### Response:")[-1].replace("<|im_end|>", "").strip()

        # --- THE VERIFIER ENGINE ---
        # We use the EXACT same logic that gave rewards during training
        data, is_clean = parse_model_output(raw_response)

        # Reporting
        if data and is_clean:
            print(f"✅ PERFECT JSON")
            stats["Clean"] += 1
        elif data and not is_clean:
            print(f"⚠️ BROKEN JSON (Regex Rescued)")
            stats["Rescued"] += 1
        else:
            print(f"❌ GARBAGE OUTPUT")
            stats["Failed"] += 1

        # Show a snippet of what it wrote
        preview = raw_response[:200].replace("\n", " ") + "..."
        print(f"   Output: {preview}")
    
    print("\n" + "="*60)
    print("📊 FINAL SCORECARD")
    print(f"   - Perfect Format:  {stats['Clean']}/5")
    print(f"   - Recoverable:     {stats['Rescued']}/5")
    print(f"   - Total Failures:  {stats['Failed']}/5")
    
    if stats["Clean"] >= 4:
        print("\n🚀 STATUS: READY FOR BATTLE. The RL training worked!")
    elif stats["Clean"] + stats["Rescued"] >= 4:
        print("\n⚠️ STATUS: PASSABLE. It's messy, but the interface won't crash.")
    else:
        print("\n🛑 STATUS: FAILURE. The model has collapsed. Tune Hyperparams.")

if __name__ == "__main__":
    verify_brain()