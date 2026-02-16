import os
import sys

# --- 🛑 WINDOWS COMPILER BYPASS 🛑 ---
# This forces PyTorch to use "Eager Mode" (Standard Python) instead of compiling C++ kernels.
# It makes the code run without Visual Studio Build Tools.
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo" # Silence compiler logs

import torch
# completely disable the compilation engine
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True

# -------------------------------------

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from rewards import format_reward_func, complexity_reward_func

# --- CONFIGURATION ---
MODEL_PATH = "models/q_agent_sft" 
OUTPUT_DIR = "models/q_agent_grpo"
DATA_FILE = "data/hard_questions.jsonl"

SYSTEM_PROMPT = """Below is an instruction that describes a task. Write a response that completes the request.
### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.
Output STRICT JSON format.
### Input:
Generate a hard question.
### Response:
"""

def train_grpo():
    print(f"🚀 Loading SFT Model for RLVR: {MODEL_PATH}")
    
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048, # Keep this high for loading
        load_in_4bit = True,
        gpu_memory_utilization = 0.6,
    )

    print("🔄 Merging SFT Adapters to create a clean base for RL...")
    model = model.merge_and_unload() 

    # 2. Add New Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        use_gradient_checkpointing = "unsloth", # Eager mode handles this fine
        random_state = 3407,
    )

    # 3. Data Prep
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.map(lambda x: {"prompt": SYSTEM_PROMPT})

    print("⚔️ Initializing GRPO Trainer (Eager Mode)...")
    
    # 4. Config (Optimized for Stability)
    training_args = GRPOConfig(
        output_dir = OUTPUT_DIR,
        learning_rate = 1e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        
        # --- STABILITY SETTINGS ---

        max_grad_norm = 0.1,            # Strict clipping
        
        # --- STOP YAPPING SETTINGS ---
        max_completion_length = 400,    # Short leash to prevent infinite loops
        temperature = 0.6,              # Focus mode
        
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations = 4,
        max_prompt_length = 256,
        max_steps = 100,
        
        use_vllm = False,               # Windows workaround
    )

    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [format_reward_func, complexity_reward_func],
        train_dataset = dataset,
        args = training_args,
        processing_class = tokenizer,
    )
    
    print("🔥 Starting RLVR Loop...")
    trainer.train()
    
    print(f"🏆 Saving RL-Trained Brain to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_grpo()