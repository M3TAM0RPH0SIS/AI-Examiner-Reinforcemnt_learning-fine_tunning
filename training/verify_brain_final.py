import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from rewards import format_reward_func, complexity_reward_func
import os

# --- CONFIGURATION ---
# We start RL from the model we just SFT'd in Step 2
MODEL_PATH = "models/q_agent_sft" 
OUTPUT_DIR = "models/q_agent_grpo"
DATA_FILE = "data/hard_questions.jsonl"

# The System Prompt must match what was used in SFT (Step 2)
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
    
    # 1. Load the SFT Model (The Student)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        load_in_4bit = True,
        gpu_memory_utilization = 0.6, # Keep room for the buffer
    )
    
    # 2. Configure LoRA for RL
    # We need to ensure we are training the adapters, not the frozen base
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Increase R for RL to allow more "thinking" capacity
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Load and Format Data
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # GRPO requires a 'prompt' column. We map our system prompt to it.
    def add_prompt_column(example):
        return {"prompt": SYSTEM_PROMPT}
    
    dataset = dataset.map(add_prompt_column)

    print("⚔️ Initializing GRPO Trainer...")
    
    # 4. Define the RL Hyperparameters
    training_args = GRPOConfig(
        output_dir = OUTPUT_DIR,
        learning_rate = 5e-6,           # Very low LR is critical for RL stability
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,                    # Use Bfloat16 for stability if on Ampere+
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        
        # GRPO Specifics
        num_generations = 4,            # G=4: Generate 4 answers per prompt to compare
        max_prompt_length = 256,
        max_completion_length = 512,    # Allow space for "Thinking"
        max_steps = 150,                # Train longer than SFT
        save_steps = 50,
        report_to = "none",
        use_vllm = True,                # Enable vLLM for fast generation (Crucial)
    )

    # 5. The Trainer
    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [format_reward_func, complexity_reward_func],
        train_dataset = dataset,
        args = training_args,
        processing_class = tokenizer,
    )
    
    print("🔥 Starting RLVR Loop...")
    print("   (This phase generates multiple responses and scores them against each other)")
    trainer.train()
    
    print(f"🏆 Saving RL-Trained Brain to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_grpo()