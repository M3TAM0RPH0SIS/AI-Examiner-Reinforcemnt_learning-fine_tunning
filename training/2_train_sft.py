import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "models/q_agent_sft"
import os
# Get the folder where this script lives (training/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root, then down into data
DATA_FILE = os.path.join(script_dir, "..", "data", "hard_questions.jsonl")
def train_sft():
    print(f"🚀 Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, use_gradient_checkpointing="unsloth"
    )

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # Format for SFT
    alpaca_prompt = """Below is an instruction that describes a task...
### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.
### Input:
Generate a hard question.
### Response:
{output_json}"""

    def format_func(examples):
        texts = []
        for q, opts, ans, exp in zip(examples["question"], examples["options"], examples["answer"], examples["explanation"]):
            json_str = f'{{"question": "{q}", "options": {opts}, "answer": "{ans}", "explanation": "{exp}"}}'
            texts.append(alpaca_prompt.format(output_json=json_str) + tokenizer.eos_token)
        return {"text": texts}

    dataset = dataset.map(format_func, batched=True)

    print("⚔️ Starting SFT...")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=2048,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR, max_steps=60, learning_rate=2e-4,
            per_device_train_batch_size=2, logging_steps=1, report_to="none"
        ),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ SFT Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_sft()