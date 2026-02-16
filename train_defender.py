import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# --- CONFIG ---
DATA_FILE = "data/hard_questions.jsonl"
OUTPUT_DIR = "models/a_agent_sft" # Distinct folder for the Defender
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def train_defender():
    print(f"🛡️ Loading {MODEL_NAME} for DEFENDER Training...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, use_gradient_checkpointing="unsloth"
    )

    print("📂 Loading Data...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # --- THE MAGIC SWITCH ---
    # We train the model to output the ANSWER, not the QUESTION.
    solver_prompt = """Below is a question about Python. Solve it step-by-step.
### Question:
{question}
### Options:
{options}
### Answer:
{explanation}
Final Answer: {answer}"""

    def format_func(examples):
        texts = []
        for q, opts, ans, exp in zip(examples["question"], examples["options"], examples["answer"], examples["explanation"]):
            # The Input is the Question + Options
            # The Output (what it learns) is Explanation + Answer letter
            text = solver_prompt.format(
                question=q, 
                options=opts, 
                explanation=exp, 
                answer=ans
            ) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_func, batched=True)

    print("🛡️ Starting Defender Training...")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=2048,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR, 
            max_steps=60,  # Quick run for now
            learning_rate=2e-4, 
            per_device_train_batch_size=2, 
            logging_steps=1,
            report_to="none"
        ),
    )
    trainer.train()
    
    print(f"✅ Defender Saved to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_defender()