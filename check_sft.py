from unsloth import FastLanguageModel

# Load the SFT model you trained in Step 1
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "models/q_agent_sft",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

prompt = """Below is an instruction that describes a task. Write a response that completes the request.
### Instruction:
You are an expert examiner. Generate a difficult multiple-choice question about Python Memory Management.
Output STRICT JSON format.
### Input:
Generate a hard question.
### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs, 
    max_new_tokens=300, 
    use_cache=True,
    # This stops the infinite loop
    repetition_penalty=1.2 
)
print(tokenizer.decode(outputs[0]))