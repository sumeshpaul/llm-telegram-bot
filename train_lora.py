import os
import torch
import json
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Set CUDA memory expansion config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
OUTPUT_DIR = "./final_lora_model_v2"
LOCAL_JSON = "./train_dataset_v2.json"

# Load model with quantization config
print("💪 Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for training with LoRA
print("💡 Preparing model for LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load local dataset
print("📦 Loading local dataset...")
with open(LOCAL_JSON, "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

# Format each example safely
def format_prompt(record):
    instruction = record.get("instruction", "")
    input_text = record.get("input") or ""
    output_text = record.get("output", "")

    prompt = f"### Instruction:\n{instruction.strip()}\n"
    if input_text.strip():
        prompt += f"\n### Input:\n{input_text.strip()}"
    prompt += f"\n\n### Response:\n{output_text.strip()}"
    return {"text": prompt}

dataset = dataset.map(format_prompt)

# Tokenize dataset
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Start training
print("🚀 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()

# Save model
print("💾 Saving final model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
