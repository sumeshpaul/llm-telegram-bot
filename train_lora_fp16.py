# train_lora_fp16.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model: fp16 from Hugging Face (not quantized)
model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},         # ✅ streams layers directly to GPU
    low_cpu_mem_usage=False,    # ✅ disables meta tensor load
    load_in_4bit=False,
    load_in_8bit=False
)

print("🧩 Applying LoRA configuration...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()

print("📚 Loading dataset...")
dataset = load_dataset("json", data_files="train_dataset_v2.json")

# Tokenize examples
def tokenize(batch):
    prompts = []
    for inst, inp, out in zip(batch.get("instruction", []), batch.get("input", []), batch.get("output", [])):
        prompt = f"{inst}\n{inp}\n{out}"
        prompts.append(prompt)
    return tokenizer(prompts, padding="max_length", truncation=True, max_length=256)

print("🧼 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training settings
training_args = TrainingArguments(
    output_dir="/app/final_lora_model_fp16"
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    torch_compile=True,
)

print("🚀 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

print("💾 Saving model...")
trainer.save_model("./final_lora_model_fp16")
tokenizer.save_pretrained("./final_lora_model_fp16")

print("✅ Training complete.")
