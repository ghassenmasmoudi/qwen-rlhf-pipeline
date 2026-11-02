import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import torch

SFT_MODEL = "qwen-sft-final"
OUTPUT_MODEL = "qwen-dpo-final"
MAX_STEPS = 500

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL, dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

def to_conversational(ex):
    prompt = ex["chosen"][0]["content"]
    chosen = next(m["content"] for m in reversed(ex["chosen"]) if m.get("role") == "assistant")
    rejected = next(m["content"] for m in reversed(ex["rejected"]) if m.get("role") == "assistant")
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }

dpo_dataset = dataset.map(to_conversational, remove_columns=dataset.column_names)
print(f"Dataset DPO complet : {len(dpo_dataset)} exemples")

config = DPOConfig(
    output_dir="qwen-dpo",
    beta=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=MAX_STEPS,
    learning_rate=5e-6,
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    bf16=False,
    max_length=1024,
)

trainer = DPOTrainer(
    model=model, args=config, train_dataset=dpo_dataset, processing_class=tokenizer
)
trainer.train()
trainer.save_model(OUTPUT_MODEL)
tokenizer.save_pretrained(OUTPUT_MODEL)
print(f"DPO sauvegardé dans {OUTPUT_MODEL}")