import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

INPUT_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_MODEL = "qwen-sft-final"

model = AutoModelForCausalLM.from_pretrained(
    INPUT_MODEL, dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("trl-lib/Capybara", split="train")

config = SFTConfig(
    output_dir="qwen-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    bf16=False,
    max_length=1024,
)

trainer = SFTTrainer(
    model=model, args=config, train_dataset=dataset, processing_class=tokenizer
)
trainer.train()
trainer.save_model(OUTPUT_MODEL)
tokenizer.save_pretrained(OUTPUT_MODEL)
print(f"SFT sauvegardé dans {OUTPUT_MODEL}")