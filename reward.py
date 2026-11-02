import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
import torch

INPUT_MODEL = "qwen-sft-final"
OUTPUT_MODEL = "qwen-reward-final"

model = AutoModelForSequenceClassification.from_pretrained(
    INPUT_MODEL, num_labels=1, dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL)

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
print(f"Dataset complet : {len(dataset)} exemples")

config = RewardConfig(
    output_dir="qwen-reward",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    bf16=False,
    max_length=1024,
)

trainer = RewardTrainer(
    model=model, args=config, train_dataset=dataset, processing_class=tokenizer
)
trainer.train()
trainer.save_model(OUTPUT_MODEL)
tokenizer.save_pretrained(OUTPUT_MODEL)
print(f"Reward Model sauvegardé dans {OUTPUT_MODEL}")