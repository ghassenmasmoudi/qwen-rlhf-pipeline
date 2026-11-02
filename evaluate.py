import torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

RM_PATH = "qwen-reward-final"
SFT_PATH = "qwen-sft-final"
DPO_PATH = "qwen-dpo-final"
N_EVAL = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Partie 1 : évaluation du Reward Model ----------
print("=== Évaluation du Reward Model ===")

rm = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH, num_labels=1, dtype=torch.float16, device_map="auto"
).eval()
rm_tok = AutoTokenizer.from_pretrained(RM_PATH)
test = load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(200))

def score_conversation(messages):
    text = rm_tok.apply_chat_template(messages, tokenize=False)
    inputs = rm_tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        return rm(**inputs).logits.item()

correct = 0
rm_diffs, human_diffs = [], []
len_c_long, correct_c_long = 0, 0

for ex in test:
    s_c = score_conversation(ex["chosen"])
    s_r = score_conversation(ex["rejected"])
    if s_c > s_r:
        correct += 1
    rm_diffs.append(s_c - s_r)
    human_diffs.append(ex["score_chosen"] - ex["score_rejected"])
    len_c = len(ex["chosen"][-1]["content"])
    len_r = len(ex["rejected"][-1]["content"])
    if len_c > len_r:
        len_c_long += 1
        if s_c > s_r:
            correct_c_long += 1

acc = correct / len(test)
corr = np.corrcoef(rm_diffs, human_diffs)[0, 1]
acc_long = correct_c_long / len_c_long if len_c_long else 0
print(f"Pairwise accuracy : {acc:.3f} ({correct}/{len(test)})")
print(f"Corrélation RM vs humains : {corr:.3f}")
print(f"Accuracy quand chosen plus long : {acc_long:.3f} ({correct_c_long}/{len_c_long})")

del rm, rm_tok
torch.cuda.empty_cache()

# ---------- Partie 2 : comparaison SFT vs DPO ----------
print("\n=== Comparaison SFT vs DPO ===")

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
prompts = [ex["chosen"][0]["content"] for ex in dataset.select(range(N_EVAL))]

def generate_textes(model_path, prompts, max_new_tokens=512):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    textes, arrets = [], 0
    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[0][len(inputs["input_ids"][0]):]
        if tok.eos_token_id in new_tokens:
            idx = (new_tokens == tok.eos_token_id).nonzero()[0].item()
            new_tokens = new_tokens[:idx]
            arrets += 1
        textes.append(tok.decode(new_tokens, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    print(f"{model_path} : {arrets}/{len(prompts)} arrêts naturels")
    return textes

textes_sft = generate_textes(SFT_PATH, prompts)
textes_dpo = generate_textes(DPO_PATH, prompts)

len_sft = [len(t) for t in textes_sft]
len_dpo = [len(t) for t in textes_dpo]
t_len, p_len = stats.ttest_rel(len_dpo, len_sft)
print(f"\nLongueur SFT : {np.mean(len_sft):.0f} ± {np.std(len_sft):.0f}")
print(f"Longueur DPO : {np.mean(len_dpo):.0f} ± {np.std(len_dpo):.0f}")
print(f"Différence : {np.mean(len_dpo) - np.mean(len_sft):+.0f} (t={t_len:.3f}, p={p_len:.4f})")

# Score RM
rm = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH, num_labels=1, dtype=torch.float16, device_map="auto"
).eval()
rm_tok = AutoTokenizer.from_pretrained(RM_PATH)

def score_reponses(prompts, textes):
    scores = []
    for p, r in zip(prompts, textes):
        messages = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        text = rm_tok.apply_chat_template(messages, tokenize=False)
        inputs = rm_tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            scores.append(rm(**inputs).logits.item())
    return scores

scores_sft = score_reponses(prompts, textes_sft)
scores_dpo = score_reponses(prompts, textes_dpo)
t_rm, p_rm = stats.ttest_rel(scores_dpo, scores_sft)
print(f"\nScore RM SFT : {np.mean(scores_sft):.3f} ± {np.std(scores_sft):.3f}")
print(f"Score RM DPO : {np.mean(scores_dpo):.3f} ± {np.std(scores_dpo):.3f}")
print(f"Différence : {np.mean(scores_dpo) - np.mean(scores_sft):+.3f} (t={t_rm:.3f}, p={p_rm:.4f})")