# RLHF Pipeline on Qwen2.5-0.5B — Reward Hacking Demonstration

A complete, from-scratch **Reinforcement Learning from Human Feedback (RLHF)** pipeline built on a 0.5B-parameter language model, trained on a single free Kaggle T4 GPU (16 GB VRAM).

The objective is not to produce a state-of-the-art aligned model, but to **empirically demonstrate Reward Hacking**: how a policy trained against an imperfect Reward Model learns to exploit its biases rather than improve actual response quality.

## 🏗️ Pipeline Architecture

```text
Qwen2.5-0.5B
     |
     v
Supervised Fine-Tuning (SFT)
     |
     | Capybara Dataset
     v
   SFT Model
     |
     +--------------------------+
     |                          |
     v                          v
Reward Model (RM)          DPO Alignment
     |                          |
     | UltraFeedback            | UltraFeedback
     v                          v
Reward Model              DPO Model
     |                          |
     +-------------+------------+
                   |
                   v
              Evaluation
                   |
                   v
        Reward Hacking Analysis
```

## 📋 Pipeline Stages

| **Stage** | **Script** | **Input** | **Output** |
|---|---|---|---|
| Supervised Fine-Tuning | `sft.py` | `Qwen/Qwen2.5-0.5B` + `trl-lib/Capybara` | `qwen-sft-final` |
| Reward Modeling | `reward.py` | `qwen-sft-final` + `trl-lib/ultrafeedback_binarized` | `qwen-reward-final` |
| DPO Alignment | `dpo.py` | `qwen-sft-final` + `trl-lib/ultrafeedback_binarized` | `qwen-dpo-final` |
| Evaluation | `evaluate.py` | SFT, Reward Model and DPO models | Metrics, statistical tests and plots |

## ⚙️ Execution Order

Run the scripts sequentially:

```bash
python sft.py
# Produces qwen-sft-final

python reward.py
# Produces qwen-reward-final

python dpo.py
# Produces qwen-dpo-final

python evaluate.py
# Evaluates the models and prints all metrics
```

## 💻 Hardware & Training Constraints

The entire pipeline was designed to run on a single free Kaggle T4 GPU.

| **Parameter** | **Configuration** |
|---|---|
| GPU | NVIDIA Tesla T4 |
| VRAM | 16 GB |
| GPU configuration | Single GPU (`CUDA_VISIBLE_DEVICES=0`) |
| Precision | FP16 (`fp16=True`) |
| BF16 | Disabled (`bf16=False`) |
| Batch size | 1 |
| Gradient accumulation | 2–4 steps |
| Maximum sequence length | 1024 |
| DPO training steps | 500 |
| DPO beta | 0.1 |

These configurations are deliberately conservative to fit within the T4's memory constraints. They prioritize training stability and compatibility over execution speed.

## 📊 Experimental Results

### 1. Reward Model Evaluation

The Reward Model was evaluated on 200 held-out preference pairs from the test split of `trl-lib/ultrafeedback_binarized`.

| **Metric** | **Result** |
|---|---:|
| Pairwise accuracy | **0.730 (146/200)** |
| Correlation with human scores | **0.279** |
| Accuracy when `chosen` is longer | **0.802 (93/116)** |
| Proportion of pairs where `chosen` is longer | **58% (116/200)** |

**Interpretation:**

The Reward Model successfully learned a preference signal, achieving 73% pairwise accuracy. However, it also developed a measurable **length bias**.

Its accuracy reaches 80.2% when the preferred response (`chosen`) is longer, suggesting that response length influences its predictions beyond actual response quality.

### 2. DPO vs SFT Comparison

The SFT and DPO models were evaluated using 100 identical prompts.

| **Metric** | **SFT** | **DPO** | **Difference** | **Paired t-test** |
|---|---:|---:|---:|---:|
| Response length (characters) | 2107 ± 652 | 2315 ± 917 | **+208** | t = 2.356, **p = 0.0205** |
| Reward Model score | -4.416 ± 2.941 | -2.990 ± 3.036 | **+1.426** | t = 5.510, **p < 0.0001** |

**Key observations:**

- DPO significantly increased the average response length by 208 characters.
- The Reward Model score increased by 1.426 points.
- Both differences are statistically significant at the 5% significance level.
- The DPO policy learned to generate responses that receive higher scores from the Reward Model.

## 🔬 Reward Hacking: Experimental Interpretation

The experimental results illustrate a potential Reward Hacking mechanism through three observations:

**1. The Reward Model develops a bias**

The Reward Model achieves 80.2% accuracy when the preferred response is longer, indicating that response length may act as a proxy for quality.

**2. DPO optimizes the Reward Model score**

Following preference optimization, the DPO model achieves a significantly higher Reward Model score (+1.426, p < 0.0001).

**3. The policy generates longer responses**

The average response length increases by 208 characters (p = 0.0205), consistent with the hypothesis that the policy exploits the Reward Model's preference for longer outputs.

**Conclusion:**

These observations provide an empirical demonstration of Reward Hacking: a policy can learn to optimize an imperfect reward signal without necessarily improving the underlying human-perceived quality.

The KL regularization parameter (`beta=0.1`) helped constrain policy drift but did not eliminate the observed behavior over 500 training steps.

## ⚠️ Limitations & Technical Considerations

### 1. Generation and EOS Behavior

Only 1 out of 100 generations naturally stopped at an EOS token. Responses were therefore truncated at 512 tokens, making character length a practical proxy for response length.

This limits the precision of the length-based analysis.

### 2. Small Evaluation Sample

The SFT vs DPO comparison uses 100 prompts. This sample is sufficient to detect a moderate effect in this experiment, but larger evaluation sets would provide more robust estimates.

### 3. Reward Model Evaluation Bias

DPO responses were evaluated using the same Reward Model used to optimize the policy.

This is intentionally circular: it measures how effectively the policy satisfies its reward function, not whether its outputs are objectively better or more useful to humans.

An independent evaluator or human preference study would be necessary to establish actual quality improvements or degradation.

### 4. TRL Version Compatibility

Recent TRL versions (≥ 1.14) may use a `chunked_logprob` kernel that encounters compatibility issues with pre-Ampere GPUs such as the NVIDIA T4, particularly with BF16.

For compatibility, the DPO script may require:

```bash
pip install "trl<1.14"
```

The exact compatible version should be validated against the installed Transformers and PyTorch versions.

## 📁 Repository Structure

```text
RLHF-Reward-Hacking/
│
├── sft.py              # Supervised Fine-Tuning
│
├── reward.py           # Reward Model training
│
├── dpo.py              # Direct Preference Optimization
│
├── evaluate.py         # Reward Model evaluation
│                       # SFT vs DPO comparison
│                       # Statistical analysis and plots
│
└── README.md           # Project documentation
```

## 🛠️ Technologies & Libraries

- **Python** — Main programming language.
- **PyTorch** — Deep learning framework.
- **Hugging Face Transformers** — Model loading and fine-tuning.
- **Hugging Face TRL** — SFT, Reward Modeling and DPO.
- **Hugging Face Datasets** — Dataset loading and processing.
- **SciPy** — Statistical analysis and paired t-tests.
- **Matplotlib** — Experimental result visualization.
- **Kaggle** — GPU training environment.

## 📚 Datasets & References

**Base Model**

- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)

**Training Datasets**

- [trl-lib/Capybara](https://huggingface.co/datasets/trl-lib/Capybara) — Supervised Fine-Tuning.
- [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) — Reward Modeling and DPO.

**Documentation**

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [SciPy](https://docs.scipy.org/doc/scipy/)

## 📄 License

This project is released under the **MIT License**.