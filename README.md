<p align="center">
<img src="[https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg](https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg)" align="center" width="30%">
</p>
<p align="center"><h1 align="center">AI-EXAMINER-RL-FINETUNE</h1></p>
<p align="center">
<em><code>Mastering Python Memory Management through Reinforcement Learning with Verifiable Rewards (RLVR).</code></em>

## Overview

This repository features a specialized dual-agent training pipeline designed to master complex Python concepts. By combining **Supervised Fine-Tuning (SFT)** for formatting and **Group Relative Policy Optimization (GRPO)** for reasoning, the system evolves a "Base" model into a high-difficulty **AI Examiner**.

The architecture uses an **Attacker/Defender** framework:

* **The Attacker (Q-Agent):** Generates logically dense, valid JSON multiple-choice questions.
* **The Defender (A-Agent):** Solve these questions with step-by-step reasoning.

---

## Features

* **Two-Stage Q-Agent Training:**
* **Warmup (SFT):** Aligns the model to strict JSON schemas.
* **Strategic Evolution (GRPO):** Optimizes for question complexity, technical keywords (e.g., `__del__`, `weakref`), and logical depth using verifiable reward functions.


* **Verifiable Reward Pipeline:** Uses a "Sniper Parser" in `rewards.py` to grade responses on format, length, and subject matter difficulty.
* **Unsloth Integration:** 2x faster training with optimized VRAM usage, enabling RL training on consumer-grade laptop GPUs (RTX 5050+).
* **Battle Interface:** A simulated environment where the trained agents are pitted against each other to verify learning progress.

---

## Project Structure

```sh
└── AI-Examiner-Reinforcemnt_learning-fine_tunning/
    ├── battle/                 # Production agent logic & simulation
    │   ├── a_agent.py          # Answer Agent interface
    │   ├── interface.py        # Battle UI/Logic
    │   └── q_agent.py          # Question Agent interface
    ├── check_sft.py            # Diagnostic tool for Stage 1 models
    ├── data/
    │   └── hard_questions.jsonl # Seed training data
    ├── train_defender.py       # SFT pipeline for the A-Agent
    └── training/               # Core Reinforcement Learning engine
        ├── 2_train_sft.py      # Q-Agent: Stage 1 SFT
        ├── 3_train_grpo.py     # Q-Agent: Stage 2 RL (GRPO)
        ├── rewards.py          # Logic for the Verifiable Reward functions
        └── verify_brain_final.py # Post-training validation suite

```

### Project Index

<details open>
<summary><b><code>Module Summaries</code></b></summary>
<details>
<summary><b>Root Directory</b></summary>
<blockquote>
<table>
<tr>
<td><b>train_defender.py</b></td>
<td>Initializes the SFT process for the A-Agent (The Solver).</td>
</tr>
<tr>
<td><b>check_sft.py</b></td>
<td>Verifies that the SFT warmup correctly taught the model JSON formatting.</td>
</tr>
</table>
</blockquote>
</details>
<details>
<summary><b>training</b></summary>
<blockquote>
<table>
<tr>
<td><b>3_train_grpo.py</b></td>
<td>The main RL loop. Uses Group Relative Policy Optimization to increase complexity.</td>
</tr>
<tr>
<td><b>rewards.py</b></td>
<td>The "Judge" script that analyzes model output for keywords and structure.</td>
</tr>
<tr>
<td><b>2_train_sft.py</b></td>
<td>Teaches the Question Agent the "Rules of the Game" (JSON format).</td>
</tr>
</table>
</blockquote>
</details>
</details>

---

## 🛠 Getting Started

### Prerequisites

* **Python:** 3.10+
* **Hardware:** NVIDIA GPU (Ampere/Ada architectures like RTX 30/40/50 series recommended for `bf16` support).
* **Compiler:** Windows users may need to disable `torch.compile` or install VS Build Tools.

### Installation

1. **Clone the Repository:**
```sh
git clone https://github.com/M3TAM0RPH0SIS/AI-Examiner-Reinforcemnt_learning-fine_tunning
cd AI-Examiner-Reinforcemnt_learning-fine_tunning

```


2. **Install Dependencies:**
```sh
pip install unsloth trl datasets accelerate bitsandbytes

```



### Usage

**Step 1: The Warmup (SFT)**

```sh
python training/2_train_sft.py

```

**Step 2: The Logic Workout (GRPO RL)**

```sh
python training/3_train_grpo.py

```

**Step 3: Verification**

```sh
python training/verify_brain_final.py

```

---

## Project Roadmap

* [x] **`Stage 1`**: Implement SFT pipeline for JSON schema alignment.
* [x] **`Stage 2`**: Deploy GRPO with custom keyword-based reward logic.
* [ ] **`Stage 3`**: Integration of Chain-of-Thought (CoT) reasoning for the Defender.
* [ ] **`Stage 4`**: Web-based Interface for live human-vs-agent testing.

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgments

* **Unsloth:** For providing the optimized kernels that make local RL training possible.
* **TRL Library:** For the GRPO implementation.

---
