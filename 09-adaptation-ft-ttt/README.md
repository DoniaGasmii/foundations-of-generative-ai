# In-Context Learning & Emergent Behavior - Adaptation, Fine-Tuning &Test-Time Training

## Introduction:

This lecture is split into two parts:

- **Part I: In-Context Learning and Emergent Behavior**  
  *(Dr. Charlotte Bunne, EPFL)*  
  Explores how foundation models solve new tasks without weight updates, using only prompts (zero-shot/few-shot) or frozen representations. Covers:
  - What in-context learning (ICL) is and why it’s surprising
  - Zero-shot vs. few-shot generalization
  - Frozen-backbone adaptation (e.g., linear probing, ABMIL)
  - Emergent abilities that appear only at scale

- **Part II: Adaptation, Fine-Tuning, and Test-Time Training**
  
<p align="left">
  <img src="https://github.com/user-attachments/assets/14aadbf9-e5fa-4e27-80cb-0c9e82bf45b7" alt="image" width="250" height="200" style="vertical-align: middle; margin-right: 20px;">
  (Prof. Andreas Krause & Dr. Jonas Hübotter, ETH Zürich)
</p>

  Investigates how models can *learn during inference* by maintaining parametric memory. Topics include:
  - The KV cache memory bottleneck in transformers
  - Self-attention as non-parametric kernel regression
  - Test-time training (TTT) with parametric memory (e.g., linear attention)
  - Active retrieval of useful examples at test time (e.g., SIFT)
  - Reinforcement learning at test time (e.g., TTC-RL, RL²)


