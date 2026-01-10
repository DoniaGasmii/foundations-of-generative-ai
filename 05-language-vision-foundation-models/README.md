# Architectures I – Language & Vision Foundation Models

> Taught by Prof. Charlotte Bunne + Guest Lecture by Dr. Imanol Schlag (ETH AI Center)  
---

## Overview

This lecture bridges foundational building blocks (tokenization, attention) to **core architectures** powering modern foundation models in **language** and **vision**, emphasizing how **self-supervision objectives shape design choices**.

---

## Part 1: Language Foundation Models

### Key Concepts
- **Autoregressive modeling**: Predict next token given previous context (e.g., GPT).
- **Scaling laws**: Performance improves predictably with more data, parameters, and compute.
- **Emergent abilities**: At scale, models gain in-context learning, reasoning, and few-shot adaptation, even though trained only on next-token prediction.

### Apertus (Swiss AI Initiative)
- Open, multilingual LLMs (8B & 70B params) trained on **15T tokens** of public web data.
- **Compliant & transparent**: Respects `robots.txt`, removes PII/toxic content, no synthetic data.
- **Sovereign AI**: Built for research autonomy, multilinguality (~1000 languages), and Swiss/European alignment.
- Outperforms other open models (Llama 3.1, Mistral v0.3) on multilingual benchmarks.

>  Why it matters: Apertus provides a **reproducible, ethical baseline** for open LLM research without reliance on proprietary systems.

---

## Part 2: Vision Foundation Models

### From CNNs to Vision Transformers (ViT)
- **ViT tokenizes images** into fixed patches → treats them like words → processes with a Transformer.
- **Minimal inductive bias**: Learns global relationships better than CNNs at scale.

### Self-Supervised Learning Paradigms
| Objective        | Architecture       | Key Idea |
|------------------|--------------------|--------|
| **Contrastive** (e.g., DINO) | Encoder-only | Match representations of augmented views via self-distillation (no labels). |
| **Masked** (e.g., MAE)      | Encoder-Decoder | Reconstruct randomly masked image patches. High masking → semantic understanding. |
| **Autoregressive** (e.g., iGPT, VAR) | Decoder-only | Generate image tokens sequentially (raster-scan or coarse-to-fine). |

### Evolution: DINO → DINOv2 → DINOv3
- **DINO**: Emergent semantic segmentation from self-attention maps—no supervision!
- **DINOv2**: Scaled to 142M curated images; strong zero-shot transfer.
- **DINOv3**: Trained on **17B Instagram images**, 7B params, uses axial RoPE embeddings. Balances **global semantics** (DINO loss) and **local detail** (iBOT masked patch loss).

### Key Insight
> The **pretraining objective dictates architecture**:  
> - Contrastive → encoder + projection head  
> - Masked → asymmetric encoder-decoder  
> - Autoregressive → causal decoder  

---

## 🔗 Resources
- [Apertus Technical Report](https://arxiv.org/abs/2504.06219)  
- [DINOv2 Paper (TMLR 2024)](https://arxiv.org/abs/2304.07193)  
- [DINOv3 Preprint (Siméoni et al., 2025)]  
