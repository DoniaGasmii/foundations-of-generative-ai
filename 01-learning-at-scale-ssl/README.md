# Learning at Scale: Supervised, Self-Supervised, and Beyond

## Overview

This lecture establishes the foundational **learning paradigms** that underpin modern foundation models. 

The key insight: **self-supervised learning is the basic training principle of foundation models**, enabling them to learn from **vast amounts of unlabeled data** by creating supervision signals from the data itself.

---

## Core Learning Paradigms

| Paradigm | Data | Loss Function | Key Idea |
|----------|------|---------------|----------|
| **Supervised** | $(x, y)$ pairs | $\ell(f_\theta(x), y)$ | Learn mapping from labels |
| **Unsupervised** | $x$ only | $\ell(f_\theta(x))$ | Discover structure (clustering, dimensionality reduction) |
| **Semi-supervised** | Few $(x,y)$ + many $x$ | $\ell_{sup} + \lambda \cdot \ell_{unsup}$ | Leverage unlabeled data structure |
| **Self-supervised** | $x$ only → synthetic labels | $\ell(f_\theta(x_{partial}), x_{hidden})$ | Create pretext tasks from data |

All paradigms minimize **expected risk**: $\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(\theta; x, y)]$

---

## Self-Supervised Learning: The Foundation Model Training Principle

### Why It Matters

- *"The dark matter of intelligence"*: Common sense acquired from background knowledge is invisible yet forms the bulk of human intelligence, just as dark matter is invisible yet makes up most of the universe. SSL is proposed as the path to acquiring this background knowledge, since we can't label everything in the world for supervised learning. It enables learning from unlimited unlabeled data and powers GPT, BERT, CLIP, and virtually all modern foundation models.

**Blog:** [Self-supervised learning: The dark matter of intelligence](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/) — **Yann LeCun & Ishan Misra, Meta AI (2021)**

Key insights:
- **SSL = Predictive learning**: Predict hidden parts from visible parts → forces learning world structure
- **NLP vs Vision gap**: NLP has finite vocabulary (softmax works); images are continuous/infinite (harder to model uncertainty)
- **Energy-Based view**: SSL learns $F(x,y)$ assigning low energy to compatible pairs; main challenge is avoiding **collapse**
- **Contrastive vs Non-contrastive**: Contrastive (SimCLR, InfoNCE) pushes up energy on negatives but scales poorly; Non-contrastive (BYOL, SwAV) uses architectural tricks instead

<img width="500" alt="NLP vs Vision uncertainty" src="https://github.com/user-attachments/assets/8e73044e-5120-4482-b311-8f98b0dcf344" />

### The Two-Stage Pipeline
1. **Pretraining**: Learn general representations $\psi_\theta: x \mapsto z \in \mathbb{R}^m$ via pretext task
2. **Evaluation/Finetuning**: Adapt to downstream tasks (zero-shot, few-shot, or full finetuning)

---

## Three Main Self-Supervised Approaches

### 1. Contrastive Learning
**Principle**: Pull similar pairs together, push dissimilar pairs apart.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/9c8c8c91-f641-4dae-8ea4-65708f60103f" />

**InfoNCE Loss** (the modern standard):

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f'(x)^\top f(c)/\tau)}{\exp(f'(x)^\top f(c)/\tau) + \sum_{i=1}^{N-1}\exp(f'(x_i)^\top f(c)/\tau)}\right]$$

- $\tau$: temperature (controls sharpness)
- Positive pairs: augmentations of same instance
- Negative pairs: other samples in batch

**Key examples**: SimCLR (vision), SimCSE (language)

### 2. Masked Learning
**Principle**: Hide parts of input, predict what's missing.

$$\mathcal{L}_{\text{mask}} = \mathbb{E}_{x}\mathbb{E}_M\left[\frac{1}{|M|}\sum_{i \in M}\ell(x_i, \hat{x}_i)\right]$$

- Language: Cross-entropy over vocabulary (BERT)
- Vision: MSE for pixel reconstruction (MAE)

**Important**: Masked models learn **pseudo-likelihood**, not true joint probability.

### 3. Autoregressive Learning
**Principle**: Predict next element given previous context.

$$\mathcal{L}_{\text{AR}} = -\mathbb{E}_{x}\left[\sum_{t=1}^T \log p_\theta(x_t | x_{<t})\right]$$

Based on chain rule: $p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t | x_{<t})$

**Key property**: Learns exact likelihood → enables perplexity computation

**Examples**: GPT series, iGPT

---

## Perplexity vs Pseudo-Perplexity

| Metric | Formula | Used By |
|--------|---------|---------|
| **Perplexity** | $\exp\left(-\frac{1}{T}\sum_t \log p(x_t \| x_{<t})\right)$ | Autoregressive models |
| **Pseudo-Perplexity** | $\exp\left(-\frac{1}{T}\sum_t \log p(x_t \| x_{\setminus t})\right)$ | Masked models |

⚠️ **These are not comparable** despite similar formulas.

---

## From Learning Principle to Architecture

The self-supervision objective **dictates architectural choices**:

| Approach | Attention Pattern | Architecture |
|----------|-------------------|--------------|
| Contrastive | Full attention | Encoder + projection head |
| Masked | Bidirectional (with masks) | Encoder-decoder |
| Autoregressive | Causal (lower triangular) | Decoder-only |

---

## Scaling Laws

### The Key Finding (Kaplan et al., 2020)
Performance follows **power laws** with respect to:
- **N**: Number of parameters → $\mathcal{L} \approx (N_c/N)^{\alpha_N}$
- **D**: Dataset size → $\mathcal{L} \approx (D_c/D)^{\alpha_D}$  
- **C**: Compute → $\mathcal{L} \approx (C_c/C)^{\alpha_C}$

### Chinchilla Correction (Hoffmann et al., 2022)
Original scaling laws were **wrong** — models were overtrained relative to data.

**Optimal scaling**: $N \propto C^{0.5}$, $D \propto C^{0.5}$ (scale both equally!)

Chinchilla (70B params, 4× more data) beat GPT-3 (175B params) with same compute.

### Current State
Pre-training scaling may be hitting diminishing returns → shift toward:
- Post-training scaling
- Test-time scaling ("long thinking")

---

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) | Original neural scaling laws |
| [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) | Chinchilla optimal scaling |
| [Chen et al., 2020 (SimCLR)](https://arxiv.org/abs/2002.05709) | Contrastive learning framework |
| [van den Oord et al., 2018](https://arxiv.org/abs/1807.03748) | InfoNCE loss derivation |

---

## Exercises This Week

1. **InfoNCE Analysis**: Connect InfoNCE to cross-entropy and mutual information
2. **Pseudo-Likelihood**: Derive MLM ↔ pseudo-likelihood relationship
3. **Code**: Implement SimCLR, explore Pythia scaling behavior

---

## Key Takeaways

1. Self-supervised learning enables foundation models to learn from unlimited unlabeled data
2. The choice of pretext task (contrastive/masked/autoregressive) shapes the entire architecture
3. Scaling laws provide principled guidance for resource allocation
4. Model size alone isn't everything — data scaling matters equally (Chinchilla)




