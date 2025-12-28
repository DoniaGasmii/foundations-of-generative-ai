# Generative Models I: Autoregressive, Adversarial, Autoencoder      <div style="margin-top: 20px;"></div>       *(+ Diffusion Covered Next Week)*

## Introduction:

**Goal**: Understand the four core generative modeling paradigms that power modern foundation models:

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/33f9f4c3-7a58-4197-ae72-9305c2096963" />

> We start with the first three this week; diffusion comes next!

Each learns the data distribution $p_{\theta}(x)$ **in an unsupervised way**: by discovering and encoding the underlying rules, patterns, and relationships that define the data’s structure **so it can generate new, realistic samples from that distribution**. 

This is achieved using one of four strategies: likelihood maximization, probabilistic reconstruction, adversarial games, or iterative denoising. We’ll compare their **objectives**, **trade-offs**, and **roles in foundation models**.

<img width="700" height="200" alt="image" src="https://github.com/user-attachments/assets/6002c9fc-64ea-43a9-828b-70b35f960d8b" />

## 1. Autoregressive Models

Autoregressive (AR) models generate data **sequentially**, predicting each element conditioned on all previous ones:  
$p(x) = \prod_{t=1}^T p(x_t \mid x_{<t})$. 

This factorization turns **density estimation** into a series of standard supervised learning problems; maximizing likelihood via next-token (or next-pixel) prediction.

We first encountered AR modeling in the [learning at scale, ssl](01-learning-at-scale-ssl/README.md) lecture as a **self-supervised learning strategy**: it enables exact likelihood optimization (unlike masking, which approximates it) and directly supports generation.

- AR models use **causal attention** (e.g., in GPT) to prevent future leakage. <img width="100" height="50" alt="image" src="https://github.com/user-attachments/assets/a8af57bc-59cb-4947-b15d-e8edd39eff3d" />

- Two parameterizations: **shared weights** (efficient; e.g., GPT) vs. **position-specific** (e.g., MADE).
  
 <img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/60f5cb8f-dc21-48ab-bb21-3bd89e23f80b" />

- Examples:
**Generative Pretrained Transformer (GPT)** (language), Radford et al., (2018)
**Image GPT(iGPT)** (vision), Chen et al., (2020)
  
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/b37b668a-4684-46f5-ba41-b64f449e610f" />













---




# Generative Models I - Autoregressive, Adversarial, and Autoencoder

## Overview
This week covers three fundamental approaches to generative modeling that form the foundation of modern AI systems: Autoregressive Models, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs).

- **Goal**: Learn to model the underlying data distribution p(x)
- **Key Insight**: By learning to generate data, models discover and encode the structure and patterns in the data
- **Maximum Likelihood Learning**: Minimize KL divergence between data and model distributions

---

## 1. Autoregressive Models

### Core Principle
Model joint distribution by decomposing it into a product of conditionals using the chain rule:
```
p(x) = p(x₁)p(x₂|x₁)p(x₃|x₁,x₂)...p(xₜ|x<ₜ)
```

### Key Properties
- Sequential prediction: one step at a time
- Fully general: works for any distribution, any ordering
- Optimizes exact likelihood
- Training reduces to supervised learning on each conditional term

### Parameterization Strategies
1. **Shared Parameters**: One model for all positions (e.g., RNNs, GPT) - more efficient
2. **Factorized Parameters**: Position-specific models (e.g., NADE, MADE) - more expressive

### Examples
- **Language**: GPT (Generative Pretrained Transformer)
- **Vision**: Image GPT (iGPT) - treats images as sequences of pixels

---

## 2. Variational Autoencoders (VAEs)

### Core Architecture
- **Encoder** qφ(z|x): Maps data to latent distribution
- **Latent space** Z: Lower-dimensional compressed representation
- **Decoder** pθ(x|z): Reconstructs data from latent code

### Key Innovation: Probabilistic Encoding
Instead of deterministic mapping, encode to distribution:
- Encoder outputs μ(x) and σ(x)
- Sample z ~ N(μ(x), σ²(x))
- Regularize latent space to match prior N(0,I)

### Evidence Lower Bound (ELBO)
```
L(φ,θ) = 𝔼[log pθ(x|z)] - D_KL(qφ(z|x)||p(z))
         ↑                    ↑
    Reconstruction      Regularization
```

### Reparameterization Trick
To enable gradient computation through sampling:
- Instead of z ~ N(μ, σ²)
- Write z = μ + σ·ε where ε ~ N(0,I)
- Now z is differentiable w.r.t. μ and σ

### Properties
- **Strengths**: Principled probabilistic framework, smooth latent space, fast generation
- **Limitations**: Blurry reconstructions (Gaussian likelihood), posterior collapse, reconstruction vs. generation trade-off

---

## 3. Generative Adversarial Networks (GANs)

### Core Principle
Two-player minimax game between:
- **Generator** gθ: Creates fake samples from noise
- **Discriminator** fφ: Distinguishes real from fake

### Objective
```
min max V(fφ, gθ) = 𝔼[log fφ(x)] + 𝔼[log(1 - fφ(gθ(z)))]
 gθ  fφ
```

### Theoretical Foundation
- Training GANs minimizes Jensen-Shannon divergence between distributions
- Optimal discriminator: f*(x) = p_data(x) / (p_data(x) + p_g(x))
- At equilibrium: pθ = p_data and discriminator outputs 0.5 everywhere

### Common Issues
- **Mode collapse**: Generator produces limited diversity
- **Training instability**: Difficult to balance generator and discriminator
- **Vanishing gradients**: When distributions don't overlap

### Wasserstein GAN (WGAN)
- Uses Wasserstein distance (W₁) instead of JS divergence
- Provides meaningful gradients even when distributions don't overlap
- Discriminator becomes "critic" outputting real values
- Requires 1-Lipschitz constraint (via weight clipping, gradient penalty)

### Properties
- **Strengths**: Sharp, high-quality samples; no explicit density needed; flexible architecture
- **Limitations**: Training instability, mode collapse, no explicit likelihood

---

## Important Mathematical Concepts

### KL Divergence
```
D_KL(p||q) = 𝔼_p[log p(x)/q(x)]
```
- Measures how one distribution differs from another
- Always non-negative, zero iff p = q
- Not symmetric

### Jensen-Shannon Divergence
```
D_JS(p||q) = ½D_KL(p||m) + ½D_KL(q||m)  where m = ½(p+q)
```
- Symmetric version of KL divergence
- Bounded: 0 ≤ D_JS ≤ log 2

### Wasserstein Distance
```
W₁(p,q) = inf_γ 𝔼[||x-y||]
```
- Minimum cost to transport mass from one distribution to another
- Provides meaningful gradients when distributions don't overlap

---

## Conditional Generative Models
Model p(x|c) instead of p(x) for controlled generation
- **c**: condition (text prompt, class label, image, etc.)
- **Methods**: concatenation, cross-attention, classifier guidance

---

## Role in Foundation Models

### Three Key Insights
1. **Generative Modeling as Self-Supervision**: Predicting unobserved from observed provides unlimited training signal
2. **Generative Models as Architectural DNA**: Spans tokenization, core architectures, learning principles
3. **From Representation to Simulation**: Traditional models learn "what is," generative models enable "what if"

### Why Generative Training Produces Foundation Capabilities
1. Compression requires understanding
2. Self-supervision scales with raw data
3. Emergence from scale: capabilities not explicitly trained
4. Conditional generation = universal interface

---

## Code Exercises (Notebook 3)
- Task 1-4: Theoretical foundations and mathematical derivations
- Demo: Exploring GAN training dynamics

---

## Looking Ahead
- Week 4: Diffusion Models (the fourth philosophical approach)
- Connection between optimal transport, GANs, and diffusion models
- Modern systems combine insights from all approaches (e.g., GPT-4o)

---

## Key Papers & Resources
- Goodfellow et al. (2014): Generative Adversarial Networks
- Kingma & Welling (2014): Auto-Encoding Variational Bayes
- Arjovsky et al. (2017): Wasserstein GAN
- Chen et al. (2020): Image GPT
- Radford et al. (2018): GPT
- ICML 2023 Tutorial: Optimal Transport in Learning, Control, and Dynamical Systems
