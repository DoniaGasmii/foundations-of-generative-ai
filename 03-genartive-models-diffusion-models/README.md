# Generative Models II : Diffusion Models

## Introduction

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. They have emerged as one of the most powerful approaches for generative modeling, achieving state-of-the-art results in image, video, 3D, and multimodal generation.

**Core Principle:** Learn to denoise by iteratively recovering data from pure noise through a learned reverse process.

---

## Part I: Fundamentals

### 1. Core Concept: Generation by Noising and Denoising

**Key Idea:** Diffusion models work in two phases:

- **Forward Process (Fixed):** Gradually adds Gaussian noise to data until it becomes pure noise
- **Reverse Process (Learned):** Neural network learns to denoise, iteratively recovering data from noise

<img width="500" height="100" alt="image" src="https://github.com/user-attachments/assets/101f8b9a-c331-4878-bafd-42b54e43145d" />

### 2. Mathematical Frameworks

#### Discrete-Time Formulation (DDPM)
```
Forward: q(x_t|x_{t-1}) - Markov chain
Direct sampling: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
Reverse: p_θ(x_{t-1}|x_t) - Learned Gaussian transitions
```

#### Stochastic Differential Equations (SDEs)
```
Forward SDE: dx_t = -½β(t)x_t dt + √β(t)dw_t
Reverse SDE: Uses score function ∇_x log q_t(x_t)
Enables flexible sampling strategies
```

#### Probability Flow ODEs
- Deterministic sampling alternative to SDEs
- Same marginal distributions as SDE
- Faster sampling but no stochastic error correction

### 3. Training Objectives

#### Denoising Score Matching
```
Objective: min_θ E[||s_θ(x_t, t) - ∇_x log q_t(x_t|x_0)||²]
```
- Match score of diffused data (tractable) instead of marginal distribution (intractable)

#### Noise Prediction Parameterization
```
Simplified: min_θ E[w(t)||ε - ε_θ(x_t, t)||²]
```
- Reparameterize to predict noise directly
- Loss weighting w(t) balances content vs. detail across timesteps

#### Connection to ELBO
- Diffusion models = hierarchical VAEs with fixed encoder
- Training maximizes Evidence Lower Bound
- Reweighted ELBO leads to practical objectives

### 4. Flow Matching

**Alternative Framework:** Directly learn vector field connecting noise to data

- Train by interpolating pairs (x_0, x_1)
- Sample by simulating ODE along learned vector field
- Simpler conceptually than explicit SDE formulation
- **Equivalence:** Flow matching with Gaussian noise ≡ diffusion models

### 5. Architecture Design

#### U-Net Based Architectures
- Convolutional ResNet blocks with skip connections
- Self-attention layers for global context
- Time conditioning via sinusoidal embeddings
- Adaptive Group Normalization for timestep injection

#### Diffusion Transformers (DiT)
- Vision Transformer backbone on image patches
- Similar to LLM architecture but for images
- Competitive performance with proper scaling

### 6. Classifier-Free Guidance

**Critical Technique** for high-quality conditional generation:

```
Goal: Sample from p(x|c) ∝ p(x)^(1-ω) p(x|c)^(1+ω)
Implementation: ∇ log p̃(x|c) = (1+ω)∇ log p(x|c) - ω∇ log p(x)
```

- Train both conditional and unconditional models
- Guidance scale ω controls quality-diversity tradeoff
- Significantly improves sample quality and prompt adherence

### 7. Latent Diffusion Models (LDMs)

**Key Innovation:** Train diffusion in compressed latent space instead of pixel space

#### Two-Stage Training:

**Stage 1 - Autoencoder:**
```
Encoder: z = ℰ(x) - compress (e.g., 8× downsampling)
Decoder: x̃ = 𝒟(z) - reconstruct
Losses: Perceptual (LPIPS) + adversarial + regularization
```

**Stage 2 - Latent Diffusion:**
- Train diffusion model on latents z instead of pixels x
- Much more efficient: lower resolution and dimensionality
- Decoder handles fine details; diffusion handles global structure

#### Latent Space Regularization:
- **KL Regularization:** Weak regularization toward standard normal (VAE-style)
- **VQ Regularization:** Discrete codebook with large vocabulary (VQ-VAE style)

#### Advantages:
- **Computational efficiency:** Training/sampling in compressed space
- **Better quality:** Regularization + adversarial training = smooth latent space
- **Flexibility:** Adapt autoencoder to different modalities
- **Separation of concerns:** Diffusion models semantics; decoder generates details

---

## Part II: Advanced Topics

### 1. What Makes Diffusion Great?

#### Perceptual Inductive Bias
- **Key Insight:** Human perception varies in sensitivity to different frequency components
  - Less sensitive to high-frequency details
  - More sensitive to low-frequency content (global structure)

#### Reweighted ELBO
```
Loss: ℒ_w(x) = ½ E[w(λ_t) · dλ/dt · ||ε_θ(z_t; λ_t) - ε||²]
```
- Different weighting functions emphasize different frequency components
- High noise (low SNR) → model low-frequency content
- Low noise (high SNR) → model high-frequency details
- Reweighting allows model to spend more capacity on perceptually important low frequencies

#### Additional Strengths:
- **Simple regression objective:** Stable for scaling up
- **Divide and conquer:** Break generation into small steps
- **Weight sharing:** Borrowing strength across timesteps (data augmentation effect)

### 2. Fast Sampling Methods

#### Approach 1: Fast ODE/SDE Solvers
- **First-order methods:** Euler's method
- **Higher-order methods:** RK4, Linear Multistep, Heun's method
- Trade-off: Speed vs. quality
- Example: DPM-Solver++ achieves good quality in 15-20 steps

#### Approach 2: Trajectory Distillation
- Learn function that reproduces multi-step mapping in one step
- Examples: Progressive distillation, consistency models
- **Pros:** One-to-one mapping to teacher model
- **Cons:** Hard to reduce to very few steps

#### Approach 3: Variational Distillation
- Match student distribution to teacher via variational objective
- **Pros:** One-step generation, very fast
- **Cons:** Mode collapse risk, extra compute cost

### 3. Video Diffusion Models

#### Early Approaches: Video Diffusion (Ho et al. 2020)
- U-Net with factorized spatial-temporal attention
- Simple extension of image diffusion to video

#### Cascaded Framework: Imagen Video (2022)
```
Base → TSR → SSR → TSR → SSR (multiple models)
```
- **Issues:** Many models to train/tune, error accumulation, expensive serving

#### Unified Framework: SORA (Feb 2024)
- Single latent diffusion model with transformer architecture
- **Benefits:**
  - Scales well with compute and model size
  - Supports arbitrary aspect ratios
  - Simpler training pipeline

#### Veo Series (Google)

**Veo 1 (May 2024):**
- Initial video generation capabilities
- Issues: Slow motion, physics hallucinations, weak text following

**Veo 2 (Dec 2024):**
- Up to 4K resolution
- Camera control understanding (wide shot, POV, drone)
- Better physics and human expressions

**Veo 3 (May 2025):**
- Native audio generation (sound effects, ambient noise, dialogue)
- Excellent physics and realism
- Improved prompt adherence for complex action sequences

### 4. World Models: Beyond Video Generation

#### Concept
```
World Model: p(state | action, history)
```
- Simulate diverse environments and interactions
- Enable training of general embodied agents

#### Genie (Google DeepMind)

**Genie 1:** Latent action control
- Learn latent action space from video data
- Generate next frame conditioned on learned actions

**Genie 2:** Keyboard/mouse action control
- Control via standard input devices
- Autoregressive video diffusion (one latent frame at a time)
- **Pros:** Fast response for closed-loop learning
- **Cons:** Autoregressive architecture limits capacity

#### Interaction with Agents
- SIMA agent provides actions → Genie 2 simulates outcomes
- Enables testing agent behavior in simulated environments

### 5. Controllable Video Generation

#### Types of Action Control:

**Motion Control:**
- Select object and define trajectory path
- Precise movement specification

**Camera Control:**
- Define agent location and viewing direction
- Enables first-person/third-person perspectives

**Scene + Character Control:**
- Combine specific agent with specific environment
- Mix and match elements

#### Open Questions:
- Unified formulation for different control modalities?
- Unified conditioning mechanism?
- How to combine multiple controls effectively?

### 6. Multimodal Models

#### Why Multimodal Matters:
The real world is inherently multimodal (vision + language + audio + actions)

#### Interleaved Text+Image Generation
```
Auto-regressively generate: text token OR image
```

**Interpretation as World Model:**
- Text = action
- Image = state
- Models: p(state|action, history), p(action|state, history), p(state, action|history)
- Unifies world model + agent + agent with mental world model

#### Image Representation Trade-offs:

**Discrete Autoregressive:**
- ✅ Unified objective with text
- ✅ Simple hyperparameter space
- ❌ Lossy compression
- ❌ Slow, inflexible sampling

**Continuous Diffusion:**
- ✅ Almost lossless compression
- ✅ Flexible sampling (distillation, guidance, advanced solvers)
- ❌ Distinct objectives (mixing clean/noisy tokens)
- ❌ Many domain-specific hyperparameters

#### Recent Approaches:
- **Chameleon (Meta):** Discrete autoregressive
- **Show-o (ByteDance):** Discrete diffusion
- **Transfusion (Meta):** Predict next token + diffuse images
- **Emu3 (Meta):** Unified multimodal pretraining

### 7. 3D and 4D Generation

#### Why 3D Matters:
- Video shows projection of 3D world
- 90%+ pixels are static in 3D space
- Camera motion vs. scene dynamics can be disentangled
- Memory doesn't need to grow linearly with frames

#### Traditional 3D Reconstruction
```
100s of images + camera poses → NeRF → 3D scene
```
- **Issue:** Requires many captured views

#### Generation Approaches:

**DreamFusion: NeRF + Score Distillation**
```
Text → Diffusion Model Score → NeRF Optimization
```
- Generate 3D from text without 3D training data
- Slow optimization per scene

**Generate-Then-Reconstruct:**
```
Input → Multi-view Diffusion (5 sec) → NeRF Optimization (55 sec)
```
- Fine-tune 2D diffusion for multi-view consistency
- Generate multiple views, then reconstruct 3D
- Examples: CAT3D, CAT4D

#### 4D (3D + Time):
- Extend 3D generation to include temporal dynamics
- Enable real-time interactive 3D experiences
- Baked 4D scenes allow free exploration without per-frame model calls

#### Trade-offs:

**Baked 4D Scenes:**
- ✅ One-time cost, then "free" exploration
- ❌ Exploration only, can't interact

**Fast Video Models:**
- ✅ Interact with diverse actions
- ❌ Expensive per-action model call

**Future Direction:** Combine best of both worlds

---

## Key Innovations Timeline

### Image Generation
- **2015:** Sohl-Dickstein et al. - Original diffusion paper
- **2020:** Ho et al. (DDPM) - Simplified training objective
- **2021:** Song et al. - SDE framework
- **2021:** Dhariwal & Nichol - Classifier guidance
- **2021:** Ho & Salimans - Classifier-free guidance
- **2022:** Rombach et al. - Latent Diffusion (Stable Diffusion)
- **2023:** Peebles & Xie - Diffusion Transformers (DiT)

### Video Generation
- **2020:** Ho et al. - First video diffusion with spatial-temporal attention
- **2022:** Imagen Video - Cascaded framework
- **2024 Feb:** SORA - Unified transformer architecture
- **2024 May:** Veo 1 - Google's video model
- **2024 Dec:** Veo 2 - 4K, camera control, better physics
- **2025 May:** Veo 3 - Native audio, excellent realism

### World Models & 3D
- **2022:** DreamFusion - Text-to-3D via score distillation
- **2024:** Genie - Latent action world model
- **2024:** Genie 2 - Keyboard/mouse controllable world model
- **2024:** CAT3D/CAT4D - Generate-then-reconstruct for 3D/4D

---

## Practical Applications

### Current Use Cases:
1. **Content Creation:** Image/video generation for media, advertising, design
2. **Game Development:** Asset generation, environment creation
3. **Film/Animation:** Concept art, storyboarding, special effects
4. **Scientific Visualization:** Molecular structures, medical imaging
5. **Agent Training:** Simulated environments for robotics and autonomous systems

### Emerging Applications:
1. **Interactive Experiences:** Real-time 3D/4D world exploration
2. **Embodied AI:** World models for training general agents
3. **Multimodal Assistants:** Unified text+image understanding and generation
4. **Virtual/Augmented Reality:** Real-time scene generation and manipulation

---

## Open Challenges

### 1. Long-Term Consistency
- **Static Consistency:** Keep environment unchanged across long sequences
- **Dynamic Consistency:** Model off-screen events progressing realistically
- **Solution Directions:** Retrieval-based methods, stateful models, 3D-grounded representations

### 2. Real-Time Interaction
- **Challenge:** Current video models too slow for interactive experiences
- **Goal:** Generate next frame fast enough for real-time feedback
- **Importance:** Critical for training agents via reinforcement learning

### 3. 3D World Representation
- **Challenge:** Video models lack explicit 3D understanding
- **Observation:** Most pixel changes explainable by camera motion
- **Direction:** Disentangle camera motion from scene dynamics
- **Goal:** Memory that doesn't grow linearly with frame count

### 4. Unified Multimodal Framework
- **Challenge:** Different modalities use different representations
- **Question:** Discrete tokens vs. continuous embeddings?
- **Question:** Autoregressive vs. diffusion for images/video?
- **Goal:** Single model handling text, image, video, audio, actions

### 5. Scaling and Efficiency
- **Challenge:** Video/3D models extremely compute-intensive
- **Need:** Better architectures, compression, sampling methods
- **Trade-off:** Quality vs. speed vs. memory

---

## Key Papers by Topic

### Foundations:
- Sohl-Dickstein et al. (2015): "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models" (DDPM)
- Song et al. (2021): "Score-Based Generative Modeling through SDEs"

### Training & Guidance:
- Dhariwal & Nichol (2021): "Diffusion Models Beat GANs"
- Ho & Salimans (2021): "Classifier-Free Diffusion Guidance"
- Karras et al. (2022): "Elucidating the Design Space of Diffusion Models"

### Latent Diffusion:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
- Esser et al. (2021): "Taming Transformers for High-Resolution Image Synthesis"

### Flow Matching:
- Lipman et al. (2023): "Flow Matching for Generative Modeling"
- Albergo et al. (2023): "Stochastic Interpolants"

### Video:
- Ho et al. (2022): "Video Diffusion Models"
- Ho et al. (2022): "Imagen Video"
- OpenAI (2024): "SORA Technical Report"

### 3D/4D:
- Poole et al. (2022): "DreamFusion"
- Barron et al. (2023): "Zip-NeRF"
- Gao et al. (2024): "CAT3D"

### World Models:
- Bruce et al. (2024): "Genie"
- DeepMind (2024): "Genie 2"

---

## Resources

### Tutorial Websites:
- [CVPR 2022 Tutorial on Diffusion Models](https://cvpr2022-tutorial-diffusion-models.github.io/)
- [CVPR 2023 Tutorial](https://cvpr2023-tutorial-diffusion-models.github.io/)
- [NeurIPS 2023 LDM Tutorial](https://neurips2023-ldm-tutorial.github.io/)

### Blog Posts:
- [MLG Cambridge: Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- [Diffusion Meets Flow Matching](https://diffusionflow.github.io/)

### Code Implementations:
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [Diffusers Library](https://github.com/huggingface/diffusers)

---

## Conclusion

Diffusion models have revolutionized generative AI, progressing from:
- **Images** → High-quality, controllable image synthesis
- **Video** → Photorealistic, long-form video generation
- **3D/4D** → Interactive 3D worlds and dynamic scenes
- **World Models** → Simulated environments for agent training
- **Multimodal** → Unified models handling multiple modalities

The field continues to evolve rapidly with ongoing challenges in:
- Long-term consistency
- Real-time interaction
- 3D-aware representations
- Unified multimodal frameworks
- Computational efficiency

**The Future:** Integration of diffusion models with 3D representations, real-time interaction capabilities, and unified multimodal understanding promises to enable truly intelligent embodied agents that can understand, reason about, and interact with the world.

---

*This guide synthesizes content from CS-461 Foundation Models and Generative AI lectures by Karsten Kreis (NVIDIA) and Ruiqi Gao (Google DeepMind), Fall 2025.*
