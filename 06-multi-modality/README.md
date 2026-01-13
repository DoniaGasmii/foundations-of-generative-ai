# Multimodality in Foundation Models

This lecture introduces how foundation models combine **multiple data modalities** (e.g., images + text) to enable richer understanding and generation. We focus on **vision-language models**, but the principles extend to biology, medicine, audio, and more.

## Key Concepts

### 1. **Why Multimodality?**
- **Complementarity**: Different modalities provide unique views (e.g., image shows *what*, text explains *why*).
- **Disambiguation & Generalization**: Joint learning reduces uncertainty and improves transfer.

### 2. **The Design Triangle**
Building multimodal models involves three interdependent choices:
- **Fusion timing**: *When* do modalities interact?
- **Fusion operation**: *How* are they combined?
- **Learning objective**: *Why* are they fused? (e.g., contrastive, autoregressive, masking)

### 3. **Fusion Strategies**
| Type | Description | Examples |
|------|-------------|----------|
| **Early Fusion** | Modalities merged at input → single shared model | **ViLT** |
| **Late Fusion** | Separate encoders → aligned via loss/output | **CLIP** |
| **Middle Fusion** | Pretrained encoders + fusion in intermediate layers | **LLaVA**, **Flamingo** |

<img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/845c1323-ec5c-4a97-aca5-7635751788c4" />

### 4. **Key Architectures**
- **CLIP**: Contrastive pretraining aligns image/text embeddings → enables **zero-shot classification**.
- **ViLT**: Treats image patches + word tokens as one sequence → full cross-attention from the start.
- **LLaVA**: Connects CLIP vision features to an LLM via a **projection layer**, then instruction-tunes for dialogue.
- **Flamingo**: Uses **Perceiver Resampler** + **gated cross-attention** to inject visual tokens into a frozen LLM.

### 5. **Fusion Operations**
- **Linear**: concat, sum, max  
- **Multiplicative**: gating, element-wise product  
- **Attention-based**: cross-attention, co-attention  
- **Bilinear**: captures pairwise interactions

### 6. **Beyond Vision-Language**
- **Biology**: DNA, RNA, protein, imaging → “AI Virtual Cell” with universal representations across scales.
- **Medicine**: Integrates EHRs, imaging, omics, signals for grounded clinical reasoning.

---

>  **Takeaway**: Multimodal foundation models don’t just *add* modalities, they **strategically align** them using scalable, modular designs that balance performance, efficiency, and flexibility.
