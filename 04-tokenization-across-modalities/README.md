# building blocks and design principles of foundation models

---

### Core Focus  
This lecture introduces the **building blocks and design principles of foundation models**, laying the groundwork for understanding modern architectures in language, vision, biology, and beyond.

---

### Key Concepts

#### 1. **Tokenization**  
- Converts raw input (text, image, DNA, etc.) into discrete or continuous tokens.  
- **Text**: Subword tokenizers like BPE (GPT) and WordPiece (BERT).  
- **Vision**: Fixed patches (ViT) or learned discrete codes (VQ-VAE → used in DALL·E).  
- **Biology**: Tokens at molecular (DNA bases), cellular (gene expression), and multicellular scales.  
- Goal: Create a **universal interface** across modalities.

#### 2. **Positional Encodings**  
- Inject sequence order since Transformers are permutation-invariant.  
- **Sinusoidal** (fixed, extrapolatable) vs. **Learned** (flexible but limited length) vs. **Rotary (RoPE)** (relative-position aware, norm-preserving).

#### 3. **Transformer Building Blocks**  
Every foundation model uses these 7 components:  
1. Tokenization  
2. Positional/structural encoding  
3. Linear projections (e.g., Q/K/V)  
4. **Context mixing** → primarily **attention**  
5. Nonlinearities (GeLU, SiLU)  
6. Normalization & stabilizers (LayerNorm, dropout)  
7. Residual connections  

#### 4. **Attention Mechanisms**  
- **Self-attention**: Tokens interact within the same modality.  
- **Cross-attention**: Links different modalities (e.g., text → image).  
- **Multi-head**: Parallel attention heads capture diverse relationships.  
- Efficient variants: FlashAttention, sparse attention, linear attention.

#### 5. **Architecture Patterns**  
- **Encoder-only** (e.g., BERT, ESM): for representation learning.  
- **Decoder-only** (e.g., GPT): for autoregressive generation.  
- **Encoder-decoder** (e.g., original Transformer): for conditional tasks like translation.

#### 6. **Beyond Standard Transformers**  
- **Protein FMs**: ESM3 (multimodal tokens + geometry-aware attention), AlphaFold (Evoformer/Pairformer with triangle updates).  
- **DNA modeling**: Replaces attention with convolutions/recurrence for long sequences.  
- **Limitations**: Quadratic cost, static reasoning, tokenization bottlenecks.
