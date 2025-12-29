# In-Context Learning & Emergent Behavior - Adaptation, Fine-Tuning &Test-Time Training

## Introduction:

This lecture is split into two parts:

- **Part I: In-Context Learning and Emergent Behavior**  
Explores how foundation models solve new tasks without weight updates, using only prompts (zero-shot/few-shot) or frozen representations. Covers:
  - What in-context learning (ICL) is and why it’s surprising
  - Zero-shot vs. few-shot generalization
  - Frozen-backbone adaptation (e.g., linear probing, ABMIL)
  - Emergent abilities that appear only at scale

- **Part II: Adaptation, Fine-Tuning, and Test-Time Training**
  
<!-- Part II Header -->
<h3>Part II: Adaptation, Fine-Tuning, and Test-Time Training</h3>

<table>
  <tr>
    <td style="vertical-align: top; padding-right: 20px; width: 60%;">
      Investigates how models can <em>learn during inference</em> by maintaining parametric memory. Topics include:
      <ul>
        <li>The KV cache memory bottleneck in transformers</li>
        <li>Self-attention as non-parametric kernel regression</li>
        <li>Test-time training (TTT) with parametric memory (e.g., linear attention)</li>
        <li>Active retrieval of useful examples at test time (e.g., SIFT)</li>
        <li>Reinforcement learning at test time (e.g., TTC-RL, RL²)</li>
      </ul>
    </td>
    <td style="vertical-align: top; text-align: center;">
      <img src="https://github.com/user-attachments/assets/14aadbf9-e5fa-4e27-80cb-0c9e82bf45b7" alt="Prof. Andreas Krause & Dr. Jonas Hübotter" width="350" height="250" style="border-radius: 8px;">
      <br>
      <small><strong>Andreas Krause</strong><br>ETHZ</small>
      &nbsp;&nbsp;&nbsp;&nbsp;
      <small><strong>Jonas Hübotter</strong><br>ETHZ</small>
    </td>
  </tr>
</table>





