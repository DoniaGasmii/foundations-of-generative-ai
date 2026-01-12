# Foundation Models, Reasoning & Decision-Making  

This lecture explores how **foundation models (FMs)** go beyond prediction to support **reasoning**, **planning**, and **human-aligned decision-making**, bridging predictive AI and agentic systems.

## Key Concepts

### 1. **Reasoning in FMs**
- Defined as *task-oriented, step-by-step justification* (not consciousness).
- Emerges via training on **reasoning traces** (e.g., “First I check colors… then cell shape…”).
- Datasets: GSM8K (math), PRONTOQA (logic), ETHICIST (ethics).

### 2. **How We Teach Reasoning**
- **Chain-of-Thought (CoT)**: Supervised fine-tuning with intermediate reasoning steps.
- **ReAct**: Combines *reasoning* + *acting*—models interleave thoughts (`"I should search…"`) with tool use (`Search("Eiffel Tower height")`).
- **RLHF**: Uses human feedback to reward high-quality reasoning via reinforcement learning (PPO + reward modeling).

### 3. **Human-AI Collaboration**
- **Defer-to-expert policies**: Learn when to let humans decide (e.g., in medicine).
- **Lab-in-the-loop**: AI generates drug candidates → ranks them → lab tests top picks → feedback improves next generation.
- **Interactive AGI**: Systems that plan, reflect, and adapt to user context, not just answer, but *assist*.

### 4. **Challenges & Insights**
- AI can improve decisions, but may also bias or mislead users (e.g., in tax reporting or clinical settings).
- Real-world evaluation requires **counterfactual reasoning** (what if AI wasn’t used?).
- Deployment must account for *selection bias* (e.g., AI used only on hardest cases).

## Takeaway
> Foundation models are evolving from **predictors** to **reasoners** and **collaborators**, but their real impact depends on *how they interact with people and real-world systems*.
