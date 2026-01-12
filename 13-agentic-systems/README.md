# Foundation Models and Agentic Systems

This lecture explores how foundation models evolve from passive generators into **active AI agents**—systems that **perceive, reason, act, and learn** over time in dynamic environments.

### Core Concepts

- **Agents vs. LLMs**:  
  Traditional LLMs are “closed-book”: they read input → generate output → forget.  
  **Agents** operate in a continuous loop: **observe → reason → act → update → repeat**, enabling long-horizon, goal-directed behavior.

- **7 Capabilities of Autonomous Agents**:
  1. **Perception** (text, images, APIs, sensors)  
  2. **Memory** (short- & long-term state)  
  3. **Goal & constraint representation**  
  4. **Reasoning & planning**  
  5. **Action execution** (via tools or actuators)  
  6. **Learning & adaptation**  
  7. **Self-monitoring** (detect errors, ask for help)

- **ReAct Framework**:  
  Interleaves *thoughts*, *actions*, and *observations* to reduce hallucination and support grounded, multi-step reasoning.

- **Tool Use**:  
  Agents call external functions (e.g., search, code exec, domain models) via **structured schemas** to perform reliable actions beyond language generation.

### Agent Infrastructure

- **Tool Libraries**: Curated functions with clear input/output schemas.
- **Model Context Protocol (MCP)**: Standardizes tool discovery, invocation, and results—like a universal API for agents.
- **Agent SDKs**: Provide the runtime loop, memory management, retries, logging, and orchestration (e.g., OpenAI/Anthropic Agent SDKs).

### Real-World Examples

- **Agentic-Tx**: Drug discovery agent using specialized models for toxicity, clinical data, and molecule design.
- **Molecular Tumor Boards**: Multimodal agents integrating pathology, genomics, and literature to support oncology decisions.
- **SIMA**: Embodied agent in video games that follows natural-language instructions (e.g., “Cook the burger”) by seeing pixels and taking actions.
- **Voyager**: Self-improving Minecraft agent that learns new skills indefinitely without catastrophic forgetting.

### Outlook (2030+)

- AI as collaborative **research partners** in math, biology, and climate science.
- **Self-improving systems** that refine their reasoning, tools, and world models over time.
- Foundation models embedded in scientific workflows—from drug design to weather prediction.

---

> **Key Takeaway**: The future isn’t just smarter models—it’s **smarter systems** that act purposefully in the world, grounded in tools, memory, and feedback.

