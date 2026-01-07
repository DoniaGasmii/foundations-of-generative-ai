# Foundation Models in Robotics  
---
Guest Lecture by **Dr. Hang Zhao**  
*Assistant Professor, Tsinghua University | Founder, Galaxea & MARS Lab*
*Embodied Intelligence: From Self-Driving to Humanoid Parkour*
---

##  Big Picture
**We don’t have native robotic foundation models yet**. Instead, we **combine vision, language, and control models** into hybrid systems that handle **spatial, athletic, and reasoning intelligence**, enabling robots to drive, manipulate, and parkour.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/c86436fc-4f5d-4c95-a3da-359ea548c4e0" />

>  *“Embodied AI = Spatial + Athletic + Reasoning Intelligence”*

---

## 🚗 Self-Driving: Dual-System Design 

###  End-to-end fails at L4 autonomy (no human driver) → can’t handle long-tail cases/black box is not reliable or controlable.
###  Solution: **DriveVLM-Dual**
- **System 1 (Fast)**: Handles 95% of routine driving (traditional or end-to-end policy).
- **System 2 (Slow)**: VLM kicks in for complex/long-tail scenarios (e.g., construction zones, accidents).

> *Like human drivers: autopilot most of the time, but pause to think when things get weird.*

---

## DriveAgent-R1: Active Perception & Multi-Turn Reasoning 

When visual info is uncertain, the VLM **requests tools**:
-  **RoI Inspection**: Zoom in on traffic lights/signs
-  **Depth Estimation**: Understand distances
-  **3D Detection**: Find novel objects (e.g., “fallen tree”)
-  **Memory Pool**: Look back 5 seconds

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/05166696-7bff-4c96-8ad5-ea515f3f8ef3" />

>  *Multi-turn reasoning (M\*_tool\*) > Single-shot (M\*_text\*) for real-world robustness.*

---

## 🤖 Manipulation: Galaxea G0 — Dual-System VLA 

- **Fast System (G0-VLA, 3B)**: Real-time motion planning (200 Hz)
- **Slow System (G0-VLM, 30B)**: Task planning & HRI (2 Hz)
- Trained on **Galaxea Open-World Dataset** (500 hrs, 200 tasks, 50 scenes): **#1 most-downloaded robot dataset**
- 
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/f7d639d8-5f8a-4898-8a13-51633fa5fce0" />

>  *Example: “Help me make the bed” → VLM breaks into subtasks → VLA executes motions.*

---

##  Perceptive Locomotion: Humanoid Parkour via RL 

### Key Innovations:
1. **Soft Dynamics Constraints** → penalize (don’t forbid) collisions → lets robots learn from falls.
2. **Curriculum Learning** → start easy (flat ground), then add hurdles, slopes, jumps.
3. **Real2Sim2Real** → scan real scenes → train in sim → deploy zero-shot on real robot.
4. 
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/db2e207f-9c9f-49f7-a8cb-94340792e760" />

>  *Result: Humanoids that run, jump, fall, and get up: using whole-body control, not just legs.*

---

##  Resources
- **Papers**: [DriveVLM (CoRL 2024)](https://arxiv.org/abs/2406.12345), [DriveAgent-R1 (arXiv 2025)](https://arxiv.org/abs/2507.20879), [Galaxea Open-World](https://huggingface.co/datasets/Galaxea)
- **Lab**: [MARS Lab, Tsinghua](https://www.mars-lab.org/)
- **Dataset**: [Galaxea Open-World Dataset](https://huggingface.co/datasets/Galaxea) (75k+ downloads)

---

##  Key Takeaway
> **Robots need more than perception: they need spatial awareness, athletic control, and high-level reasoning. Current systems are hybrid, but they’re getting smarter, faster, and more embodied every day.**


