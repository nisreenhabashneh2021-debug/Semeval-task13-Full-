SemEval-2026 Task 13 — Full System (Subtasks A, B, C)

This repository contains our **complete end-to-end system** for **SemEval-2026 Task 13: Detecting Machine-Generated Code**.  
The solution covers **all subtasks**:

- **Subtask A — Binary Code Generation Detection**  
  Classify code as *Human-written* or *Machine-generated*.

- **Subtask B — Multi-Class Authorship Detection**  
  Identify whether the code was written by a human or one of **10 LLM families**.

- **Subtask C — Hybrid & Adversarial Code Detection**  
  Predict whether a snippet is *Human*, *Machine*, *Hybrid*, or *Adversarial*.

Our system includes:
- simple baselines
- Transformer models
- ensemble models
- GPU-optimized training loops  
- Logging, metrics, and plotting utilities  
- Fully modular folder structure for scalable experimentation  


```
<!-- ==========================  DIAGRAM CONTAINER  =========================== -->
<div style="font-family:Arial; padding:20px;">

<!-- ==========================  TITLE  ======================================= -->
<h2 style="text-align:center; color:#1E90FF;">SemEval-2026 Task 13 – Label Flow Diagram</h2>

<!-- ==========================  FLEX CONTAINER =============================== -->
<div style="display:flex; gap:40px; justify-content:center; margin-top:30px;">

<!-- ==========================  TASK A  ====================================== -->
<div style="width:280px; background:#0A2540; padding:20px; border-radius:15px; color:white;">
    <h3 style="text-align:center; color:#4DA3FF;">Task A</h3>
    <p style="background:#4DA3FF; padding:10px; border-radius:8px; text-align:center;">
        Sexist
    </p>
    <p style="background:#B0B0B0; padding:10px; border-radius:8px; text-align:center;">
        Not Sexist
    </p>
</div>

<!-- ==========================  TASK B  ====================================== -->
<div style="width:300px; background:#102C57; padding:20px; border-radius:15px; color:white;">
    <h3 style="text-align:center; color:#FFB84D;">Task B</h3>

    <p style="background:#FFB84D; padding:10px; border-radius:8px; margin-bottom:10px;">Threats</p>
    <p style="background:#54D1C0; padding:10px; border-radius:8px; margin-bottom:10px;">Derogation</p>
    <p style="background:#B57BFF; padding:10px; border-radius:8px; margin-bottom:10px;">Animosity</p>
    <p style="background:#FF6F61; padding:10px; border-radius:8px; margin-bottom:10px;">Prejudiced Discussion</p>
</div>

<!-- ==========================  TASK C  ====================================== -->
<div style="width:420px; background:#19376D; padding:20px; border-radius:15px; color:white;">
    <h3 style="text-align:center; color:#FFCDEA;">Task C</h3>

    <p style="background:#FFB084; padding:10px; border-radius:8px;">Threats of harm</p>
    <p style="background:#FFD19A; padding:10px; border-radius:8px;">Incitement & encouragement of harm</p>

    <p style="background:#CCF6FF; padding:10px; border-radius:8px;">Descriptive attacks</p>
    <p style="background:#A0E9FF; padding:10px; border-radius:8px;">Aggressive & emotive attacks</p>
    <p style="background:#B6F7D1; padding:10px; border-radius:8px;">Dehumanisation & sexual objectification</p>

    <p style="background:#D4C1F2; padding:10px; border-radius:8px;">Gendered slurs, insults</p>
    <p style="background:#E0D4FF; padding:10px; border-radius:8px;">Immutable gender stereotypes</p>
    <p style="background:#C9B8FF; padding:10px; border-radius:8px;">Backhanded gendered compliments</p>
    <p style="background:#EAD9FF; padding:10px; border-radius:8px;">Condescending explanations</p>

    <p style="background:#FFB7B2; padding:10px; border-radius:8px;">Mistreatment of individual women</p>
    <p style="background:#FF9F9F; padding:10px; border-radius:8px;">Systemic discrimination</p>
</div>

</div>
</div>

```










