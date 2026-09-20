# Peer Review: Symbolic Knowledge Distillation & Disposition Layer

**Document Reviewed:** `docs/intake/opportunity.md`  
**Verdict:** **HOLD** (Fail closed for Spec Writer until kill criteria become verified artifacts)  
**Methodology:** Paine Triad (Argument / Counter / Rebuttal)

---

## 1. Argument (Strongest Case)

*Why does this opportunity belong in `Distilled_Agents` now?*

1. **Addresses the Core Failure Mode of Small Model Distillation:** Small parameter models (0.5B–3B, such as Qwen2.5-Instruct) distilled solely via behavior cloning on teacher trajectories suffer tail degradation, hallucination, and inability to handle combinatorial or strict logic constraints. By introducing symbolic knowledge distillation paired with a deterministic harness disposition layer, we allow the small student to propose candidate solutions while delegating verification and constraint resolution to a deterministic component.
2. **Strictly Aligns with ADR 0006 Architectural Contracts:** ADR 0006 formalizes the agent loop as $G = (H, R_\delta)$, where policy $R_\delta$ proposes actions and harness $H$ disposes them via a frozen tool registry and AST/JSON dispatch without in-process `exec`. Adding a symbolic solver tool (`sqe_constraint_solver`) directly fits this contract without modifying the runtime architecture or breaching the sandbox security boundary.
3. **Preserves Enterprise Licensing and Offline Posture:** Distilling verified symbolic rationales from local open-weight teachers (Apache-2.0 / MIT) directly preserves the repo's clean MIT license profile, while keeping execution within local trajectory BC workflows without dependencies on proprietary cloud APIs.

---

## 2. Counter (Strongest Objections & Vulnerabilities)

*Where does the draft proposal overclaim, introduce false friends, or risk project distraction?*

1. **False Claim on Current Evaluation Maturity:** The initial draft asserted that the repo "evaluates with Pass@K AQA." In reality, `make aqa-gate` (`scripts/harness/run_aqa_gate.py`) is merely a single-pass `pass_rate` threshold run over a 4-row golden set (`configs/golden_sets/core_sdlc.jsonl`). True Pass@K ($k \ge 3$) on a hard or out-of-distribution (OOD) benchmark does not exist in the repo today; claiming it as an existing capability masks a major evaluation deficit.
2. **False Claims Regarding ADR 0006 Disposition:** ADR 0006 establishes harness disposition, but today's disposition is strictly AST allowlisting and JSON schema validation. It offers zero semantic or symbolic soundness. Existing registry tools (`pytest_runner`, `sqe_checklist`) are shallow string checklists (`pytest_runner` does not execute pytest; `sqe_checklist` checks for the substring `"assert"`). Calling today's disposition "symbolic" is an overstatement.
3. **Overreach into External Initiatives (Mango INV-16 / Edge-AI VII):** The initial intake conflated local trajectory BC on Qwen2.5 with broader organizational initiatives like Mango INV-16 and Edge-AI VII delivery. Distilled_Agents must remain strictly scoped to its core repository boundary.
4. **Paper False Friends & Deferred Mechanisms:** Research literature (Kang et al., DualDistill, SCoRe) relies on execution sandboxes, live code execution, and online policy optimization. In this repo:
   - CodeAct sandbox execution is deferred (tools do not `exec`).
   - TRL GKD and SDPO are deferred pending library version pins.
   - DualDistill must be same-task $x$, reference answer $a$, and two heterogeneous teachers with transition stitching—not arbitrary role mixing.
   - Self-Agreement (SAG) is majority voting over parsed strings, not observation-based execution consistency.
5. **Unmeasurable Constraint Satisfaction Metric:** Success metric #2 originally stated "Constraint satisfaction rate = 100%" without specifying a domain, problem vertical, or tool implementation, making it unfalsifiable.

---

## 3. Rebuttal & Fail-Closed Gating (Surviving Position)

*What survives peer critique, what was cut, and how is the project guarded against scope creep?*

1. **Demoted "Neurosymbolic Net" to "Symbolic KD + Disposition Tool":** All language suggesting a hybrid neurosymbolic neural network architecture has been excised. The student remains a standard LoRA causal language model proposer; the symbolic component is a deterministic tool inside harness $H$.
2. **Scope Restricted Strictly to Distilled_Agents:** All references to Mango INV-16 and Edge-AI VII are eliminated. "Edge/offline" is explicitly bounded to local trajectory BC on Qwen2.5 0.5B–3B.
3. **Hard-Slice Pass@K and Tool ID Formally Grounded:**
   - Pass@K is re-classified as a **gate to build**, requiring an extended golden set and exact-match grading (`answers_match`). Semantic matching is explicitly disqualified from counting as success on the hard slice.
   - The constraint satisfaction metric is pinned to a specific vertical (Software Quality Acceptance) and a named tool ID (`sqe_constraint_solver`).
4. **Explicit Kill Criteria Enacted (HOLD for Spec Writer):**
   Spec Writer hand-off is blocked until the following artifacts exist:
   - Automated CI Pass@K gate with hard/OOD slice and exact-match evaluation.
   - Critic rejection telemetry logging explicit reason codes in collect/compose scripts.
   - Proven fail-closed behavior terminating OOD tasks with `BLOCKED`/`tool_error` with zero synthetic success.
