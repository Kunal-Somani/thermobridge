# Project 2 — Finalized Specification & Roadmap

> **Working title (provisional):** *ThermoBridge: A Thermodynamic Brownian-Bridge Diffusion Framework for Bidirectional 3D MR↔CT Synthesis with Edge-Preserving Anisotropic Operators*
>
> **Status:** Pre-implementation. This document is the frozen agreement on scope, method, and success criteria. Everything below is what we said we would build; future progress is measured against it.
>
> **Author:** Solo (single author).
> **Context:** Second project under the same professor; follows the SPL IEEE dual-branch MRI reconstruction paper.
> **Target venue (primary):** MICCAI. **Secondary / fallback:** MIDL, IEEE TMI, Medical Image Analysis. (Not CVPR/NeurIPS — domain mismatch; see §11.)
> **Last updated:** at creation.

---

## 1. One-sentence pitch

A thermodynamically grounded Brownian-bridge diffusion process that transports directly between paired MRI and CT volumes in **both directions**, using a **hybrid transformer denoiser** whose local mixer is a novel **learnable 3D anisotropic-diffusion operator** (edge-preserving by construction), evaluated **cross-organ** on brain and pelvis, with a **deep bone-boundary failure analysis**.

## 2. Clinical motivation (the "why this matters" for the intro)

Synthetic CT (sCT) generated from MRI lets radiotherapy and surgical-planning workflows obtain CT-equivalent electron-density / bone information **without exposing the patient to ionizing X-ray radiation**, and enables MR-only treatment planning. The inverse direction (CT→MR) supports soft-tissue visualization and retrospective multimodal registration when only CT exists. A single model that does **both directions** across **multiple anatomies** is operationally valuable: one deployable artifact instead of a zoo of organ- and direction-specific models. This is the same "universal single-model deployment" thesis as Project 1, now applied to cross-modality translation instead of reconstruction.

## 3. Core hypotheses / research questions

- **RQ1 (process):** Does a *bridge* formulation (direct distribution-to-distribution transport between MR and CT) outperform standard noise-conditioned diffusion for paired cross-modality synthesis, and does it make bidirectionality native rather than two separate models?
- **RQ2 (operator):** Does replacing standard 3D convolution with a learnable anisotropic-diffusion operator as the *local* mixer reduce error specifically at tissue **boundaries** (esp. bone interfaces) without hurting bulk-tissue accuracy?
- **RQ3 (generalization):** Does a single bidirectional model trained jointly on brain + pelvis match or beat organ-specific models — and which anatomy benefits from sharing? (Direct sequel to Project 1's Table IV finding that the data-poor anatomy benefits from joint training.)
- **RQ4 (failure):** Where does MR↔CT synthesis fail, quantitatively, and is the dominant residual error concentrated at bone–soft-tissue boundaries as hypothesized?

## 4. Novelty statement (honest positioning — read this before writing related work)

The novelty is **concentrated in two places**, not scattered across six. Everything else is standard-but-well-used. State this honestly; do not overclaim.

**Novel contribution A — Bidirectional thermodynamic bridge for cross-modality medical synthesis.**
- *What is borrowed:* The Brownian-bridge diffusion formulation for image-to-image translation (BBDM, Li et al., CVPR 2023) and, conceptually, image-to-image Schrödinger bridges (I2SB, ICML 2023). Heat/diffusion-as-generation lineage (Inverse Heat Dissipation, ICLR 2023; Cold Diffusion, 2022).
- *What is new:* Adapting a bridge process to **3D, paired, bidirectional MR↔CT** with a **single shared model** conditioned on direction; framing the variance/transport schedule in explicitly thermodynamic terms; and (if upgraded) the Schrödinger-bridge entropic-OT formulation tied to statistical thermodynamics.

**Novel contribution B — Learnable 3D anisotropic-diffusion operator as a transformer's local mixer.**
- *What is borrowed:* Perona–Malik anisotropic diffusion (1990) and Trainable Nonlinear Reaction Diffusion (TNRD, Chen & Pock, TPAMI ~2016); Diffusion-Transformer denoisers (DiT, ICCV 2023); global/local hybrid mixing (MetaFormer-style reasoning).
- *What is new:* A 3D, learnable, edge-stopping anisotropic-diffusion update used as the **local token mixer inside a transformer denoiser**, with learnable conductance — designed so the *forward* corruption is isotropic (destroys boundaries) and the *operator* is anisotropic (restores them edge-aware). The forward process, the operator, and the boundary failure analysis are one unified story.

**Explicitly NOT claimed as novel:** transformers, 3D processing, patch-based training, cross-organ evaluation. These satisfy the professor's required ingredients but are used as standard machinery.

## 5. Method overview (spec-level; full math pinned at implementation)

### 5.1 The Brownian-bridge diffusion process
Let the two paired domains be source `x_0` and target `x_T` (e.g., MR and CT for one direction). A Brownian bridge interpolates between fixed endpoints with variance peaking mid-trajectory. Marginal (spec form):

```
q(x_t | x_0, x_T) = N( (1 - m_t)·x_0 + m_t·x_T ,  δ_t · I )
m_t = t / T
δ_t = 2·s·(m_t - m_t²)      # variance peaks at t = T/2; s = max-variance hyperparameter
```

The network learns the reverse transition; at sampling we start from the *actual source image* `x_T` (not pure noise) and walk back to `x_0`. **Bidirectionality:** the same model is conditioned on a direction token `d ∈ {MR→CT, CT→MR}`; for each direction the (source, target) roles swap. Exact schedule, objective (noise- vs `x_0`- vs velocity-prediction), and step count are pinned during implementation and reported.

### 5.2 Hybrid transformer denoiser
- **Patchify** the 3D patch into tokens (DiT-style 3D patch embedding).
- **Global mixer:** multi-head self-attention over tokens (the "transformer" requirement, used where it's cheap and useful).
- **Local mixer:** the novel 3D anisotropic-diffusion operator (§5.3), operating on the spatial representation to preserve boundaries — replaces / augments the standard MLP/conv local path.
- **Timestep conditioning:** adaptive layer norm (adaLN), DiT-style.
- **Direction conditioning:** learned embedding for `d`, injected alongside the timestep embedding.
- **Source conditioning:** the source volume is available throughout (bridge endpoint), so conditioning is via the bridge formulation itself plus channel concatenation; cross-attention optional and ablated only if needed.

### 5.3 The learnable 3D anisotropic-diffusion operator
Discretized Perona–Malik-style update, made trainable in 3D:

```
∂I/∂t = div( g(|∇I|) · ∇I )
g(s)  = 1 / (1 + (s/K)²)        # edge-stopping conductance; K learnable (optionally per-channel)
```

Implemented as one or a few explicit Euler steps inside a residual block: compute 3D spatial gradients, apply learnable edge-stopping conductance, divergence, update. Learnable parameters: conductance `K`, step size, optionally a small MLP for `g`. Cite TNRD; the novelty is the 3D + in-transformer + bridge-denoiser integration, not the idea of trainable diffusion.

### 5.4 3D patch pipeline
Train on 3D patches (e.g., 96³ or 128³ — pinned after VRAM profiling on the allocated GPU); sliding-window inference with overlap-blending to reconstruct full volumes; evaluate on full volumes within the patient body mask.

## 6. How all six professor ingredients are covered

| Ingredient | How it appears | Treated as novelty? |
|---|---|---|
| Novel convolutional operator | 3D learnable anisotropic-diffusion local mixer | **Yes (B)** |
| Transformer based | Hybrid transformer denoiser (global attention) | No — required machinery |
| 3D dataset | Volumetric MR/CT, 3D denoiser | No — substrate |
| Patch based | 3D patch training + sliding-window inference | No — substrate |
| Cross modality / cross organ | Bidirectional MR↔CT, brain + pelvis, joint model | Partly (evaluation/story) |
| Physics + diffusion (thermodynamics) | Brownian/Schrödinger bridge transport | **Yes (A)** |

## 7. Dataset & preprocessing

- **Primary:** SynthRAD2023 — paired, deformably registered MR & CT, 3D, two anatomies (**brain + pelvis**). Satisfies cross-modality and cross-organ from a single public source. (Stanford 67 GB set discarded as redundant unless it adds paired MR–CT we lack.)
- **Known caveat to address honestly:** MR/CT pairs are *deformably registered*, so residual misregistration exists — concentrated at high-gradient edges (bone, air interfaces). This is both a real error source and part of the failure story; acknowledge it, don't hide it.
- **Preprocessing:** body/patient mask extraction; CT in **Hounsfield Units (HU)**; consistent intensity handling for MR (per-volume normalization); resampling to common spacing; patching with foreground sampling. CT clipping range and MR normalization scheme pinned at implementation and reported exactly (Project-1 lesson: the paper must match the code).
- **Access verification:** SynthRAD portal access terms and exact case counts to be re-checked at data-setup time (challenge portals change).

## 8. Tech stack

- **Framework:** PyTorch + PyTorch Lightning (continuity with Project 1).
- **Medical 3D toolkit:** MONAI — for 3D transforms, patch samplers, sliding-window inference, and metrics (do not hand-roll what MONAI already provides correctly).
- **Mixed precision:** yes (with FP32-safe handling of any FFT/gradient ops, as in Project 1).
- **Logging:** CSV + (recommended) Weights & Biases or TensorBoard for the long multi-run campaign.
- **FLOPs/params:** use a real counter (fvcore / ptflops / thop) — Project 1 reported GFLOPs with no code that computed them; do not repeat that.
- **Hardware:** allocated cloud GPU (≥24 GB) for training; local HP Omen / RTX 4060 (8 GB) for dev/debug only.
- **Reproducibility:** fixed seeds, pinned dependency versions, deterministic eval masks (Project 1's validation masks were not reproducibly seeded under multi-worker loading — fix this).

## 9. Evaluation protocol

- **Headline metric:** **MAE in HU within the body mask** (the field-standard sCT metric) for MR→CT. Lead with this, not SSIM.
- **Secondary:** PSNR, SSIM (in-mask), and per-direction reporting for CT→MR (where HU is not the unit — use MAE on normalized intensity + PSNR/SSIM).
- **Per-organ breakdown:** brain vs pelvis, separately, every table.
- **Statistical rigor:** report **mean ± std across cases** and a significance test (e.g., paired Wilcoxon) for every headline comparison. No single "peak" numbers in comparison tables (the central Project-1 mistake).
- **Clinical stretch goal:** dose / gamma-pass analysis for the sCT (feasibility to be assessed; if not done, say so explicitly rather than implying it).
- **Baselines:** recent (2023–2025) MR↔CT methods — GAN-based (e.g., CycleGAN/Pix2Pix-family for medical), recent diffusion-based synthesis, and SynthRAD2023 challenge reference methods. **To be surveyed via literature search at the baseline stage** — do not reuse 2022 baselines.

## 10. Ablation matrix (designed to NOT explode)

Each contribution gets exactly one clean on/off comparison against the full model; we do not run the full cross-product.

| # | Comparison | Isolates | Expected story |
|---|---|---|---|
| A1 | Bridge vs standard conditional diffusion (same denoiser) | Contribution A (process) | Bridge ≥ noise-diffusion, faster sampling |
| A2 | Bidirectional shared model vs two one-directional models | Bidirectional design | Shared ≈ or > separate; one artifact |
| B1 | Anisotropic operator vs standard 3D conv (same backbone) | Contribution B (operator) | Lower boundary error, similar bulk error |
| B2 | Operator on/off, boundary-region MAE only | Operator's *specific* effect | The key plot of the paper |
| C1 | Joint brain+pelvis vs organ-specific | Generalization (RQ3) | Data-poor organ benefits |
| D1 | (Optional) Schrödinger vs Brownian bridge | Upgrade payoff | Only if time allows |

## 11. Venue strategy & honest bar

- **MICCAI** is the ambitious-but-correct primary target: it is the top medical-imaging conference, its reviewers value MAE-HU/clinical metrics and boundary analysis, and MR→CT synthesis lives there.
- **CVPR/NeurIPS are the wrong room** for this contribution (domain mismatch; generalist reviewers won't credit the clinical depth, and competition is against large teams).
- **The real acceptance bar** (state plainly): a genuinely novel core (we have two), recent SOTA baselines, statistical significance with error bars, clean isolating ablations, honest failure analysis, and reproducibility. This is a substantial step up from a 6-page SPL paper. Treat the gap as rigor, not component-count.

## 12. Risk register (eyes open)

| Risk | Severity | Mitigation |
|---|---|---|
| Bridge fails to converge / unstable training | High | Start with Brownian (simpler) before Schrödinger; small-scale sanity run first; well-tested schedule |
| 3D bridge sampling over 2 organs × 2 directions = heavy eval | Medium | Budget eval compute; cap sampling steps; subset for development, full set for final |
| Ablation matrix explodes | Medium | Fixed matrix in §10; no cross-products |
| Baselines outdated → desk reject | High | Mandatory literature search at baseline stage; 2023–2025 only |
| Overclaiming / paper ≠ code (Project-1 relapse) | High | Every equation, loss term, and number must match the released code; no peak-vs-mean mixing |
| Misregistration in data misread as model failure | Medium | Quantify registration error; report it as a confound, not hide it |
| Scope creep toward "use all 6 as novelties" | High | §4 novelty freeze + §13 anti-goals |

## 13. Anti-goals (what we will deliberately NOT do)

- Will **not** make the transformer a claimed novelty.
- Will **not** add undocumented auxiliary loss terms (Project 1 had TV + frequency losses absent from the paper).
- Will **not** report peak single-slice/single-case metrics in comparison tables.
- Will **not** add a fourth or fifth "novel" component to seem more impressive.
- Will **not** target a generalist CV venue for a clinical-synthesis paper.

## 14. Roadmap & milestones (progress checklist)

> Use this section to track "how much is achieved." Check items as completed.

**Phase 0 — Setup & data**
- [ ] Cloud GPU allocated and profiled (max patch size at target batch size determined)
- [ ] SynthRAD2023 downloaded, access terms verified
- [ ] Preprocessing pipeline: masking, HU handling, MR normalization, resampling, patch sampler
- [ ] Data sanity visualizations (paired MR/CT slices, mask overlays) at ≥300 DPI
- [ ] Reproducible, deterministic validation split + eval masks

**Phase 1 — Baselines & literature**
- [ ] Literature search: 2023–2025 MR↔CT SOTA + SynthRAD reference methods
- [ ] At least one strong recent baseline reproduced or fairly cited with matched protocol
- [ ] Evaluation harness: MAE-HU-in-mask, PSNR, SSIM, per-organ, per-direction, mean±std + significance test

**Phase 2 — Core method (Brownian bridge)**
- [ ] Brownian-bridge diffusion implemented (forward schedule + reverse training objective)
- [ ] Hybrid transformer denoiser (global attention + adaLN timestep + direction embedding)
- [ ] Bidirectional training (single shared model, both directions)
- [ ] First full MR→CT and CT→MR volumes synthesized end-to-end
- [ ] Sanity metrics beat trivial baseline (identity / mean / simple regressor)

**Phase 3 — Novel operator**
- [ ] 3D learnable anisotropic-diffusion operator implemented and unit-tested (gradient flow, edge-stopping behavior)
- [ ] Operator integrated as local mixer in the denoiser
- [ ] Ablation B1/B2: operator vs standard conv, with boundary-region MAE

**Phase 4 — Experiments & ablations**
- [ ] A1, A2, B1, B2, C1 completed with mean±std + significance
- [ ] Cross-organ generalization study (RQ3)
- [ ] Robustness checks as appropriate

**Phase 5 — Deep failure / boundary analysis**
- [ ] Bone-boundary HU error quantified (brain skull, pelvis bone)
- [ ] Boundary error maps and distance-to-boundary error profiles at ≥300 DPI
- [ ] Honest discussion incl. registration confound and remaining failure modes

**Phase 6 — Writing & figures**
- [ ] All figures regenerated at ≥300 DPI (publication setting enforced in plotting code)
- [ ] Paper ≤ 9–10 pages; every equation/number matches released code
- [ ] Reproducibility: code released, configs pinned, seeds fixed
- [ ] Internal honesty pass against §13 anti-goals before submission

**Phase 7 (optional) — Schrödinger upgrade**
- [ ] Schrödinger-bridge formulation, D1 ablation vs Brownian
- [ ] Tighten thermodynamic framing if upgrade succeeds

## 15. Definition of "done" (end goal)

A reproducible, single, bidirectional 3D model that translates MR↔CT across brain and pelvis, where: (1) the bridge formulation is justified against standard diffusion with significance; (2) the anisotropic operator is shown to specifically reduce boundary error; (3) the joint cross-organ result is characterized honestly; (4) a deep bone-boundary failure analysis is the paper's analytical centerpiece; and (5) the manuscript is ≤10 pages, ≥300 DPI throughout, with the paper matching the code exactly — submitted to MICCAI (or fallback medical venue).

## 16. Open decisions / parking lot

- Final working title / acronym.
- Patch size & sampling step count (pin after GPU profiling).
- Objective parameterization (noise vs x₀ vs velocity prediction) for the bridge.
- Whether cross-attention conditioning is needed or channel-concat suffices.
- Dose/gamma analysis feasibility.
- Exact recent baselines (pending literature search).
- Brownian → Schrödinger upgrade go/no-go (time-dependent).
