# Coherence-Rupture-Regeneration (CRR) Framework
## Summary for Collaboration Meeting

**Prepared for:** Nicolás Hinrichs
**Date:** Tuesday 14th January 2026
**Author:** Alexander Sabine
**Affiliation:** Active Inference Institute • cohere.org.uk

---

## Executive Summary

This document provides an overview of the Coherence-Rupture-Regeneration (CRR) framework, summarising the current state of development, empirical validation, and potential publication pathways. CRR is a candidate **coarse-grain temporal grammar** that bridges the Free Energy Principle (FEP) / Active Inference with a slow-manifold temporal structure capturing ontological reorganisation across development and interaction.

As Nicolás noted: *"FEP/Active Inference providing the fast, locally Markovian inferential machinery, and CRR supplying a slow-manifold temporal grammar that captures ontological reorganisation across development and interaction."*

---

# Part I: What Has Been Achieved

## 1. Mathematical Framework

CRR provides three fundamental operators formalising how systems maintain identity through discontinuous change:

### 1.1 Core Equations

| Operator | Equation | Interpretation |
|----------|----------|----------------|
| **Coherence** | C(x,t) = ∫₀ᵗ L(x,τ) dτ | Accumulated integration over time; system's "temporal budget" |
| **Rupture** | δ(t - t*) when C(t*) ≥ Ω | Instantaneous discontinuous transition at threshold |
| **Regeneration** | R[φ] = ∫ φ(x,τ)·exp(C/Ω)·Θ(t-τ) dτ | Memory-weighted reconstruction from historical field |

### 1.2 The Omega (Ω) Parameter

- **Ω = 1/π ≈ 0.318** for binary/toggle systems (Z₂ symmetry)
- **Ω = 1/2π ≈ 0.159** for continuous/rotational systems (SO(2) symmetry)
- Controls rigidity-liquidity: Low Ω = frequent brittle ruptures; High Ω = rare transformative ruptures

### 1.3 Mathematical Proofs

The repository contains **24 independent proof sketches** deriving CRR from established mathematical domains:

- **Information Geometry**: Coherence as geodesic arc length (Bonnet-Myers theorem → Ω = π/√κ)
- **Martingale Theory**: Coherence as quadratic variation (Wald's identity, Optional Stopping)
- **Ergodic Theory**: Rupture from Poincaré recurrence (Kac's lemma → Ω = 1/μ(A))
- **Category Theory**: CRR as natural transformation and Kan extension
- **Optimal Transport**: CRR as Wasserstein gradient flow
- Plus: Gauge theory, quantum mechanics, tropical geometry, HoTT, Floer homology, CFT, etc.

**Meta-theorem**: All proofs arise from a single principle: **Bounded Observer → CRR Dynamics**

---

## 2. Empirical Validation

### 2.1 Biological Systems

| Domain | Finding | Evidence Quality |
|--------|---------|------------------|
| **Wound Healing** | R² = 0.9989; 80% max recovery ceiling reflects inability to access developmental coherence | Strong fit to data |
| **Muscle Hypertrophy** | R² = 0.9985; 10/10 predictions confirmed; myonuclei as coherence retention ("muscle memory") | Strong prospective predictions |
| **Saltatory Growth** | 11/11 predictions validated; stasis/burst maps to chondrocyte CRR cycles | Pattern confirmation |
| **Bone Remodeling** | Derived Ω ≈ 1.2 predicts 4-5× formation:resorption asymmetry (empirical: 4-5×) | Novel system test |
| **Coral Bleaching** | Derived Ω ≈ 3-10 predicts extreme recovery:bleaching asymmetry (empirical: 50-500×) | Order of magnitude match |
| **Dwarf Nova** | Derived Ω ≈ 1.25 predicts 4-6× quiescence:outburst (empirical: 4-8×) | Physical system test |

### 2.2 The 16 Nats Hypothesis

A key prediction tested across **16 diverse systems**:

**Hypothesis**: Information accumulated at rupture converges on Ω ≈ 16 nats ≈ 23 bits

**Results**:
- **Mean**: 15.6 nats (prediction: 16 nats)
- **95% CI**: 14.5 - 16.7 nats (contains predicted value)
- **p-value**: > 0.4 (prediction indistinguishable from empirical mean)

Systems tested: Working memory, conscious awareness, visual STM, cognitive control, protein folding, cell signalling, language processing, retinal processing, morphogen gradients, T-cell activation, network cascades, synaptic storage, apoptosis decisions.

---

## 3. FEP-CRR Integration

A complete theoretical synthesis showing CRR and FEP as complementary lenses:

| FEP | CRR |
|-----|-----|
| Within-model inference | Between-model transitions |
| Continuous gradient descent | Discontinuous rupture events |
| Epistemic structure (belief updating) | Temporal structure (past → future) |
| Model updating | Model switching |
| Markovian (state-based) | Non-Markovian (history-weighted) |

### Key Mathematical Relationships

| Relationship | Equation |
|--------------|----------|
| **Coherence-Free Energy Duality** | C(t) = F(0) - F(t) |
| **Precision-Rigidity Correspondence** | Π = (1/Ω)·exp(C/Ω) |
| **Rupture as Model Comparison** | Rupture when C_m - C_m' > Ω |

### Problems FEP-CRR Resolves

1. **The Dark Room Problem**: Why don't agents seek minimal stimulation?
2. **The Prior Problem**: Where do priors come from?
3. **The Model Switching Problem**: How do agents transition between fundamentally different models?

---

## 4. Interactive Demonstrations

**Website**: [https://alexsabine.github.io/CRR/](https://alexsabine.github.io/CRR/)

The repository contains **88 interactive HTML simulations** across domains:

### Categories

| Category | Examples | Count |
|----------|----------|-------|
| **Biological** | Fish learning, butterfly metamorphosis, wound healing, ecosystem | 20+ |
| **Physical** | Black holes, thermodynamics, weather, hurricane dynamics | 15+ |
| **Cognitive** | Child development, perceiving agents, maze solving, FEP integration | 15+ |
| **Chemical/Material** | Ice formation, snowflakes, mother of pearl, periodic table | 12+ |
| **FEP Integration** | FEP-CRR 5-stages, agent shapes, game demonstration | 8+ |

### Key Demonstrations

- [FEP-CRR Integration Demo](https://alexsabine.github.io/CRR/fep-crr-finale-wspeech.html)
- [Child Development Stages](https://alexsabine.github.io/CRR/child_dev.html)
- [Dirac Delta Rupture](https://alexsabine.github.io/CRR/dirac-delta-crr.html)
- [Entropic Brain Dynamics](https://alexsabine.github.io/CRR/entropic-crr.html)

---

## 5. Repository Contents Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Markdown Documentation** | 17 | Mathematical foundations, proofs, empirical validation |
| **HTML Simulations** | 88 | Interactive demonstrations across domains |
| **PDF Treatises** | 17 | Formal academic treatments |
| **Python Scripts** | 5 | Empirical validation and computational analysis |
| **Diagrams/Images** | 60+ | Visualisations, validation plots, concept diagrams |

---

# Part II: Book Chapters Overview

## "Mathematics of Change" Monograph Structure

The manuscript develops CRR as mathematised process philosophy with FEP resonance:

### Current Chapters

| Chapter | Title | Content | Development Status |
|---------|-------|---------|-------------------|
| **Prologue** | Temporal Glossary | Personal narrative of framework development; translation tables mapping CRR to 12 domains | Complete |
| **1** | Stability As Change | Philosophical foundations: Heraclitus → Whitehead; Prigogine's dissipative structures; metastability | Complete |
| **2** | Coherence Through Time | C(x,t) = ∫L(x,τ)dτ exposition; FEP connection; bioelectric signalling (Levin); participatory universe | Complete |
| **3** | Rupture and Temporal Phase Shifts | When optimisation fails; Dirac delta mathematics; scale-free rupture | Complete |
| **4** | Regeneration and Repair | Historical field signal φ(x,τ); exponential weighting exp(C/Ω); Ω as rigidity modulator | Complete |
| **5** | Jacob's Ladder: Human-AI Systems | Human-LLM interaction analysis; catastrophic forgetting in ML; continual learning strategies | Complete |

### Cross-Domain Translations (Prologue)

The monograph includes translation tables mapping CRR to:
- Physics & Thermodynamics
- FEP/Active Inference
- Dynamical Systems
- Neuroscience
- Developmental Psychology
- Trauma/Clinical
- Whitehead Process Philosophy
- Buddhist Philosophy
- Contemplative Traditions
- Ecology
- Social Systems
- Literature & Music

---

# Part III: Critical Periods Connection

## Nicolás's Suggestion: Critical Periods as Ω Modulation

This maps directly to Alexander's Child Development background and provides principled developmental interpretation:

### CRR Interpretation of Critical Periods

| Plasticity Concept | CRR Mapping | Empirical Prediction |
|-------------------|-------------|---------------------|
| Critical period = high plasticity | Elevated Ω (low precision, broad memory access) | Higher CV in developmental windows |
| CP closure = stability | Decreasing Ω with maturation | CV decreases with age |
| Experience-dependent refinement | Coherence accumulation shapes regeneration weights | exp(C/Ω) peaks at formative experiences |
| Stability landscape (Knudsen) | Ω determines landscape curvature | Low Ω = deep attractors; High Ω = shallow |
| CP reopening (Cisneros-Franco) | Ω elevation through intervention | Testable via drugs/training |
| PV cell maturation (Hensch) | Inhibitory regulation as precision control | PV density correlates with 1/Ω |

### Key References

- **Knudsen (2004)**: "Sensitive Periods in the Development of the Brain and Behavior" - [J Cogn Neurosci](https://pubmed.ncbi.nlm.nih.gov/15509387/) - Stability landscape metaphor maps to Ω
- **Cisneros-Franco et al. (2020)**: "Critical periods of brain development" - [Handbook of Clinical Neurology](https://pubmed.ncbi.nlm.nih.gov/32958196/) - CP reopening as Ω elevation
- **Hensch & Quinlan (2018)**: Critical period regulation - PV cells/gamma oscillations associated with CP plasticity

### Distinctive Angle

Alexander's background in Child Development (Psychology, Early Childhood Education) positions CRR uniquely as a **theory of developmental transformation**—how systems navigate change through time—rather than pure computational neuroscience. This differentiates from existing approaches and connects to Nicolás's "slow-manifold temporal grammar" framing.

---

# Part IV: Publication Strategy

## Immediate Targets (Q1-Q2 2026)

| # | Target | Focus | Collaborators | Priority |
|---|--------|-------|---------------|----------|
| 1 | **Royal Society Phil Trans A** | Agency, System 1/2, World Models (Open call closes Mar 31) | Nicolás, Peter?, Maxwell? | HIGH |
| 2 | **Frontiers in Computational Neuroscience** | FEP-CRR Integration; precision-rigidity correspondence | Maxwell Ramstead? | HIGH |
| 3 | **Developmental Cognitive Neuroscience** | Critical Periods as Ω Modulation | Nicolás (plasticity) | MEDIUM |
| 4 | **Computational Brain & Behavior** | 16 Nats Identity, information thresholds | | MEDIUM |
| 5 | **Conference: CCN 2026** | Computational validation paper | Peter Waade (pymdp) | MEDIUM |

## Paper Ideas by Strength of Evidence

### Tier 1: Strong Evidence Base

1. **"CRR as Temporal Grammar for Active Inference"**
   - FEP-CRR integration with precision-rigidity correspondence
   - Mathematical framework already complete
   - Addresses model switching problem in Active Inference

2. **"Empirical Validation of Phase Asymmetry Predictions"**
   - Bone remodeling, coral bleaching, dwarf nova results
   - System-specific Ω derivation via Kac's lemma
   - Novel prospective predictions

### Tier 2: Promising Connections

3. **"Critical Periods as Ω Modulation: A CRR Framework"**
   - Direct application to developmental neuroscience
   - Maps to Knudsen's stability landscape
   - Connects to Cisneros-Franco's CP reopening research

4. **"The 16 Nats Hypothesis: Universal Information Thresholds at Phase Transition"**
   - Statistical analysis across 16 systems
   - Convergence on 15.6 nats (p > 0.4 from prediction)

### Tier 3: Theoretical Extensions

5. **"CRR and Catastrophic Forgetting: Graceful Forgetting in Continual Learning"**
   - ML application with biological grounding
   - Connects to sleep/consolidation literature

6. **"Process Philosophy Formalised: CRR as Whiteheadian Temporal Structure"**
   - C→δ→R maps to concrescence→perishing→transition
   - Philosophical foundations paper

---

## Monograph Publishers

| Publisher | Fit | Notes |
|-----------|-----|-------|
| **MIT Press** | ★★★★★ | Published Active Inference textbook (Parr et al. 2022); Direct to Open (OA) |
| **Cambridge UP** | ★★★★☆ | Strong philosophy of mind list; Process philosophy series |
| **Oxford UP** | ★★★★☆ | Oxford Psychology Series |
| **Springer** | ★★★☆☆ | Studies in Brain and Mind series |

---

# Part V: Epistemic Status Assessment

## What CRR IS (Current Status)

- A mathematically coherent framework with explicit, testable equations
- Empirically validated across multiple domains with prospective predictions
- Grounded in established mathematics (drift-diffusion, competing accumulators, path integrals)
- Philosophically grounded in Whitehead's process philosophy
- Conjectured to connect to FEP through Ω = 1/π = σ² (precision relationship)

## What CRR IS NOT (Yet)

- NOT peer-reviewed or published in academic journals
- NOT formally derived from first principles (FEP derivation is conjectured, not proven)
- NOT experimentally tested in controlled laboratory conditions
- NOT yet subjected to adversarial critique from domain experts

## Confidence Levels

| Claim | Confidence | Status |
|-------|------------|--------|
| Core operators (C, δ, R) mathematically well-defined | HIGH (≥90%) | Sound |
| Integrate-and-fire / first-passage structure | HIGH (≥90%) | Established |
| MaxEnt derivation of exp(C/Ω) weighting | HIGH (≥90%) | Standard |
| Meta-theorem (CRR from bounded observation) | MODERATE (60-90%) | Compelling |
| 16 nats convergence | MODERATE (60-90%) | Needs more validation |
| FEP-CRR integration equations | MODERATE (60-90%) | Consistent |
| Specific value Ω = 1/π | LOWER (30-60%) | Lacks first-principles derivation |
| Consciousness interpretations | LOWER (30-60%) | Speculative |

---

# Part VI: Collaboration Opportunities

## For Tuesday's Meeting

1. **Identify co-authorship opportunities**: Which chapters/papers align with Nicolás's expertise?
2. **Critical periods paper**: Develop Ω-plasticity interpretation with Knudsen/Cisneros-Franco grounding
3. **Strategy for Peter and Maxwell meetings**

## Peter Thestrup Waade (Thursday)

- **pymdp integration**: CRR operators in Active Inference simulations
- Potential Python package collaboration
- Computational validation pathway

## Maxwell Ramstead (End of January)

- Key validation: Ω = 1/π correspondence to FEP precision
- Demo cohere.org.uk simulations
- Address representationalism concerns via process philosophy framing

---

# Appendix: Canonical Equations Reference

| Name | Equation | Meaning |
|------|----------|---------|
| Coherence | C(x,t) = ∫L(x,τ)dτ | Accumulated evidence of local fit |
| Rupture Condition | δ(t*) when E(t*) - C(t*) > Ω | Discontinuous transition |
| Regeneration | R = ∫φ(x,τ)exp(C/Ω)Θ(t*-τ)dτ | Memory-weighted reconstruction |
| Ω (Z₂ symmetry) | Ω = 1/π ≈ 0.318 | Binary/toggle systems |
| Ω (SO(2) symmetry) | Ω = 1/2π ≈ 0.159 | Continuous/rotational systems |
| CV Prediction | CV = Ω/2 | Derived from symmetry class |
| FEP Bridge | Ω = σ² = 1/π; Precision = π | Conjectured connection |
| 16 Nats Identity | Ω ≈ 16 nats ≈ 23 bits | Information threshold at rupture |
| Precision-Rigidity | Π = (1/Ω)·exp(C/Ω) | Dynamic precision from coherence |

---

## Key Resources

- **Repository**: [github.com/alexsabine/CRR](https://github.com/alexsabine/CRR)
- **Interactive Demos**: [alexsabine.github.io/CRR](https://alexsabine.github.io/CRR/)
- **Website**: [cohere.org.uk](https://cohere.org.uk)

---

**Document generated:** January 12, 2026
**Repository files reviewed:** 212 files across documentation, simulations, proofs, and validation code

---

## Sources

- [Don Tucker Wikipedia](https://en.wikipedia.org/wiki/Don_M._Tucker)
- [BEL Company](https://belco.tech/bel-company)
- [Knudsen 2004 - PubMed](https://pubmed.ncbi.nlm.nih.gov/15509387/)
- [Cisneros-Franco et al. 2020 - PubMed](https://pubmed.ncbi.nlm.nih.gov/32958196/)
- [Active Inference Institute - Board of Directors](https://www.activeinference.institute/board-of-directors)
