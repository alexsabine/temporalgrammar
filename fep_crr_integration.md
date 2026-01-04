# FEP-CRR Integration: A Complete Synthesis

## The Free Energy Principle and Active Inference with Coherence-Rupture-Regeneration

### Abstract

This document provides a rigorous integration of the Free Energy Principle (FEP), Active Inference, and Coherence-Rupture-Regeneration (CRR). We show that:

1. **FEP handles within-model inference; CRR handles between-model transitions**
2. **Coherence is the dual of free energy**: C(t) = F₀ - F(t)
3. **Precision and rigidity are inverses**: Π = 1/Ω (up to normalization)
4. **Rupture is Bayesian model switching** made explicit and discontinuous
5. **The complete system** is "FEP-CRR Active Inference" with explicit temporal structure

The synthesis resolves several open problems in FEP and grounds Ω in terms of precision.

---

# Part I: Foundations

## 1. The Free Energy Principle (Review)

### 1.1 Core Statement

The Free Energy Principle (Friston, 2010) states:

> *Any self-organizing system that maintains itself against entropy must minimize variational free energy.*

**Variational Free Energy:**

$$F = D_{KL}[q(\theta) \| p(\theta | y)] - \log p(y)$$

$$= \underbrace{D_{KL}[q(\theta) \| p(\theta)]}_{\text{Complexity}} - \underbrace{\mathbb{E}_q[\log p(y | \theta)]}_{\text{Accuracy}}$$

where:
- q(θ) = approximate posterior (beliefs)
- p(θ|y) = true posterior
- p(y) = model evidence
- θ = hidden states
- y = observations

### 1.2 Key Properties

**Proposition 1.1 (Free Energy Bounds Surprise).**
$$F \geq -\log p(y) = \text{Surprise}$$

with equality when q(θ) = p(θ|y).

**Proposition 1.2 (Minimizing F ≈ Bayesian Inference).**
$$\arg\min_q F = p(\theta | y)$$

Free energy minimization recovers exact Bayesian inference.

### 1.3 The FEP Dynamics

Belief updating follows gradient descent on F:

$$\dot{q} = -\nabla_q F$$

For Gaussian beliefs q(θ) = N(μ, Σ):

$$\dot{\mu} = -\frac{\partial F}{\partial \mu} = \Pi \cdot \varepsilon$$

where:
- Π = precision (inverse variance)
- ε = y - g(μ) = prediction error

**This is perception: updating beliefs to minimize prediction error.**

---

## 2. Active Inference (Review)

### 2.1 Core Extension

Active Inference extends FEP to action:

> *Agents minimize free energy not just by updating beliefs, but by acting on the world.*

**Expected Free Energy:**

$$G(\pi) = \underbrace{\mathbb{E}_{q}[D_{KL}[q(\theta | \pi) \| q(\theta)]]}_{\text{Epistemic: Information Gain}} + \underbrace{\mathbb{E}_{q}[-\log p(y | \pi)]}_{\text{Pragmatic: Goal Achievement}}$$

where π is a policy (sequence of actions).

### 2.2 The Active Inference Loop

1. **Perceive:** Minimize F by updating beliefs μ
2. **Plan:** Evaluate G(π) for candidate policies
3. **Act:** Select π* = argmin G(π)
4. **Observe:** Receive new observation y
5. **Repeat**

### 2.3 Precision and Attention

**Precision** Π modulates the influence of prediction errors:

$$\dot{\mu} = \Pi \cdot \varepsilon$$

- High Π: Strong influence of sensory data (attention)
- Low Π: Beliefs dominate (prior expectations)

**Precision is itself inferred:**

$$\dot{\Pi} = -\frac{\partial F}{\partial \Pi}$$

This creates a hierarchy: beliefs about beliefs about beliefs...

---

## 3. The Gap: What FEP Doesn't Handle

### 3.1 The Model Switching Problem

FEP assumes a fixed generative model m. It handles:
- Updating beliefs about states θ within m
- Optimizing precision Π within m
- Selecting actions π within m

**FEP does NOT handle:**
- When to switch from model m to model m'
- How to transition between fundamentally different models
- What happens to accumulated beliefs during switching

### 3.2 The Continuity Assumption

FEP dynamics are continuous:

$$\dot{\mu} = -\nabla_\mu F, \quad \dot{\Pi} = -\nabla_\Pi F$$

**But model switching is discontinuous.** At some moment, the agent must:
- Abandon model m (with its accumulated beliefs)
- Adopt model m' (with different structure)
- Somehow preserve useful information

**This is the rupture problem.**

### 3.3 The Temporal Structure Gap

FEP is essentially Markovian: the current state μ_t depends only on μ_{t-1} and y_t.

**But agents have non-Markovian memory:**
- Past experiences influence current decisions
- Historical coherence accumulates
- Expertise requires long-term integration

**How is this encoded?**

---

# Part II: CRR Fills the Gap

## 4. CRR as the Between-Model Structure

### 4.1 The Fundamental Correspondence

| FEP | CRR |
|-----|-----|
| Within-model inference | Between-model transitions |
| Continuous gradient descent | Discontinuous rupture events |
| Precision Π | Rigidity Ω⁻¹ |
| Free energy F | Coherence C = F₀ - F |
| Model updating | Model switching |
| Markovian (state-based) | Non-Markovian (history-weighted) |

### 4.2 Coherence as Dual of Free Energy

**Theorem 4.1 (Coherence-Free Energy Duality).**

Define coherence as accumulated free energy reduction:

$$\boxed{\mathcal{C}(t) = F(0) - F(t) = -\int_0^t \frac{dF}{d\tau} \, d\tau}$$

Then:
1. **C ≥ 0** (free energy decreases under gradient flow)
2. **Ċ = -Ḟ ≥ 0** (coherence increases as free energy decreases)
3. **C measures learning:** Higher C = more free energy minimized = more learned

**Proof.**

(1) By the FEP, F decreases under belief updating:
$$\frac{dF}{dt} = \nabla_\mu F \cdot \dot{\mu} = -\|\nabla_\mu F\|^2 \leq 0$$

(2) Direct differentiation.

(3) F measures surprise; reducing F means better predictions; C measures cumulative prediction improvement. ∎

### 4.3 Rupture as Bayesian Model Comparison

**Theorem 4.2 (Rupture = Model Switching).**

Under FEP with model comparison, rupture occurs when:

$$\log \frac{p(y_{1:n} | m')}{p(y_{1:n} | m)} > \log \frac{p(m)}{p(m')}$$

This is equivalent to:

$$\mathcal{C}_m(n) - \mathcal{C}_{m'}(n) > \Omega$$

where Ω = log(p(m)/p(m')) is the log prior odds.

**Proof.**

By the coherence-likelihood correspondence:
$$\mathcal{C}_m(n) = -\log p(y_{1:n} | m) + \text{const}$$

The Bayes factor condition:
$$\frac{p(m' | y_{1:n})}{p(m | y_{1:n})} > 1$$

becomes:
$$\log p(y_{1:n} | m') - \log p(y_{1:n} | m) > \log \frac{p(m)}{p(m')}$$
$$-\mathcal{C}_{m'}(n) + \mathcal{C}_m(n) > \Omega$$
$$\mathcal{C}_m(n) - \mathcal{C}_{m'}(n) > \Omega$$ ∎

### 4.4 Regeneration as Precision-Weighted Integration

**Theorem 4.3 (Regeneration and Precision).**

The regeneration operator:

$$\mathcal{R}[\Phi](t) = \frac{1}{Z}\int_0^t \Phi(\tau) \cdot e^{\mathcal{C}(\tau)/\Omega} \, d\tau$$

has an FEP interpretation:

$$e^{\mathcal{C}/\Omega} = \Pi \cdot \Omega$$

where Π is the precision accumulated during learning.

**Interpretation:** History is weighted by precision. High-precision (confident) beliefs contribute more to regeneration than low-precision (uncertain) beliefs.

---

## 5. The Precision-Rigidity Correspondence

### 5.1 Precision in FEP

Precision Π = 1/σ² is the inverse variance of beliefs:
- High Π: Sharp, confident beliefs
- Low Π: Diffuse, uncertain beliefs

Precision modulates prediction error:
$$\dot{\mu} = \Pi \cdot \varepsilon$$

### 5.2 Rigidity in CRR

Rigidity Ω is the rupture threshold:
- High Ω: Slow to rupture, stable model
- Low Ω: Fast rupture, flexible model

### 5.3 The Fundamental Relationship

**Theorem 5.1 (Precision-Rigidity Duality).**

$$\boxed{\Pi = \frac{1}{\Omega}}$$

or more precisely:

$$\Pi(t) = \frac{1}{\Omega} \cdot e^{\mathcal{C}(t)/\Omega}$$

**Interpretation:**
- **At t = 0:** Π(0) = 1/Ω (prior precision = inverse rigidity)
- **As C grows:** Precision increases exponentially (learning tightens beliefs)
- **At rupture (C = Ω):** Π = e/Ω (maximum precision before reset)

**Proof Sketch.**

In FEP, precision is updated to minimize F:
$$\dot{\Pi} = -\frac{\partial F}{\partial \Pi}$$

For Gaussian beliefs:
$$F = \frac{1}{2}\Pi \varepsilon^2 - \frac{1}{2}\log \Pi + \text{const}$$

Setting ∂F/∂Π = 0:
$$\Pi^* = \frac{1}{\varepsilon^2}$$

But ε² ~ exp(-C/Ω) for a well-learning system (prediction error decreases). Therefore:
$$\Pi^* \propto e^{\mathcal{C}/\Omega}$$

The constant of proportionality is 1/Ω from dimensional analysis. ∎

### 5.4 The Ω = 1/π Conjecture in FEP Terms

If Ω = 1/π, then the prior precision is:

$$\Pi_0 = \frac{1}{\Omega} = \pi \approx 3.14$$

**Why might this be natural?**

In FEP, the "natural" precision scale is set by the generative model. For periodic processes (oscillations, rhythms), the fundamental period is 2π. The precision of phase estimation is:

$$\Pi_{\text{phase}} = \frac{1}{\sigma_{\text{phase}}^2} \sim \frac{1}{(1/\pi)^2} = \pi^2$$

Taking the square root for standard deviation gives Π ~ π.

**Conjecture 5.2.** The universal value Ω = 1/π arises from the periodic structure of temporal prediction in biological systems.

---

## 6. Extended Active Inference with CRR

### 6.1 The FEP-CRR Active Inference Loop

The complete cycle combines FEP and CRR:

```
┌─────────────────────────────────────────────────────────┐
│                    FEP-CRR CYCLE                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │          COHERENCE PHASE (FEP)                   │    │
│  │                                                  │    │
│  │  1. Perceive: μ̇ = -∇μF = Π·ε                    │    │
│  │  2. Accumulate: C += -∫Ḟ dt                     │    │
│  │  3. Update precision: Π = (1/Ω)·exp(C/Ω)        │    │
│  │  4. Plan: π* = argmin G(π)                       │    │
│  │  5. Act: execute action                          │    │
│  │  6. Check: C < Ω?                                │    │
│  │     • Yes → continue                             │    │
│  │     • No → RUPTURE                               │    │
│  └──────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼ (when C ≥ Ω)                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │           RUPTURE (CRR)                          │   │
│  │                                                   │   │
│  │  7. Compare models: BF = exp(Cm - Cm')           │   │
│  │  8. Switch: m → m' if BF > exp(Ω)                │   │
│  │  9. Mark: δ(t - t*)                              │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │         REGENERATION (CRR)                       │   │
│  │                                                   │   │
│  │  10. Weight history: w(τ) ∝ exp(C(τ)/Ω)          │   │
│  │  11. Reconstruct: μ' = R[μ](t*) = ∫w(τ)μ(τ)dτ   │   │
│  │  12. Reset: C → αC (partial reset, 0 < α < 1)   │   │
│  │  13. Initialize: new model with μ'               │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│              Return to COHERENCE PHASE                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 6.2 The Expected Coherence Functional

Active Inference uses Expected Free Energy G(π). The CRR-extended version uses **Expected Coherence Gain**:

$$\boxed{E[\Delta\mathcal{C}](\pi) = \underbrace{\mathbb{E}_q[I(Y; \Theta | \pi)]}_{\text{Epistemic: Learning}} + \underbrace{\mathbb{E}_q[\mathcal{C}(Y | \pi)]}_{\text{Pragmatic: Consolidation}}}$$

**Epistemic term:** Information gain about hidden states (exploration)
**Pragmatic term:** Coherence consolidation toward preferences (exploitation)

**Proposition 6.1.** E[ΔC] = -G (up to sign conventions).

The expected coherence gain is the negative expected free energy.

### 6.3 The Role of Ω in Exploration-Exploitation

Ω controls the exploration-exploitation tradeoff:

$$\text{Policy Score}(\pi) = \Omega \cdot \text{Epistemic}(\pi) + \frac{1}{\Omega} \cdot \text{Pragmatic}(\pi)$$

| Ω | Regime | Behavior |
|---|--------|----------|
| Low Ω (< 1/π) | **Rigid/Exploitation** | Prior-dominated, fast rupture, high precision, routine |
| Ω ≈ 1/π | **Balanced** | Optimal tradeoff, adaptive, resilient |
| High Ω (> 1/π) | **Fluid/Exploration** | Data-dominated, slow rupture, low precision, curious |

### 6.4 Hierarchical FEP-CRR

Real agents have hierarchical generative models. The FEP-CRR structure applies at each level:

```
Level 3:  C³ ──────────────────────► Ω³ ──► Rupture³
          ▲                          (slowest, rarest)
Level 2:  C² ─────────► Ω² ──► Rupture²
          ▲              (intermediate)
Level 1:  C¹ ──► Ω¹ ──► Rupture¹
                 (fastest, most frequent)
```

**Scale Coupling (from Multiscale CRR):**

$$L^{(n+1)}(t) = \lambda \sum_{t_k \in T^{(n)}} R^{(n)}(t_k) \cdot \delta(t - t_k)$$

What appears as smooth coherence at level n+1 is counting ruptures at level n.

**Hierarchical Precision:**

$$\Pi^{(n)} = \frac{1}{\Omega^{(n)}} \cdot e^{\mathcal{C}^{(n)}/\Omega^{(n)}}$$

Higher levels have larger Ω (slower rupture, more integration) and thus lower prior precision (more uncertainty about abstract features).

---

# Part III: Theoretical Implications

## 7. Resolving FEP Open Problems

### 7.1 The Dark Room Problem

**Problem:** If agents minimize free energy, why don't they seek dark rooms (zero sensory input = zero prediction error)?

**CRR Resolution:**
- Coherence still accumulates in dark rooms (maintenance cost)
- Rupture eventually occurs (Ω is finite)
- Regeneration requires historical richness
- Agents that only minimize immediate F get trapped; agents that manage C across time explore

**Formally:** The expected coherence gain E[ΔC] includes epistemic value:
$$E[\Delta\mathcal{C}](\text{dark room}) < E[\Delta\mathcal{C}](\text{explore})$$

Exploration builds richer regeneration potential.

### 7.2 The Prior Problem

**Problem:** Where do priors come from? FEP assumes them.

**CRR Resolution:**
- Priors emerge from regeneration: weighted historical integration
- Each cycle refines priors through exp(C/Ω) weighting
- High-coherence experiences become strong priors
- Priors are not given but constructed through CRR cycles

$$p(\theta | m_{n+1}) = \mathcal{R}[p(\theta | m_n)](t_n^*)$$

### 7.3 The Precision Problem

**Problem:** How is precision set? Self-referential optimization is unstable.

**CRR Resolution:**
- Precision is dual to rigidity: Π = 1/Ω (prior precision)
- Precision grows with coherence: Π(t) = (1/Ω)exp(C/Ω)
- Rupture resets precision dynamics (prevents runaway)
- Ω is set by system architecture (not self-optimized)

### 7.4 The Model Switching Problem

**Problem:** How do agents switch between fundamentally different models?

**CRR Resolution:**
- Rupture is the explicit switching mechanism
- Threshold Ω = log prior odds sets the switching criterion
- Regeneration transfers useful information to new model
- The Dirac delta δ(t-t*) marks the discontinuity

---

## 8. Implications for Consciousness and Agency

### 8.1 Consciousness as Coherence-Rupture Dynamics

**Proposal:** Conscious experience arises from the interface between coherence accumulation and rupture potential.

| State | C relative to Ω | Phenomenology |
|-------|-----------------|---------------|
| C ≪ Ω | Low coherence | Scattered, distractible, dreamlike |
| C → Ω | Pre-rupture | Focused, "flow," present-moment |
| C = Ω | Rupture | Insight, decision, "now-moment" |
| C > Ω | Post-rupture | Integration, consolidation, memory |

**The "present moment"** is the δ-function: dimensionless, instantaneous, the point of genuine choice.

### 8.2 Agency as Ω Modulation

**Proposal:** Agency is the capacity to modulate Ω.

- **Increase Ω:** Become more exploratory, flexible, open
- **Decrease Ω:** Become more exploitative, rigid, focused

**Practices that modulate Ω:**
- Meditation: Increases Ω (opens to larger history)
- Stress: Decreases Ω (narrows to immediate)
- Psychedelics: Dramatically increases Ω ("entropic brain")
- Trauma: Decreases Ω (rigid, repetitive patterns)

### 8.3 The Self as CRR Pattern

**Proposal:** The "self" is not a thing but a CRR pattern—a characteristic signature of:
- Typical coherence accumulation rate
- Characteristic Ω value
- Preferred rupture dynamics
- Regeneration style (what history is weighted)

**Personal growth** = changing one's CRR signature.
**Personality** = stable CRR signature.
**Transformation** = rupture-regeneration with new Ω.

---

## 9. Formal Integration: The Master Equations

### 9.1 The FEP-CRR Equations of Motion

**Coherence Phase (t < t*):**

$$\dot{\mu} = -\Omega \nabla_\mu F(\mu, y)$$

$$\dot{\mathcal{C}} = \Omega \|\nabla_\mu F\|^2$$

$$\Pi(t) = \frac{1}{\Omega} e^{\mathcal{C}(t)/\Omega}$$

**Rupture Condition:**

$$t^* = \inf\{t : \mathcal{C}(t) \geq \Omega\}$$

**Regeneration (at t = t*):**

$$\mu(t^{*+}) = \frac{\int_0^{t^*} \mu(\tau) \cdot e^{\mathcal{C}(\tau)/\Omega} \cdot d\tau}{\int_0^{t^*} e^{\mathcal{C}(\tau)/\Omega} \cdot d\tau}$$

$$\mathcal{C}(t^{*+}) = \alpha \cdot \mathcal{C}(t^{*-}), \quad 0 < \alpha < 1$$

### 9.2 The Variational Objective

The complete objective combines FEP and CRR:

$$\mathcal{L} = \underbrace{\int_0^T F(\mu(t), y(t)) \, dt}_{\text{FEP: Minimize Free Energy}} + \underbrace{\Omega \cdot N_{\text{ruptures}}}_{\text{CRR: Rupture Penalty}}$$

where N_ruptures is the number of rupture events.

**Interpretation:**
- Minimize total free energy (FEP)
- But ruptures are costly (each costs Ω)
- Optimal dynamics: minimize F while managing ruptures

### 9.3 The Partition Function

The complete statistical mechanics of FEP-CRR:

$$Z = \sum_{m \in \text{Models}} \int \mathcal{D}\mu \, e^{-\mathcal{L}[\mu]/\Omega}$$

$$= \sum_m e^{-F_m^{\text{baseline}}/\Omega} \cdot Z_m$$

where Z_m is the partition function for model m.

**Physical Interpretation:**
- Sum over models (model selection)
- Integrate over belief trajectories (inference)
- Weight by exp(-L/Ω) (Boltzmann factor)
- Ω plays the role of temperature

---

# Part IV: Experimental Predictions

## 10. Testable Predictions of FEP-CRR

### 10.1 Precision Dynamics

**Prediction 1:** Precision should increase exponentially during stable learning:

$$\Pi(t) \propto e^{\mathcal{C}(t)/\Omega}$$

**Test:** Measure confidence/precision in perceptual learning tasks. Expect exponential growth until plateau/reset.

### 10.2 Rupture Timing

**Prediction 2:** Rupture intervals should have a characteristic distribution determined by Ω:

$$P(\Delta t_{\text{rupture}}) \sim \text{First-passage time distribution to } \Omega$$

For constant coherence rate L: $\mathbb{E}[\Delta t] = \Omega / L$

**Test:** Measure inter-insight intervals in problem-solving. Should scale with task difficulty (higher Ω for harder problems).

### 10.3 Memory Weighting

**Prediction 3:** Recall probability should be weighted by exp(C/Ω):

$$P(\text{recall item } i) \propto e^{\mathcal{C}_i / \Omega}$$

High-coherence (well-learned) items recalled more easily.

**Test:** Measure recall probability vs. learning depth. Expect exponential relationship.

### 10.4 Ω Modulation

**Prediction 4:** Interventions that increase Ω should:
- Slow rupture
- Increase exploration
- Flatten precision dynamics
- Access deeper history

**Test:** Compare meditation practitioners (predicted high Ω) vs. controls on exploration-exploitation tasks.

### 10.5 Hierarchical Regularization

**Prediction 5:** Higher hierarchical levels should have:
- Larger Ω
- Lower variance in rupture timing (CLT regularization)
- More Gaussian interval distributions

**Test:** Compare timing variability of micro-saccades (low level) vs. attention shifts (mid level) vs. task switching (high level).

---

## 11. Computational Implementation

### 11.1 Algorithm: FEP-CRR Active Inference Agent

```
Algorithm: FEP-CRR-Active-Inference

Initialize:
    μ ← prior mean
    C ← 0
    Ω ← system parameter (e.g., 1/π)
    model ← initial generative model

Loop:
    # PERCEPTION (FEP)
    y ← observe()
    ε ← y - predict(μ, model)
    Π ← (1/Ω) * exp(C/Ω)
    μ ← μ + learning_rate * Π * ε

    # COHERENCE ACCUMULATION (CRR)
    ΔC ← 0.5 * Π * ε² * dt  # Free energy reduction
    C ← C + ΔC

    # PLANNING (Active Inference)
    for π in candidate_policies:
        G[π] ← expected_free_energy(π, μ, model)
        EC[π] ← expected_coherence_gain(π, μ, model)
    π* ← argmin(G) or argmax(EC)

    # ACTION
    execute(π*)

    # RUPTURE CHECK (CRR)
    if C ≥ Ω:
        # Compare models
        for m' in alternative_models:
            BF[m'] ← bayes_factor(m', model, y_history)

        if max(BF) > exp(Ω):
            # RUPTURE
            model_new ← argmax(BF)

            # REGENERATION
            history_weights ← exp(C_history / Ω)
            μ_new ← weighted_average(μ_history, history_weights)

            # RESET
            μ ← μ_new
            model ← model_new
            C ← α * C  # Partial reset

    # Store history
    history.append((μ, C, y, t))
```

### 11.2 Key Implementation Considerations

1. **History Management:** Store (μ, C, y, t) tuples with exponential decay for efficiency
2. **Model Space:** Define candidate alternative models for comparison
3. **Regeneration:** Implement weighted integration over history
4. **Hierarchical Extension:** Nest the algorithm for multi-level systems

---

# Part V: Summary

## 12. The Complete Picture

### 12.1 What FEP Provides
- Variational inference framework
- Perception as prediction error minimization
- Action as environment modification
- Precision as confidence weighting

### 12.2 What CRR Adds
- Between-model transitions (rupture)
- Non-Markovian memory (regeneration)
- Temporal structure (C accumulates, δ marks discontinuity)
- Scale hierarchy (multi-level CRR)

### 12.3 The Synthesis

**FEP-CRR Active Inference** is the complete framework for bounded agents:

$$\boxed{\text{Agent} = \text{FEP}(\text{within-model}) + \text{CRR}(\text{between-model})}$$

| Component | FEP | CRR | FEP-CRR |
|-----------|-----|-----|---------|
| Perception | ✓ | | ✓ |
| Action | ✓ | | ✓ |
| Learning | ✓ | | ✓ |
| Model switching | | ✓ | ✓ |
| Non-Markovian memory | | ✓ | ✓ |
| Discontinuity | | ✓ | ✓ |
| Hierarchy | ✓ | ✓ | ✓ |

### 12.4 The Key Equations

**Coherence-Free Energy Duality:**
$$\mathcal{C}(t) = F(0) - F(t)$$

**Precision-Rigidity Correspondence:**
$$\Pi = \frac{1}{\Omega} \cdot e^{\mathcal{C}/\Omega}$$

**Rupture Condition:**
$$\mathcal{C}_m - \mathcal{C}_{m'} > \Omega$$

**Regeneration Operator:**
$$\mathcal{R}[\Phi] = \frac{1}{Z}\int_0^{t^*} \Phi(\tau) \cdot e^{\mathcal{C}(\tau)/\Omega} \, d\tau$$

### 12.5 Open Questions

1. **Ω from FEP:** Can Ω = 1/π be derived from FEP precision dynamics?
2. **Biological Ω:** What determines Ω in neural systems?
3. **Optimal Ω:** Is there an optimal Ω for given environments?
4. **Ω plasticity:** How does Ω change with development/learning?
5. **Multi-agent CRR:** How do CRR cycles synchronize between agents?

---

## Conclusion

The integration of FEP and CRR provides a complete framework for understanding bounded, temporally-extended agents. FEP handles the smooth flow of inference within a model; CRR handles the discontinuous transitions between models. Together, they describe:

- How agents learn (coherence accumulation / free energy minimization)
- When agents must change (rupture at threshold)
- What agents remember (precision-weighted regeneration)
- Why agents explore (to build regeneration potential)

The precision-rigidity correspondence Π = 1/Ω unifies the two frameworks, suggesting they are dual descriptions of the same underlying process: **bounded observation of a complex world.**

---

**Document Status:** Complete theoretical synthesis. Awaiting empirical validation.

**References:**
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
- Friston, K. et al. (2017). Active Inference: A Process Theory. *Neural Computation*.
- Parr, T., Pezzulo, G., & Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*.
- Ramstead, M. et al. (2018). Answering Schrödinger's question: A free-energy formulation. *Physics of Life Reviews*.

**Citation:**
```
CRR-FEP Integration. A Complete Synthesis.
https://alexsabine.github.io/CRR/
```
