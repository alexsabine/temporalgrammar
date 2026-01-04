# Active Inference and Artificial Reasoning: A CRR Formulation

## Coherence-Rupture-Regeneration Approach to Structure Learning

### Based on Friston et al. (2025) arXiv:2512.21129

---

## Abstract

This technical note reformulates active inference and artificial reasoning through the lens of Coherence-Rupture-Regeneration (CRR). Where the original formulation uses continuous Bayesian model reduction for structure learning, we introduce explicit discontinuous transitions via rupture events. The key innovations are:

1. **Coherence accumulation** replaces free energy minimization as the primary dynamic
2. **Rupture** provides an explicit threshold mechanism for model switching
3. **Regeneration** replaces posterior-based reconstruction with trajectory-weighted integration
4. **Expected Coherence Gain** replaces Expected Free Energy for policy selection

We demonstrate that this formulation:
- Makes model switching explicit and discontinuous (rather than implicit and continuous)
- Provides non-Markovian memory through history weighting
- Unifies within-model and between-model inference under a single framework
- Predicts qualitatively different dynamics near model transitions

We illustrate using the same three-ball paradigm, showing how CRR captures "aha moments" as genuine rupture events rather than gradual evidence accumulation.

---

# Part I: Review of Original Formulation

## 1. Active Inference with Expected Free Energy

### 1.1 The Generative Model

Following the original paper, consider a partially observed Markov decision process with:
- Hidden states: s ∈ S
- Observations: o ∈ O
- Actions: a ∈ A (equivalently, policies π)
- Parameters: θ (e.g., transition matrices)
- Model structure: m ∈ M (which parameters exist)

The generative model factorizes as:

$$P(o, s, \theta, m | \pi) = P(o | s) P(s | \pi, \theta) P(\theta | m) P(m)$$

### 1.2 Expected Free Energy (Original)

Policy selection minimizes expected free energy:

$$G(\pi) = \mathbb{E}_{Q(o,s|\pi)}\left[\log Q(s|\pi) - \log P(o, s | \pi)\right]$$

This decomposes into:

$$G(\pi) = \underbrace{-\mathbb{E}_Q[D_{KL}[Q(s|o,\pi) \| Q(s|\pi)]]}_{\text{Epistemic: State Information Gain}} + \underbrace{\mathbb{E}_Q[-\log P(o)]}_{\text{Pragmatic: Expected Surprise}}$$

### 1.3 Bayesian Model Reduction (Original)

For model comparison, the original uses BMR. Given a full model m_F with posterior Q(θ|m_F), the evidence for a reduced model m_R is:

$$\log P(o | m_R) = \log P(o | m_F) + \log \frac{P(\theta | m_R)}{P(\theta | m_F)} - D_{KL}[Q(\theta | m_F) \| Q(\theta | m_R)]$$

Under Gaussian assumptions:

$$\log P(o | m_R) = \log P(o | m_F) + \frac{1}{2}\log\frac{|\Sigma_R|}{|\Sigma_F|} - \frac{1}{2}(\mu - \mu_R)^T \Sigma_R^{-1} (\mu - \mu_R) + \frac{1}{2}\text{tr}(\Sigma_R^{-1}C)$$

### 1.4 Structure Learning Information Gain (Original)

The original paper adds a third type of information gain—about model structure:

$$I_m(\pi) = \mathbb{E}_{Q(o|\pi)}[D_{KL}[Q(m|o,\pi) \| Q(m|\pi)]]$$

Total expected free energy becomes:

$$G(\pi) = G_{\text{state}}(\pi) + G_{\text{param}}(\pi) + G_{\text{structure}}(\pi)$$

---

# Part II: CRR Reformulation

## 2. Core CRR Operators for Active Inference

### 2.1 Coherence Accumulation

**Definition 2.1 (Model-Specific Coherence).**

For each model m, define the coherence accumulated from observations o_{1:t}:

$$\boxed{\mathcal{C}_m(t) = \sum_{i=1}^{t} \frac{1}{2}(o_i - g_m(\hat{s}_i))^T \Pi_m (o_i - g_m(\hat{s}_i))}$$

where:
- g_m(s) is the expected observation under model m at state s
- ŝ_i is the estimated state at time i
- Π_m is the precision (inverse noise covariance) under model m

**Proposition 2.1 (Coherence-Free Energy Duality).**

$$\mathcal{C}_m(t) = F_m(0) - F_m(t) + \text{const}$$

Coherence is accumulated free energy reduction.

### 2.2 Rupture Condition

**Definition 2.2 (Model Rupture).**

Rupture from model m to model m' occurs at time t* defined by:

$$\boxed{t^* = \inf\left\{t : \mathcal{C}_m(t) - \mathcal{C}_{m'}(t) > \Omega_{m \to m'}\right\}}$$

where the rigidity is:

$$\Omega_{m \to m'} = \log \frac{P(m)}{P(m')}$$

**Proposition 2.2 (Equivalence to Bayesian Model Comparison).**

The rupture condition is equivalent to:

$$\frac{P(m' | o_{1:t^*})}{P(m | o_{1:t^*})} > 1$$

**Proof.** By Bayes' theorem:
$$\frac{P(m' | o)}{P(m | o)} = \frac{P(o | m')}{P(o | m)} \cdot \frac{P(m')}{P(m)}$$

Taking logarithms:
$$\log \frac{P(m' | o)}{P(m | o)} = \log P(o | m') - \log P(o | m) + \log \frac{P(m')}{P(m)}$$

By the coherence-likelihood correspondence:
$$= -\mathcal{C}_{m'}(t) + \mathcal{C}_m(t) - \Omega_{m \to m'}$$

The posterior favors m' when this exceeds 0, i.e., when C_m - C_{m'} > Ω. ∎

### 2.3 The Rupture Event

**Definition 2.3 (Dirac Delta Representation).**

The rupture event is represented as:

$$\boxed{\delta(t - t^*)}$$

This is:
- **Instantaneous:** No intrinsic duration
- **Scale-free:** Only marks the crossing moment
- **Discontinuous:** A genuine break in model identity

### 2.4 Regeneration Operator

**Definition 2.4 (CRR Regeneration).**

Upon rupture at t*, the system regenerates beliefs under the new model m':

$$\boxed{\hat{s}(t^{*+}) = \mathcal{R}[\hat{s}](t^*) = \frac{\int_0^{t^*} \hat{s}(\tau) \cdot e^{\mathcal{C}_m(\tau)/\Omega} \, d\tau}{\int_0^{t^*} e^{\mathcal{C}_m(\tau)/\Omega} \, d\tau}}$$

**Interpretation:** The regenerated state is a weighted average over historical states, with weights exponential in coherence. High-coherence (well-learned) periods contribute more.

**Proposition 2.3 (Regeneration is MaxEnt Optimal).**

The weights w(τ) ∝ exp(C(τ)/Ω) are the unique maximum entropy distribution subject to:
1. Normalization
2. Mean coherence constraint

---

## 3. Expected Coherence Gain (Replacing Expected Free Energy)

### 3.1 Definition

**Definition 3.1 (Expected Coherence Gain).**

For a policy π, the expected coherence gain is:

$$\boxed{E[\Delta\mathcal{C}](\pi) = \underbrace{\mathbb{E}_{Q(o|\pi)}[\Delta\mathcal{C}_{\text{state}}]}_{\text{State Learning}} + \underbrace{\mathbb{E}_{Q(o|\pi)}[\Delta\mathcal{C}_{\text{param}}]}_{\text{Parameter Learning}} + \underbrace{\mathbb{E}_{Q(o|\pi)}[\Delta\mathcal{C}_{\text{structure}}]}_{\text{Structure Learning}}}$$

### 3.2 State Coherence Gain

The state component measures how much we learn about hidden states:

$$\Delta\mathcal{C}_{\text{state}}(\pi) = \mathbb{E}_{Q(o|\pi)}\left[\frac{1}{2}\varepsilon^T \Pi \varepsilon \right] = \frac{1}{2}\text{tr}(\Pi \cdot \text{Var}[o|\pi])$$

High expected prediction error → high coherence gain → epistemic value.

### 3.3 Parameter Coherence Gain

The parameter component measures learning about model parameters:

$$\Delta\mathcal{C}_{\text{param}}(\pi) = \mathbb{E}_{Q(o|\pi)}[D_{KL}[Q(\theta|o,\pi) \| Q(\theta|\pi)]]$$

### 3.4 Structure Coherence Gain (The Key Addition)

**Definition 3.2 (Structure Coherence Gain).**

$$\boxed{\Delta\mathcal{C}_{\text{structure}}(\pi) = \mathbb{E}_{Q(o|\pi)}\left[\max_{m'} \left(\mathcal{C}_m(t+1) - \mathcal{C}_{m'}(t+1)\right) - \max_{m'}\left(\mathcal{C}_m(t) - \mathcal{C}_{m'}(t)\right)\right]}$$

This measures how much the observation under π would increase the coherence gap between models—i.e., how much it would accelerate or delay rupture.

**Alternative Formulation:**

$$\Delta\mathcal{C}_{\text{structure}}(\pi) = \mathbb{E}_{Q(o|\pi)}[D_{KL}[Q(m|o,\pi) \| Q(m|\pi)]]$$

where the posterior over models uses coherence-based weights:

$$Q(m|o) \propto P(m) \cdot e^{-\mathcal{C}_m / \Omega}$$

---

## 4. CRR Active Reasoning Algorithm

### 4.1 The Complete Algorithm

```
Algorithm: CRR-Active-Reasoning

Initialize:
    m ← initial model
    s_history ← []
    C_history ← {m: [] for m in Models}
    Ω ← log prior odds (or system parameter 1/π)

For each time step t:

    # === COHERENCE PHASE ===

    # 1. Observe
    o_t ← observe()

    # 2. Infer state under current model
    ε_t ← o_t - g_m(ŝ_t)
    Π_t ← (1/Ω) * exp(C_m(t)/Ω)  # Precision grows with coherence
    ŝ_t ← ŝ_{t-1} + learning_rate * Π_t * ε_t

    # 3. Accumulate coherence for all models
    For each m' in Models:
        ε_m' ← o_t - g_{m'}(ŝ_t)
        C_{m'}(t) ← C_{m'}(t-1) + 0.5 * ε_m'^T Π_{m'} ε_m'

    # 4. Store history
    s_history.append((ŝ_t, C_m(t), t))

    # === POLICY SELECTION VIA EXPECTED COHERENCE GAIN ===

    # 5. Evaluate policies
    For each policy π:
        # State coherence gain
        EC_state[π] ← E[0.5 * ε^T Π ε | π]

        # Parameter coherence gain
        EC_param[π] ← E[D_KL[Q(θ|o,π) || Q(θ|π)]]

        # Structure coherence gain (KEY INNOVATION)
        For each alternative model m':
            ΔC_gap[m',π] ← E[C_m(t+1) - C_{m'}(t+1) | π] - (C_m(t) - C_{m'}(t))
        EC_structure[π] ← max_{m'} |ΔC_gap[m',π]|

        # Total expected coherence gain
        EC[π] ← EC_state[π] + EC_param[π] + EC_structure[π]

    # 6. Select policy maximizing expected coherence gain
    π* ← argmax(EC)

    # 7. Execute action
    execute(π*)

    # === RUPTURE CHECK ===

    # 8. Check rupture condition for all alternative models
    For each m' ≠ m:
        If C_m(t) - C_{m'}(t) > Ω_{m→m'}:

            # === RUPTURE EVENT ===

            # 9. Mark rupture
            t* ← t
            trigger_rupture(m, m', t*)

            # === REGENERATION ===

            # 10. Compute history weights
            weights ← [exp(C_m(τ)/Ω) for (ŝ, C_m, τ) in s_history]
            weights ← weights / sum(weights)  # Normalize

            # 11. Regenerate state under new model
            ŝ_new ← sum([w * ŝ for (w, (ŝ, _, _)) in zip(weights, s_history)])

            # 12. Transform to new model coordinates (if needed)
            ŝ_new ← transform(ŝ_new, m → m')

            # 13. Partial coherence reset
            α ← 0.3  # Retention factor
            C_{m'}(t*+) ← α * C_m(t*)

            # 14. Switch model
            m ← m'
            ŝ ← ŝ_new

            # 15. Optionally clear or decay history
            s_history ← decay(s_history, α)

            break  # Only one rupture per step

    # Continue to next time step
```

### 4.2 Key Differences from Original BMR Approach

| Aspect | Original (BMR) | CRR Version |
|--------|---------------|-------------|
| Model evidence | Computed continuously via BMR | Tracked via coherence accumulation |
| Model switching | Implicit (evidence ratio) | Explicit (rupture at C_m - C_{m'} > Ω) |
| Transition dynamics | Continuous (prior shrinkage) | Discontinuous (δ-function) |
| Post-switch beliefs | Carry forward posteriors | Regenerate from weighted history |
| Memory structure | Markovian (current posterior) | Non-Markovian (full trajectory) |
| Precision | Fixed or slowly adapting | Exponential in coherence: Π ∝ exp(C/Ω) |

---

## 5. The Three-Ball Paradigm: CRR Analysis

### 5.1 Setup

Following the original paper, consider three balls that can be:
- **Same color** (model m_S): All balls drawn from same distribution
- **Different colors** (model m_D): Balls drawn from different distributions

The agent observes ball colors sequentially and must infer which model applies.

### 5.2 Coherence Dynamics

Under model m_S (same):
$$\mathcal{C}_S(t) = \sum_{i=1}^{t} \frac{1}{2\sigma_S^2}(c_i - \bar{c})^2$$

where c_i is observed color and c̄ is the running mean.

Under model m_D (different):
$$\mathcal{C}_D(t) = \sum_{i=1}^{t} \frac{1}{2\sigma_D^2}(c_i - \mu_i)^2$$

where μ_i is the expected color for ball i.

### 5.3 Rupture: The "Aha Moment"

**Proposition 5.1.** The "aha moment" (insight that balls are different/same) corresponds to rupture:

$$t^* = \inf\{t : |\mathcal{C}_S(t) - \mathcal{C}_D(t)| > \Omega\}$$

**Key Prediction:** Unlike BMR which shows gradual evidence accumulation, CRR predicts:
1. Coherence gap grows gradually
2. At threshold, **discontinuous** transition occurs
3. Post-rupture, system **regenerates** beliefs using high-coherence observations

### 5.4 Active Reasoning: Which Ball to Observe?

Given the choice of which ball to observe next, the agent selects to maximize structure coherence gain:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{Q(c|\pi)}[|\Delta(\mathcal{C}_S - \mathcal{C}_D)|]$$

**Interpretation:** Choose observations that maximally accelerate rupture (resolve model uncertainty fastest).

**CRR-Specific Prediction:** Near the rupture threshold (C_S - C_D ≈ Ω), the agent should become **increasingly focused** on structure-discriminating observations, as precision grows exponentially:

$$\Pi_{\text{structure}}(t) = \frac{1}{\Omega} e^{|\mathcal{C}_S(t) - \mathcal{C}_D(t)|/\Omega}$$

---

## 6. Mathematical Details: CRR Model Reduction

### 6.1 CRR Model Evidence

**Theorem 6.1 (CRR Model Evidence).**

The evidence for model m given observations o_{1:t} is:

$$\log P(o_{1:t} | m) = -\mathcal{C}_m(t) - \frac{t}{2}\log(2\pi\sigma_m^2)$$

**Proof.** Under Gaussian likelihood:
$$\log P(o_{1:t} | m) = \sum_{i=1}^{t} \log \mathcal{N}(o_i; g_m(\hat{s}_i), \sigma_m^2)$$
$$= -\sum_{i=1}^{t} \frac{(o_i - g_m(\hat{s}_i))^2}{2\sigma_m^2} - \frac{t}{2}\log(2\pi\sigma_m^2)$$
$$= -\mathcal{C}_m(t) - \frac{t}{2}\log(2\pi\sigma_m^2)$$ ∎

### 6.2 CRR Bayes Factor

**Corollary 6.2.** The log Bayes factor is the coherence difference:

$$\log BF(m' : m) = \mathcal{C}_m(t) - \mathcal{C}_{m'}(t) + \frac{t}{2}\log\frac{\sigma_m^2}{\sigma_{m'}^2}$$

For equal noise (σ_m = σ_{m'}):
$$\log BF(m' : m) = \mathcal{C}_m(t) - \mathcal{C}_{m'}(t)$$

### 6.3 Rupture as Bayes Factor Threshold

**Theorem 6.3.** Rupture at C_m - C_{m'} > Ω is equivalent to:

$$BF(m' : m) > \frac{P(m)}{P(m')}$$

i.e., the posterior favors switching when coherence gap exceeds log prior odds.

---

## 7. Precision Dynamics and Attention

### 7.1 CRR Precision

In the CRR framework, precision is not fixed but evolves with coherence:

$$\boxed{\Pi_m(t) = \frac{1}{\Omega} e^{\mathcal{C}_m(t)/\Omega}}$$

### 7.2 Interpretation

| Regime | Precision | Behavior |
|--------|-----------|----------|
| C ≪ Ω | Π ≈ 1/Ω (low) | Exploratory, uncertain, prior-dominated |
| C → Ω | Π → e/Ω (growing) | Focusing, confident, data-driven |
| C = Ω | Π = e/Ω (maximum) | Pre-rupture, maximal attention |
| C > Ω | RUPTURE | Reset, regenerate, new model |

### 7.3 Attention Allocation

For structure learning, define structure-sensitive precision:

$$\Pi_{\text{structure}}(t) = \frac{1}{\Omega} e^{|\mathcal{C}_m(t) - \mathcal{C}_{m'}(t)|/\Omega}$$

**Prediction:** Attention to model-discriminating features should increase exponentially as the system approaches rupture threshold.

---

## 8. Comparison: Original vs. CRR Formulation

### 8.1 Expected Free Energy vs. Expected Coherence Gain

**Original:**
$$G(\pi) = -\underbrace{I(O; S | \pi)}_{\text{State Info}} - \underbrace{I(O; \Theta | \pi)}_{\text{Param Info}} - \underbrace{I(O; M | \pi)}_{\text{Structure Info}} + \underbrace{\mathbb{E}[-\log P(o)]}_{\text{Value}}$$

**CRR:**
$$E[\Delta\mathcal{C}](\pi) = \underbrace{\Delta\mathcal{C}_{\text{state}}(\pi)}_{\text{State Learning}} + \underbrace{\Delta\mathcal{C}_{\text{param}}(\pi)}_{\text{Param Learning}} + \underbrace{\Delta\mathcal{C}_{\text{structure}}(\pi)}_{\text{Structure Learning}}$$

**Key Difference:** CRR formulation directly tracks coherence accumulation toward rupture threshold, making the "distance to model switch" explicit.

### 8.2 Model Reduction

**Original BMR:**
- Compute reduced model evidence analytically from full model posterior
- Smooth, continuous model comparison
- Switch when evidence ratio crosses threshold

**CRR:**
- Track coherence accumulation for each model
- Explicit threshold: rupture when C_m - C_{m'} > Ω
- Discontinuous transition with regeneration from weighted history

### 8.3 Novel CRR Predictions

1. **Pre-rupture acceleration:** Precision (attention) should increase exponentially near model switch
2. **Discontinuous insight:** "Aha moments" should be genuinely discontinuous, not gradual
3. **Memory effects:** Post-switch beliefs should reflect coherence-weighted history, not just current posteriors
4. **Threshold sensitivity:** Small parameter changes near Ω should cause large behavioral changes

---

## 9. Simulation: Three-Ball Paradigm

### 9.1 Setup

```python
# Generative process
Models = {
    'Same': all balls from N(μ_common, σ²),
    'Different': ball i from N(μ_i, σ²)
}

# True generative process (unknown to agent)
true_model = 'Different'
true_μ = [0.2, 0.5, 0.8]  # Different means for each ball

# Agent's prior
P(Same) = P(Different) = 0.5
Ω = log(1) = 0  # Equal priors, so Ω = 0 (any evidence suffices)
# Or with prior bias: P(Same) = 0.75 → Ω = log(3) ≈ 1.1
```

### 9.2 CRR Dynamics

```python
def simulate_crr_three_ball(observations, Ω=1.0):
    C_same, C_diff = 0, 0
    history = []
    current_model = 'Same'  # Start with prior

    for t, o in enumerate(observations):
        # Coherence accumulation under each model
        if current_model == 'Same':
            μ_same = mean(observations[:t+1])
            C_same += 0.5 * (o - μ_same)**2 / σ²

        μ_diff = true_μ[t % 3]  # Ball-specific mean
        C_diff_increment = 0.5 * (o - μ_diff)**2 / σ²
        C_diff += C_diff_increment

        # Precision (attention) dynamics
        Π = (1/Ω) * exp(abs(C_same - C_diff) / Ω)

        history.append({
            't': t,
            'o': o,
            'C_same': C_same,
            'C_diff': C_diff,
            'gap': C_same - C_diff,
            'Π': Π
        })

        # Rupture check
        if C_same - C_diff > Ω:
            # RUPTURE: Switch to 'Different'
            t_star = t

            # Regeneration: weighted average of history
            weights = [exp(h['C_same']/Ω) for h in history]
            weights = [w/sum(weights) for w in weights]

            # Regenerated state estimate
            s_regen = sum(w * h['o'] for w, h in zip(weights, history))

            current_model = 'Different'
            print(f"RUPTURE at t={t_star}: Switched to 'Different'")
            print(f"Regenerated estimate: {s_regen:.3f}")

            # Partial reset
            C_diff = 0.3 * C_same
            C_same = 0

        elif C_diff - C_same > Ω:
            # RUPTURE: Switch to 'Same' (unlikely given true model)
            pass

    return history
```

### 9.3 Expected Results

**With true model = 'Different':**

1. Initially: C_same and C_diff both grow (learning)
2. Gradually: C_same grows faster (same-model makes worse predictions)
3. Gap widens: C_same - C_diff increases toward Ω
4. Pre-rupture: Π grows exponentially (increasing attention)
5. **Rupture:** At C_same - C_diff = Ω, discontinuous switch
6. Post-rupture: Regenerated beliefs from high-coherence observations

**Key CRR Signatures:**
- Exponential precision growth before insight
- Discontinuous transition (not gradual)
- Memory of high-coherence (well-predicted) observations

---

## 10. Theoretical Results

### 10.1 Sample Efficiency

**Theorem 10.1 (CRR Sample Efficiency).**

Under the CRR formulation, the expected number of observations to rupture is:

$$\mathbb{E}[t^*] = \frac{\Omega}{\mathbb{E}[\Delta\mathcal{C}_{\text{gap}}]}$$

where ΔC_gap = E[C_m - C_{m'}] per observation.

**Proof.** By linearity of expectation and the threshold condition. ∎

### 10.2 Active Reasoning Optimality

**Theorem 10.2 (Optimal Reasoning Policy).**

The policy maximizing expected structure coherence gain minimizes expected time to rupture:

$$\pi^* = \arg\max_\pi \Delta\mathcal{C}_{\text{structure}}(\pi) = \arg\min_\pi \mathbb{E}[t^* | \pi]$$

### 10.3 Regeneration Preserves Information

**Theorem 10.3 (Information Preservation).**

The regeneration operator preserves expected coherence:

$$\mathbb{E}[\mathcal{R}[\hat{s}]] = \frac{\int_0^{t^*} \hat{s}(\tau) \cdot e^{\mathcal{C}(\tau)/\Omega} d\tau}{\int_0^{t^*} e^{\mathcal{C}(\tau)/\Omega} d\tau}$$

is the coherence-weighted centroid of the historical trajectory.

---

## 11. Discussion: What CRR Adds

### 11.1 Explicit Discontinuity

The original BMR approach treats model switching as implicit—the agent's beliefs about models change continuously. CRR makes the switch explicit:

- **Before rupture:** Agent operates under model m
- **At rupture (t = t*):** Instantaneous transition δ(t - t*)
- **After rupture:** Agent operates under model m'

This captures the phenomenology of insight: sudden, not gradual.

### 11.2 Non-Markovian Memory

BMR uses only current posteriors. CRR regenerates from the full trajectory:

$$\mathcal{R}[\Phi] = \int_0^{t^*} \Phi(\tau) \cdot e^{\mathcal{C}(\tau)/\Omega} d\tau$$

High-coherence periods contribute more. This models:
- **Expertise:** Well-learned patterns persist through transitions
- **Selective memory:** Noisy periods are down-weighted
- **Consolidation:** Integration over time, not just current state

### 11.3 Precision Dynamics

CRR predicts precision grows exponentially with coherence:

$$\Pi(t) = \frac{1}{\Omega} e^{\mathcal{C}(t)/\Omega}$$

This means:
- Early learning: Diffuse, exploratory
- Near mastery: Sharp, confident
- Pre-rupture: Maximal attention to discriminating features

### 11.4 The Ω Parameter

CRR introduces Ω as a fundamental parameter:
- **Ω = log prior odds** for Bayesian model comparison
- **Ω = 1/π ≈ 0.318** as conjectured universal value
- **Ω controls rigidity:** Low Ω = frequent switches; high Ω = stable models

---

## 12. Conclusion

We have reformulated active inference and artificial reasoning through the CRR framework. The key contributions are:

1. **Expected Coherence Gain** replaces Expected Free Energy, with explicit structure learning term
2. **Rupture** provides discontinuous model switching at threshold C_m - C_{m'} > Ω
3. **Regeneration** reconstructs beliefs from coherence-weighted history
4. **Precision dynamics** show exponential growth toward rupture

The CRR formulation predicts qualitatively different dynamics:
- Insight is discontinuous (rupture), not gradual
- Memory is non-Markovian (trajectory-weighted)
- Attention grows exponentially before model switch
- Post-switch beliefs reflect integrated history, not just current posteriors

This provides a richer framework for understanding artificial reasoning that captures the phenomenology of insight, the role of history in belief formation, and the discontinuous nature of conceptual change.

---

## Appendix A: Equations Summary

### A.1 Core CRR Operators

| Operator | Equation |
|----------|----------|
| Coherence | $\mathcal{C}_m(t) = \sum_{i=1}^{t} \frac{1}{2}\varepsilon_i^T \Pi_m \varepsilon_i$ |
| Rupture condition | $t^* = \inf\{t : \mathcal{C}_m(t) - \mathcal{C}_{m'}(t) > \Omega\}$ |
| Rupture event | $\delta(t - t^*)$ |
| Regeneration | $\mathcal{R}[\Phi] = \frac{1}{Z}\int_0^{t^*} \Phi(\tau) e^{\mathcal{C}(\tau)/\Omega} d\tau$ |
| Precision | $\Pi(t) = \frac{1}{\Omega} e^{\mathcal{C}(t)/\Omega}$ |

### A.2 Expected Coherence Gain

| Component | Equation |
|-----------|----------|
| State | $\Delta\mathcal{C}_{\text{state}} = \mathbb{E}[\frac{1}{2}\varepsilon^T\Pi\varepsilon]$ |
| Parameter | $\Delta\mathcal{C}_{\text{param}} = D_{KL}[Q(\theta\|o) \| Q(\theta)]$ |
| Structure | $\Delta\mathcal{C}_{\text{structure}} = \mathbb{E}[\|\Delta(\mathcal{C}_m - \mathcal{C}_{m'})\|]$ |

### A.3 Correspondences

| FEP/BMR | CRR |
|---------|-----|
| Free Energy F | Coherence C = F₀ - F |
| Expected Free Energy G | Expected Coherence Gain E[ΔC] |
| Precision Π | Π = (1/Ω)exp(C/Ω) |
| Model evidence | exp(-C) |
| Posterior Q(m) | Threshold crossing |
| BMR reduction | Rupture + Regeneration |

---

**Document Status:** Complete CRR reformulation of Friston et al. (2025).

**Citation:**
```
CRR Reformulation of Active Inference and Artificial Reasoning.
Based on Friston et al. (2025) arXiv:2512.21129.
https://alexsabine.github.io/CRR/
```
