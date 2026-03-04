# CPG + Reflex RL: Architecture Analysis and Plan

## Current Architecture

```
obs(48) ──────────────────────────────┐
    │                                  │
    │ cmd = obs[:, 9:12]               │
    ▼                                  ▼
BatchedQuadrupedDecoder (~104 params)  reflex_mlp [128,128] → 12
    │                                  │
    ├─► dn(B,4)  ── gates CPG         0.3 * tanh(output)
    ├─► sht(B,4) ── sets frequency     │
    ├─► W_eff(B,4,3,2) ── readout      │
    │                                  │
    ▼                                  │
4× MANC Euler step (differentiable)    │
    │                                  │
    ▼                                  │
cpg_offsets = W_eff @ r[:,:,:2]        │
    │                                  │
    └──────────── + ───────────────────┘
                  │
            action_mean (B, 12)
```

### Obs layout (48 dims)

| Slice     | Dims | Content          |
|-----------|------|------------------|
| `[0:3]`   | 3    | base_lin_vel     |
| `[3:6]`   | 3    | base_ang_vel     |
| `[6:9]`   | 3    | projected_gravity |
| `[9:12]`  | 3    | commands (vx, vy, omega) |
| `[12:24]` | 12   | joint_pos        |
| `[24:36]` | 12   | joint_vel        |
| `[36:48]` | 12   | prev_actions     |

### rsl_rl library

Custom fork at `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/`,
modified by Fan Yang (ETH Zurich, 2025). Contains MDPO (Meta Distilled Policy Optimization):
dual-policy PPO where two actor-critics learn on interleaved env subsets and distill into each
other via symmetric KL divergence. Not standard leggedrobotics rsl_rl.

### MANC CPG (per leg)

3-neuron oscillator (E1, E2, I1) from Drosophila connectome. Produces limit-cycle oscillation
via adaptation currents. Per-leg modulation:
- **DN** (`dn`): multiplicatively gates recurrent connections (amplitude/on-off)
- **5-HT** (`sht`): modulates time constants (frequency)
- **W_eff**: per-leg readout matrix mapping E1, E2 firing rates to 3 joint offsets

All CPG dynamics are differentiable (Euler integration, no detach). Gradients flow via BPTT
through the rollout. CPG state `(cpg_x, cpg_a)` is stored in LSTM `(h, c)` slots for
rsl_rl's recurrent rollout storage.

---

## The Problem

**If we restrict the MLP too much:** CPG alone can't learn good stepping — the robot
doesn't really step.

**If we don't restrict the MLP:** The MLP learns the cyclic stepping itself, and the CPG
becomes irrelevant.

This is the classic **credit assignment / lazy learner** problem in additive hierarchical
architectures.

---

## Root Cause Analysis

### 1. Gradient asymmetry (~30x)

The MLP reflex path has **direct gradients**:
```
loss → action → 0.3*tanh(mlp(obs)) → mlp weights
```
One clean backprop step.

The CPG path has **attenuated gradients**:
```
loss → action → W_eff @ r → r (Euler steps) → dn, sht → decoder weights
```
Each Euler step attenuates by `dt/τ ≈ 0.01/0.3 ≈ 0.033`. Even with `cpg_lr_scale=10x`,
the effective gradient reaching `w_dn`, `w_sht` is ~30x weaker than what the MLP gets.

Note: the `W_eff` → `cpg_offsets` path does NOT go through Euler steps (it just reads out
`r` after the step), so `W_delta` gets decent gradients. The problem is specifically with
the dynamics parameters `w_dn`, `b_dn`, `w_sht`, `b_sht`.

### 2. Information leakage in MLP input

The MLP sees the full 48-dim obs including:
- `prev_actions(12)` — directly carries CPG rhythm from last timestep
- `joint_pos(12)` — reflects the robot's current rhythmic motion

Even bounded to [-0.3, 0.3], the MLP can reconstruct cyclic patterns by "echoing" these
rhythmic inputs. With `prev_actions → current_correction → next prev_actions`, the MLP
becomes an implicit single-step RNN with a temporal feedback loop. It doesn't need internal
state to produce oscillations.

### 3. Additive composition = credit ambiguity

With `action = cpg_offsets + corrections`, the optimizer has two paths to reduce loss.
Since MLP gradients are ~30x stronger, Adam naturally routes credit through the MLP path.
The CPG offsets drift toward zero because the optimizer has no incentive to keep them alive
when the MLP can do the job.

### 4. No architectural enforcement

Nothing in the architecture *forces* the MLP to be a reflex. It has the capacity, the input
information, and the gradient advantage to become the primary motor controller. The `0.3`
scaling is a soft constraint that only limits magnitude per timestep, not functional role.

---

## Literature Review

### Bellegarda et al., CPG-RL (2022)
**Key idea:** RL policy outputs **CPG parameters** (amplitude, frequency per leg), NOT joint
corrections. All motor output goes through the CPG. No additive bypass path.

The action space is `[amplitude_i, frequency_i]` for each leg. The CPG (Hopf oscillator)
converts these to joint trajectories. The MLP can only influence the robot through the CPG —
it modulates the rhythm, it cannot replace it.

Found that explicit inter-oscillator couplings and LSTM memory improve energy efficiency and
sim-to-real robustness. Deployed on Unitree A1.

### NCAP — Neural Circuit Architectural Priors (2024)
**Key idea:** Freeze the CPG (Rhythm Generation layer) entirely. Train only Pattern Formation
(linear, 48 params) and Afferent Feedback (linear, 44 params) with constrained weight signs.
Total: **92 learnable parameters** vs 79,372 for MLP baseline.

Results: matched MLP final performance, vastly better sim-to-real transfer. MLP "falls
immediately" on real robot due to erratic actions; NCAP walks stably without any sim-to-real
techniques.

Critical insight: *"the specific structure of NCAP matters"* — not just fewer parameters,
but the right architectural constraints that prevent pathological solutions.

### Visual CPG-RL (2023)
Extended CPG-RL with exteroceptive vision. LSTM + CPG coupling + explicit inter-oscillator
connections all matter for navigation. CPG provides structured prior; LSTM adds
memory-enabled corrections to CPG parameters.

### HRL-CPG (2025)
Hierarchical RL: high-level policy modulates CPG parameters, low-level CPG produces joint
targets. Clean separation of concerns. Shows faster convergence than flat RL on varied
terrain.

### Common theme across all papers

**The MLP never has a direct additive path to the joints.** It always goes through the CPG.
This is the architectural invariant that prevents bypass.

---

## Plan: Fix the Architecture

### Option A — MLP modulates CPG parameters (recommended)

Make the reflex MLP output **residual CPG parameters** instead of joint-space corrections.
All motor output still flows through the CPG dynamics.

```
obs(reduced) → reflex_mlp → [Δdn(4), Δsht(4), Δreadout_scale(4)]
                                │          │           │
                                ▼          ▼           ▼
cmd → decoder → dn ──+──► dn_final ──► CPG Euler step
                 sht ─+──► sht_final ──►     │
                 W_eff ────────────────► readout × (1 + Δreadout_scale)
                                                │
                                          action_mean (B, 12)
```

**What changes in `_compute_action_mean`:**
```python
# Decoder gives base parameters from command
dn_base, sht_base, W_eff = self.decoder(cmd)

# Reflex MLP gives residual modulations from proprioception
reflex_out = self.reflex_mlp(reflex_obs)  # (B, 12)
delta_dn   = 0.2 * torch.tanh(reflex_out[:, 0:4])    # (B, 4)
delta_sht  = 0.2 * torch.tanh(reflex_out[:, 4:8])    # (B, 4)
delta_gain = 0.3 * torch.tanh(reflex_out[:, 8:12])   # (B, 4) per-leg readout scale

# Combine: reflex modulates CPG, never bypasses it
dn_final  = (dn_base + delta_dn).clamp(0, 1)
sht_final = (sht_base + delta_sht).clamp(min=0.3)

# CPG step with modulated parameters
new_r, new_cpg_x, new_cpg_a = self._cpg_euler_step(cpg_x, cpg_a, dn_final, sht_final)

# Readout with per-leg gain modulation
cpg_offsets = torch.einsum("bljn,bln->blj", W_eff, new_r[:, :, :2])  # (B, 4, 3)
gain = (1.0 + delta_gain).unsqueeze(-1)  # (B, 4, 1)
cpg_offsets = (cpg_offsets * gain).reshape(-1, 12)

action_mean = cpg_offsets  # NO additive bypass
```

**Pros:**
- CPG is guaranteed to be the rhythmic backbone — MLP can't bypass it
- MLP output is interpretable: which leg speeds up, which gets louder, etc.
- Gradient flows through both paths naturally (no asymmetry problem)
- Biologically accurate: sensory afferents modulate CPG interneurons, they don't have
  their own motor output

**Cons:**
- MLP can only modulate the rhythm, not produce qualitatively different corrections
  (e.g., a fast recovery step that breaks the cycle)
- 12-dim MLP output might not be expressive enough for rough terrain

### Option B — MLP modulates CPG + small residual for emergencies

Hybrid of Option A with a very small residual path for non-rhythmic corrections:

```python
# Same as Option A for CPG modulation
action_mean = cpg_offsets_modulated  # CPG backbone

# Tiny residual for stumble recovery only
# Gated by proprioceptive surprise (high joint_vel deviation)
surprise = (joint_vel.abs() - 1.0).clamp(min=0).sum(dim=-1, keepdim=True) / 12.0
residual = 0.1 * torch.tanh(self.residual_mlp(reflex_obs)) * surprise.clamp(max=1.0)
action_mean = action_mean + residual
```

The residual is gated by proprioceptive surprise — it only activates when something
unexpected happens (stumble, push). During normal walking, `surprise ≈ 0` and the
residual vanishes.

### Option C — Information bottleneck (minimal change, quick test)

Keep the additive architecture but strip rhythmic information from MLP input:

```python
# MLP only sees non-rhythmic proprioception
reflex_obs = torch.cat([
    obs[:, 3:6],    # angular velocity (3) — balance
    obs[:, 6:9],    # projected gravity (3) — tilt
    obs[:, 9:12],   # commands (3) — what we want
    obs[:, 24:36],  # joint velocities (12) — stumble detection
], dim=-1)  # 21 dims, NO joint_pos, NO prev_actions

corrections = 0.3 * torch.tanh(self.reflex_mlp(reflex_obs))
action_mean = cpg_offsets + corrections
```

Without `prev_actions` and `joint_pos`, the MLP can't reconstruct the rhythm. It can only
react to instantaneous proprioceptive signals (angular velocity, gravity, joint velocity
spikes). This is the quickest change to test.

### Option D — Two-phase curriculum

```python
# Phase 1 (iter 0-500): CPG only, MLP frozen
# Phase 2 (iter 500+): unfreeze MLP, gradually increase scale
if iteration < 500:
    for p in self.reflex_mlp.parameters():
        p.requires_grad = False
    corrections = torch.zeros_like(cpg_offsets)
else:
    for p in self.reflex_mlp.parameters():
        p.requires_grad = True
    reflex_scale = min(0.3, 0.3 * (iteration - 500) / 500)
    corrections = reflex_scale * torch.tanh(self.reflex_mlp(obs))
```

Forces CPG to learn a working gait first, then lets MLP add corrections. The risk is that
once MLP unfreezes, it can still gradually take over if the additive bypass exists.

---

## Recommended Implementation Order

1. **Start with Option C** (information bottleneck) — minimal code change, quick to test.
   Remove `prev_actions` and `joint_pos` from MLP input. This alone may fix the bypass.

2. **If CPG still struggles, do Option A** (MLP modulates CPG parameters) — the
   architecturally clean solution. More code change but guarantees the right behavior.

3. **If terrain adaptability is poor with Option A, add Option B's surprise-gated residual**
   — a small emergency escape hatch that only activates during perturbations.

4. **Option D (curriculum) can be combined with any of the above** as an additional
   training stabilizer, but should not be the primary fix since it doesn't address the
   fundamental architectural issue.

---

## References

- Bellegarda et al. (2022). *CPG-RL: Learning Central Pattern Generators for Quadruped
  Locomotion.* https://arxiv.org/abs/2211.00458
- Bellegarda et al. (2023). *Visual CPG-RL: Learning Central Pattern Generators for
  Visually-Guided Quadruped Locomotion.* https://arxiv.org/abs/2212.14400
- Luo et al. (2024). *Neural Circuit Architectural Priors for Quadruped Locomotion.*
  https://arxiv.org/abs/2410.07174
- Scientific Reports (2025). *Hierarchical RL with CPG for quadruped locomotion on varied
  terrains.* https://www.nature.com/articles/s41598-025-94163-2
- Scientific Reports (2025). *Bio-inspired neural networks with CPG for multi-skill
  locomotion.* https://www.nature.com/articles/s41598-025-94408-0
