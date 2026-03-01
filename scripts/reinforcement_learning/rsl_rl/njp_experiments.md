# Neural Jacobian Policy (NJP) Experiments

## Task
OpenArm bimanual reach (Isaac-Reach-OpenArm-Bi-v0), **7 DOF per arm**, **14 actions** total.
EMA relative joint position actions: `scale=0.5, alpha=0.7, use_zero_offset=True`.

## Best Result
**Score: ~8.25 mean reward** — NJP with trig features, nonlinear key (ELU), error gate, trained with MDPO.

---

## Observation & Action Layout

### Observation (`OpenArmReachEnvCfgErrObs`, 54 dims total)

| Index     | Term               | Dims | Description                                    |
|-----------|--------------------|------|------------------------------------------------|
| `0:7`     | `left_joint_pos`   | 7    | Left arm relative joint positions (l_q)        |
| `7:14`    | `right_joint_pos`  | 7    | Right arm relative joint positions (r_q)       |
| `14:21`   | `left_joint_vel`   | 7    | Left arm relative joint velocities (l_dq)      |
| `21:28`   | `right_joint_vel`  | 7    | Right arm relative joint velocities (r_dq)     |
| `28:34`   | `left_pose_command` | 6   | Left EE pose error: pos_err(3) + axis_angle(3) |
| `34:40`   | `right_pose_command`| 6   | Right EE pose error: pos_err(3) + axis_angle(3)|
| `40:47`   | `left_actions`     | 7    | Left arm previous actions (l_prev_a)           |
| `47:54`   | `right_actions`    | 7    | Right arm previous actions (r_prev_a)          |

The NJP **actor uses only the first 40 dims** (q, dq, error). Previous actions (40:54) are **unused by the actor** but available to the **critic** (which takes full 54-dim obs via MLP).

### Action (14 dims)
EMA-smoothed relative joint position deltas. Each action dim corresponds to one joint.
`target_pos += alpha * (scale * action)`, clipped to joint limits.

| Index   | Dims | Description       |
|---------|------|-------------------|
| `0:7`   | 7    | Left arm joints   |
| `7:14`  | 7    | Right arm joints  |

---

## NJP Forward Pass: Step-by-Step Dimension Trace

Notation: `B` = batch, `n` = 7 (joints/arm), `e` = 6 (error dims/arm), `d` = 64 (attn_dim)

### 1. Split Obs → per-arm components
```
obs: [B, 54]
  ├── left_q:    obs[:, 0:7]    → [B, 7]
  ├── right_q:   obs[:, 7:14]   → [B, 7]
  ├── left_dq:   obs[:, 14:21]  → [B, 7]
  ├── right_dq:  obs[:, 21:28]  → [B, 7]
  ├── left_err:  obs[:, 28:34]  → [B, 6]   ← task-space error
  ├── right_err: obs[:, 34:40]  → [B, 6]   ← task-space error
  └── (prev_actions: obs[:, 40:54] → ignored by actor)
```

### 2. Query: per-joint trig features → local embedding
Each joint gets its own 4D trig token, projected to `d`-dim:
```
left_joint_tokens = stack([sin(l_q), cos(l_q), sin(l_dq), cos(l_dq)], dim=-1)
                  → [B, 7, 4]          (7 joints × 4 trig features)

left_query = Linear(4 → 64)(left_joint_tokens)
           → [B, 7, 64]               (per-joint embedding)

left_query += joint_id_embed[1, 7, 64]  (broadcast: learnable per-joint bias)
            → [B, 7, 64]              = Q
```

### 3. Key: global arm trig features → error-dim key vectors
All joints' trig features concatenated into one flat vector, projected into `e` key vectors:
```
left_state = cat([sin(l_q), cos(l_q), sin(l_dq), cos(l_dq)], dim=-1)
           → [B, 28]                  (7×4 = 28, all joints' trig features flat)

left_key = arm_key(left_state)
         = Linear(28 → 64) → ELU() → Linear(64 → 384)
         → [B, 384]                   (384 = 6 × 64 = e × d)

left_key = left_key.view(B, 6, 64)
         → [B, 6, 64]                = K  (6 key vectors, one per error dim)
```

### 4. Pseudo-Jacobian: A = Q @ K^T
Attention logits give a `(joints × error_dims)` matrix — the learned pseudo-Jacobian:
```
left_A = einsum("bnd,bmd->bnm", Q[B,7,64], K[B,6,64]) × (1/√64)
       → [B, 7, 6]                    = A  (pseudo-Jacobian: 7 joints × 6 error dims)
```
Each entry `A[i,j]` = how much error dimension `j` drives joint `i`.

### 5. Action = A @ error
Matrix-vector product maps 6D task-space error to 7D joint-space action:
```
left_u = bmm(A[B,7,6], left_err[B,6,1])
       → [B, 7, 1] → squeeze → [B, 7]   (per-joint action for left arm)
```

### 6. Error-magnitude gating (optional, `gate=True`)
Scalar gate modulates action based on error magnitude:
```
||left_err|| → [B, 1]                    (L2 norm of 6D error)
left_gate = sigmoid(Linear(1 → 1)(||left_err||))
          → [B, 1]                       (scalar ∈ (0, 1))

left_u = left_gate × left_u              (broadcast multiply)
       → [B, 7]
```

### 7. Combine both arms
```
action = cat([left_u[B,7], right_u[B,7]], dim=-1)
       → [B, 14]                         = final action output
```

### Summary diagram
```
obs [B, 54]
  │
  ├── per-joint trig: [B,7,4] ──Linear(4→64)──+joint_id──→ Q [B, 7, 64]
  │                                                              │
  ├── global trig:    [B, 28] ──Linear(28→64)──ELU──Linear(64→384)──reshape──→ K [B, 6, 64]
  │                                                              │
  │                                            A = Q @ Kᵀ / √d → [B, 7, 6]
  │                                                              │
  ├── error:          [B, 6]  ────────────────── A @ err ──────→ [B, 7]
  │                     │                                        │
  │                     └── ||err|| → gate ∈ (0,1) ──×──────────→ [B, 7]
  │                                                              │
  └── (×2 for both arms) ──────────────── cat ─────────────────→ [B, 14] = action
```

### Critic (separate path)
Standard MLP on **full** 54-dim observation (including prev_actions):
```
critic = Sequential(Linear(54→64), ELU, Linear(64→64), ELU, Linear(64→1))
       → [B, 1]  (value estimate)
```

---

## Architecture Evolution

### V0: Baseline MLP (`ActorCritic_MLP`)
- Standard 2-layer MLP actor [64, 64] + tanh output
- `obs[B,54] → hidden → hidden → action_mean[B,14]`
- Works but treats all obs dimensions uniformly, no structure

### V1: Raw Q/K Attention NJP (first version)
- **Query**: per-joint `(q_i, dq_i)` → `Linear(2, 64)` + `joint_id_embed`
- **Key**: `Linear(14, 384)` from raw `(q_all, dq_all)` → reshape to 6 key vectors
- `A = Q @ K^T` → pseudo-Jacobian `[B, 7, 6]`
- `action = A @ error`
- Learns but limited: raw q/dq values mean network can only learn linear combinations of joint angles

### V2: Trig Features (breakthrough)
- **Query**: per-joint `(sin(q_i), cos(q_i), sin(dq_i), cos(dq_i))` → `Linear(4, 64)`
- **Key**: `(sin(q_all), cos(q_all), sin(dq_all), cos(dq_all))` → `Linear(28, 64)` → `Linear(64, 384)`
- Trig features provide nonlinearity: sin/cos capture periodic joint space structure
- Can represent linear combinations of sin(q_i), cos(q_i) — matches real Jacobian structure
- Significant improvement over raw features

### V3: Nonlinear Key Projection (ELU between layers)
- arm_key: `Linear(28, 64) → ELU() → Linear(64, 384)`
- The ELU enables learning **cross-joint products** like `sin(q1)*cos(q2)` which appear in real Jacobians
- Without activation: two stacked linears collapse to one affine transform — only linear combinations of trig features
- With activation: network can approximate products of trig features via the universal approximation property

**Status across scripts**:
| Script | act_fn() in arm_key |
|--------|-------------------|
| `simple_train_mdpo.py` (best) | **active** (ELU) |
| `oparm_reach_train_spo.py` | commented out |
| play scripts | commented out |

Note: the play scripts should match whichever training script produced the checkpoint being loaded.

### V4: Error-Magnitude Gating (current best)
- Added `gate = sigmoid(Linear(||error||, 1))` per arm
- `action = gate(||e||) * A(q,dq) @ error`
- When error is small → gate can suppress residual action (prevents oscillation near target)
- When error is large → gate opens up for aggressive correction
- `use_gate = True` in current best config

---

## Algorithm Comparison on NJP

| Algorithm | Best Reward | Notes |
|-----------|------------|-------|
| SPO       | ~7.5       | Quadratic penalty surrogate, KL early stopping, obs sanitization |
| MDPO      | **~8.25**  | Two-policy mutual distillation, fastest learning |
| PPO       | ~6-7       | Standard clipped surrogate (baseline) |

---

## Key Design Decisions

### What worked
1. **Error as explicit input** (not buried in obs): `action = A(q,dq) @ error` gives the network clean structure
2. **Trig features**: `sin(q), cos(q)` naturally capture joint space periodicity
3. **Per-joint query tokens**: each joint gets its own attention query (local → global)
4. **Global key**: arm-level state info shared across all 6 error dimensions
5. **Joint ID embedding**: learnable per-joint bias in query (breaks symmetry between joints)
6. **Nonlinear key projection (ELU)**: enables cross-joint product terms in the Jacobian
7. **Error-magnitude gating**: adaptive action scaling based on error size
8. **MDPO algorithm**: dual-policy distillation gives better exploration
9. **Staggered env resets**: `episode_length_buf = randint(0, max_ep_len)`

### What didn't work / was removed
- `softmax` over error dims in A matrix (broke learning — over-constrains the Jacobian rows)
- `a_scale` multiplicative scaling of A (still in code but effectively unused, set to ones)
- `tanh` on actor output for NJP (unnecessary, EMA action already clips)
- `prev_actions` in actor input (in obs but unused — actor only needs q, dq, error)

### What's debatable
- **act_fn() in arm_key**: active in MDPO (best), commented out in SPO. Worth running controlled ablation.

---

## Hyperparameters (MDPO, current best)

```python
num_envs = 4096            # split 2048/2048 between two policies
num_steps_per_env = 24
max_iterations = 1500
actor_hidden_dims = [64, 64]   # attn_dim = 64
critic_hidden_dims = [64, 64]
activation = "elu"
init_noise_std = 1.0
learning_rate = 1e-2
schedule = "exponential"       # decay_rate=5.0
num_learning_epochs = 8
num_mini_batches = 4
clip_param = 0.2
gamma = 0.99
lam = 0.95
distill_coef = 0.02
entropy_coef = 0.001
```

---

## Training Scripts

| Script | Algorithm | Policy Classes |
|--------|-----------|---------------|
| `simple_train_mdpo.py` | MDPO | MLP, NJP, NJP_Local |
| `oparm_reach_train_spo.py` | SPO | MLP, NJP, NJP_Local |
| `simple_train.py` | PPO (OnPolicyRunner) | rsl_rl ActorCritic |
| `oparm_reach_play_mdpo.py` | — (play) | MLP, NJP, NJP_Local |
| `oparm_reach_play_spo.py` | — (play) | MLP, NJP, NJP_Local |

---

## Future Directions

### 1. Adaptive Bias Term (Feedforward Control Pathway)
**`action = A(q,dq) @ error + b(q,dq)`**

Classical robotics: `u = J(q) @ Kp @ e + g(q)` where `g(q)` is gravity compensation.
The bias term `b(q)` would:
- Separate geometric tracking (A @ error) from dynamics compensation (bias)
- Enable transfer across payload changes (only fine-tune b, keep A frozen)
- Mirror the feedforward torque in computed-torque control

Implementation: small MLP `b_net: (sin(q), cos(q)) → R^{n_joints}` or reuse the query features.

### 2. Temporal Context for Load Adaptation
Current NJP is feedforward — can't detect payload changes from a single timestep.
Options:
- **GRU/LSTM on query tokens**: accumulate evidence of unexpected dq response over time
- **Action history**: `prev_action` is already in obs (indices 40:54) but unused — could feed to actor
- **Residual integrator**: `b(t) = b(t-1) + alpha * (dq_expected - dq_actual)` — adaptive bias that accumulates steady-state error

### 3. Multi-Head Attention
Multiple heads in the Q/K attention could capture different aspects of the Jacobian:
- Head 1: position-error to joint-velocity mapping
- Head 2: velocity damping
- Head 3: cross-coupling between arms

### 4. Learned Scaling per A Entry
Currently `a_scale` is unused (ones). Could learn per-entry scaling of the Jacobian to handle different joint/error-dim magnitudes without relying on the attention logits alone.

### 5. Curriculum / Domain Randomization
- Gradually increase target reach distance
- Randomize link masses to force robust gravity compensation
- Add external perturbation forces to test robustness
