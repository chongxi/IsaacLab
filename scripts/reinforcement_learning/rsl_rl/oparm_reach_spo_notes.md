# OpenArm Reach SPO Notes

## 1) Current setup

- Train script: `scripts/reinforcement_learning/rsl_rl/oparm_reach_train_spo.py`
- Play script: `scripts/reinforcement_learning/rsl_rl/oparm_reach_play_spo.py`
- Action mode: **relative joint position delta** with **EMA smoothing**.
- Safety guards are enabled for NaN/Inf in observation/action/reward and update minibatches.

## 2) Actor → action pipeline (math)

Let observation be $o_t$.

1. Actor forward:

$$
h_1 = \phi(W_1 o_t + b_1),\quad
h_2 = \phi(W_2 h_1 + b_2),\quad
\mu_t = W_\mu h_2 + b_\mu
$$

2. Stochastic sampling (Gaussian):

$$
\sigma = \exp(\log\sigma),\quad
a_t \sim \mathcal{N}(\mu_t, \sigma^2)
$$

3. Optional policy output squashing/clipping:

$$
	ilde a_t = \operatorname{clip}(\tanh(a_t), -1, 1)
$$

4. Relative delta command:

$$
\Delta q_t = s \cdot \tilde a_t
$$

5. EMA smoothing (custom action term):

$$
\Delta q_t^{ema} = \alpha\,\Delta q_t + (1-\alpha)\,\Delta q_{t-1}^{ema}
$$

6. Applied target (relative position mode):

$$
q_t^{target} = q_t^{current} + \Delta q_t^{ema}
$$

## 3) Reward timing

- Reward is computed **every environment step**.
- PPO/SPO update happens after rollout buffer is filled (e.g., 24 steps).
- So the objective uses per-step rewards, then bootstrapped/aggregated for update.

## 4) Added reward shaping terms

- Position tracking error terms.
- Progress reward (`position_command_progress`): distance improvement vs previous step.
- Orientation term can be gated by position threshold (`orientation_command_error_when_close`).

## 5) Numerical stability protections

- Sanitize non-finite obs/action/reward.
- Force `done` for invalid env rows.
- Exclude invalid rows from gradient minibatches.
- Check finite logits/loss/grad norm and skip unsafe updates.
- Clamp policy `log_std` range.

## 6) Experimental observation mode

- Toggle in train/play: `USE_ERROR_OBS_EXPERIMENT`.
- Experimental config replaces command-style obs with explicit EE pose error obs.
- Keep baseline and experiment switchable for A/B runs.

## 7) Suggested next A/B checks

1. Run baseline (`USE_ERROR_OBS_EXPERIMENT=False`) with fixed seed.
2. Run error-obs experiment (`USE_ERROR_OBS_EXPERIMENT=True`) with same seed.
3. Compare:
	- EE position error trend,
	- orientation error near-goal,
	- episode return stability,
	- policy std growth/collapse signs.
