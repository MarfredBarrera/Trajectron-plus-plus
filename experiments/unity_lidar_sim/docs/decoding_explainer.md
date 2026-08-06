# Trajectron++ decoding & trajectory prediction — source brief for slides

Audience for the slides: someone who knows CVAEs and vehicle dynamics but has not read this
codebase. Everything below was verified against the code at the file:line references given;
they are stable anchors for anyone who wants to check a claim.

Config values referenced throughout come from `config/nuScenes.json`:
`N = 1`, `K = 25`, `GMM_components = 1`, `pred_state = {position: [x, y]}`,
VEHICLE dynamics = `Unicycle`, PEDESTRIAN dynamics = `SingleIntegrator`.

---

## 0. One-paragraph version

Trajectron++ is a conditional VAE with a **discrete** latent. The encoder compresses history,
neighbors and map into a condition vector `x`. A 25-way categorical latent `z ~ p(z|x)` selects
a behavioral mode. A GRU decoder, conditioned on `[z, x]`, autoregressively emits one bivariate
Gaussian **per horizon step** — over *controls*, not positions. Those controls are then pushed
through an explicit dynamics model to get positions, in two parallel ways: analytically
(mean + EKF-propagated covariance → the Gaussian blobs) and by Monte Carlo (sampled controls
integrated through the true nonlinear dynamics → the sample fan). Training never sees a control
label; the control distribution is learned purely by backpropagating position NLL through the
dynamics Jacobians.

---

## 1. Pipeline overview

```
history ─┐
neighbors ├─> encoders ─> x  (condition vector)
map ─────┘                │
                          ├─> p_z_x  ──> 25-way categorical over z
                          │
              [z, x] ─────┴─> GRU decoder, ph steps
                                  │
                                  └─> per-step bivariate Gaussian over CONTROLS
                                          │
                                          ├─ integrate_distribution ─> analytic position GMM
                                          └─ integrate_samples ──────> position sample fan
```

Entry point: `MultimodalGenerativeCVAE.predict` — [mgcvae.py:1084](../../../trajectron/model/mgcvae.py#L1084).
Core of the decode: `p_y_xz` — [mgcvae.py:779](../../../trajectron/model/mgcvae.py#L779).

**Slide suggestion:** this diagram is the spine of the deck. Build it up incrementally across
slides 1–5 rather than showing it all at once.

---

## 2. Encoder → the condition vector `x`

`obtain_encoded_tensors` — [mgcvae.py:359](../../../trajectron/model/mgcvae.py#L359).

`x` is a concatenation of:
- node history LSTM encoding
- aggregated neighbor/edge encodings (per edge type, then attention-combined)
- map CNN encoding
- encoded robot future, if `incl_robot_node`

Everything downstream is conditioned on `x`. It is computed **once per prediction**; in the
online engine the recurrent parts are cached across simulation timesteps by `RecurrentStateStore`
in `trajectron/model/online/batched_online_mgcvae.py`.

---

## 3. The discrete latent `z` — where multimodality lives

`p_z_x` — [mgcvae.py:741](../../../trajectron/model/mgcvae.py#L741) → `dist_from_h` —
[discrete_latent.py:20](../../../trajectron/model/components/discrete_latent.py#L20).

`z` is `td.OneHotCategorical` over `N x K = 1 x 25` — 25 discrete behavioral modes, with logits
a learned function of `x`. Not a Gaussian latent.

Logits are mean-zero-shifted at
[discrete_latent.py:22](../../../trajectron/model/components/discrete_latent.py#L22). This does
not change the distribution (softmax is invariant to a constant offset); it is for numerical
conditioning and for the `z_logit_clip` annealing used during training.

How `z` is obtained at predict time — `sample_p`,
[discrete_latent.py:37](../../../trajectron/model/components/discrete_latent.py#L37):

| flags | behavior | `num_components` |
|---|---|---|
| `full_dist=True` | enumerate all 25 one-hot `z`, decode every one | 25 |
| `all_z_sep=True` | enumerate all 25, but do not merge into a mixture | 1 |
| `z_mode=True` | argmax `z` only | 1 |
| none of the above | draw `z ~ p(z\|x)`, independently per sample and per agent | 1 |

**Gotcha worth a slide:** these are an `if / elif` chain with `full_dist` checked first
([discrete_latent.py:39-49](../../../trajectron/model/components/discrete_latent.py#L39-L49)).
Passing `z_mode=True` together with `full_dist=True` **silently ignores `z_mode`**. The
"most likely" pass in the online engine does exactly this
([online_engine.py:216-217](../src/online_engine.py#L216)) — see §8 for what it actually computes.

---

## 4. The decoder rollout → a per-step GMM over **controls**

### 4a. What the two output dimensions mean

Four linear heads, built at
[mgcvae.py:215-226](../../../trajectron/model/mgcvae.py#L215-L226), each of width
`GMM_components * pred_state_length` = `1 * 2` = 2:

| head | output | constraint |
|---|---|---|
| `proj_to_GMM_mus` | `mu = (mu_1, mu_2)` | unbounded |
| `proj_to_GMM_log_sigmas` | `log sigma`, exponentiated at [gmm2d.py:44](../../../trajectron/model/components/gmm2d.py#L44) | positive by construction |
| `proj_to_GMM_corrs` | `rho`, `tanh`-squashed | in (-1, 1) |
| `proj_to_GMM_log_pis` | 1 logit | effectively unused (see §5) |

**The single most confusing thing in the codebase, and worth its own slide:** those 2 numbers
are *not positions*, even though `pred_state` in the config literally says `position: [x, y]`.
Their meaning is set entirely by the dynamics module:

- `SingleIntegrator` reads them as **velocity** `(vx, vy)` —
  [single_integrator.py:22](../../../trajectron/model/dynamics/single_integrator.py#L22)
- `Unicycle` reads them as **turn rate and acceleration** `(dphi, a)` —
  [unicycle.py:42-43](../../../trajectron/model/dynamics/unicycle.py#L42-L43)

`pred_state` only fixes the *dimension count* (2) and the shape of the label tensor. Two configs
with identical `pred_state` produce heads meaning entirely different physical quantities.

**Units are physical, not standardized.** Initial conditions come from raw `inputs`, not
`inputs_st` ([mgcvae.py:419-420](../../../trajectron/model/mgcvae.py#L419-L420)); training NLL is
explicitly on unstandardized labels ([mgcvae.py:993](../../../trajectron/model/mgcvae.py#L993)).
Only encoder/decoder *inputs* are standardized. So `mu_a` really is in m/s^2, directly comparable
to the config's `max_a: 4`.

Because `rho` is a free parameter, each step's control distribution is a **full correlated
bivariate Gaussian** — the model can express coupling like "hard turns come with braking".
The Cholesky factor is built by hand at
[gmm2d.py:49-53](../../../trajectron/model/components/gmm2d.py#L49-L53) rather than via
`torch.cholesky`.

### 4b. The autoregressive loop

```
a_0    = state_action(n_s_t0)                      # mgcvae.py:811 — bootstrap control
input_ = [z, x, a_0]
for j in range(ph):
    h_state = cell(input_, state)
    gmm_j   = GMM2D(*project_to_GMM_params(h_state))   # 1-component control Gaussian
    a_t     = gmm_j.rsample()   or   gmm_j.mode()      # mgcvae.py:827-830
    record (log_pi, mu, log_sigma, rho)_j
    input_  = [z, x, a_t]                              # <-- the draw feeds back
```

**The subtlety worth a slide:** because `a_t` feeds back, step `j+1`'s Gaussian parameters depend
on the *realized draw* at step `j`. So the recorded parameter sequence is not a marginal
distribution over the horizon — it describes the distribution *along one particular realized
control path*. This is why:

- `gmm_mode=True` (feedback = the mean) is required for the analytic output to be self-consistent;
- the compounding of feedback noise is something the analytic covariance structurally cannot
  represent, since it linearizes around a single fixed mean path.

Note `rsample` (reparameterized, `mu + L*xi`,
[gmm2d.py:76](../../../trajectron/model/components/gmm2d.py#L76)) rather than `sample` is required
here — a plain sample would block gradients through the feedback path during training.

### 4c. When `full_dist=True`, the 25 modes are *parallel independent rollouts*

They are not a mixture at decode time. `z` is stacked and `x` repeated at
[mgcvae.py:800-801](../../../trajectron/model/mgcvae.py#L800-L801), giving an effective batch of
`num_samples x 25 x bs` rows, ordered sample-major → mode → agent. The GRU cell advances all rows
in lockstep but there is **no cross-row interaction** — a GRU cell has none.

```
mode c=0:  h0_j -> (mu0, sigma0, rho0)_j -> a0_j --+ feeds back into row 0 only
mode c=1:  h1_j -> (mu1, sigma1, rho1)_j -> a1_j --+ feeds back into row 1 only
  ...                                                 (no coupling between rows)
mode c=24: h24_j -> ...                  -> a24_j --+
```

`rsample()` in the loop draws **one `a_t` per row** — 25 separate controls, each fed back only
into the row it came from. This is a common misconception worth an explicit "not this / this"
slide.

---

## 5. Assembling the 25-component GMM

Only *after* the loop. The reshape block at
[mgcvae.py:842-854](../../../trajectron/model/mgcvae.py#L842-L854) moves the mode axis out of the
batch dimension and into the component dimension:

```
mu_t: [num_samples * 25 * bs, 2]
  .reshape(num_samples, 25, bs, 2)     # split the flattened batch
  .permute(0, 2, 1, 3)                 # -> [num_samples, bs, 25, 2]
  .reshape(-1, 2 * 25)                 # 25 modes now ride in the feature axis
```

Mixture weights are **not** taken from `proj_to_GMM_log_pis`. When `num_components > 1`,
[mgcvae.py:832-836](../../../trajectron/model/mgcvae.py#L832-L836) overwrites them with the
latent's logits — `p_dist.logits` at predict time, `q_dist.logits` at train time. So the
components of the exported mixture **are the 25 latent modes**, weighted by `p(z|x)`. The
per-step head's `log_pis` is dead weight with `GMM_components = 1`.

When `num_components == 1` the weights are set to `ones_like(...)`
([mgcvae.py:838-840](../../../trajectron/model/mgcvae.py#L838-L840)) — a degenerate single-component
GMM. See §9 for why that is the correct choice and not a shortcut.

### Shapes — worth a slide on its own

`a_dist` is **not** one 2D GMM. It is `num_samples x bs x ph` separate 25-component 2D GMMs
packed into one batched object. The `ph` axis is *inside* the distribution:

| tensor | shape |
|---|---|
| `log_pis` | `[ns, bs, ph, 25]` |
| `mus` | `[ns, bs, ph, 25, 2]` |
| `sigmas` | `[ns, bs, ph, 25, 2]` |
| `L` (Cholesky) | `[ns, bs, ph, 25, 2, 2]` |

In `td.Distribution` terms: `[ns, bs, ph]` is the batch shape, `[2]` the event shape. Every
`(sample, agent, horizon step)` triple indexes its own independent 25-component bivariate GMM.
`mu` and `sigma` differ across `t` (they came from `h_state` at step `j`); `log_pis` does **not**
(same latent logits appended every step).

One `rsample()` call vectorizes `ns * bs * ph` draws and returns `[ns, bs, ph, 2]` — one 2D
control per horizon step. Slice `k` along the sample axis is a complete `ph`-step control
sequence, i.e. one trajectory.

**Key framing for the audience:** you never build a horizon *out of* a single 2D distribution.
The horizon was already unrolled by the GRU, which produced `ph` sets of parameters. Sampling
just realizes all `ph` of them at once. The recurrence builds the time structure; the GMM only
supplies the per-step noise.

---

## 6. Dynamics integration — controls become positions

Two parallel outputs from
[mgcvae.py:873-884](../../../trajectron/model/mgcvae.py#L873-L884). This deserves a side-by-side
slide, because the difference between them is the source of most practical confusion.

### 6a. Analytic — `integrate_distribution`

**No sampling occurs.** The `ph` controls that get applied are the per-step **means**.

`SingleIntegrator` ([single_integrator.py:24](../../../trajectron/model/dynamics/single_integrator.py#L24)):
means by cumsum, covariance by the exact linear Kalman recursion `Sigma <- F Sigma F^T`. **Exact.**

`Unicycle` ([unicycle.py:214](../../../trajectron/model/dynamics/unicycle.py#L214)): two coupled
recursions, initial state shared by all 25 modes, `Sigma_0 = 0` (no initial-state uncertainty):

```
for t = 1 .. ph:
    F_t   = compute_jacobian(x_{t-1}, u_bar_t)          # d f / d x at the mean, [4,4]
    G_t   = compute_control_jacobian(x_{t-1}, u_bar_t)  # d f / d u at the mean, [4,2]
    Sigma_t = F_t Sigma_{t-1} F_t^T  +  G_t Sigma^u_t G_t^T     # covariance: LINEARIZED
    x_t     = dynamic(x_{t-1}, u_bar_t)                         # mean: TRUE nonlinear f
```

Points to make:
- The controls are consumed **sequentially, one step at a time, with state carried forward** —
  exactly like the sample path. There is no "apply all at once". The only difference from the
  sample path is *which* control is fed in (the mean, not a draw) plus the covariance recursion.
- All 25 modes run in parallel by broadcasting: `x_0` is `[4, bs, 1]` and the control slice is
  `[2, ns, bs, 25]`, so after the first `dynamic()` call `x` is `[4, ns, bs, 25]`.
- State is 4D `(x, y, phi, v)`; only the `[:2, :2]` block of `Sigma` is exported.
- **Blobs grow because `Sigma_0 = 0` and each step adds `G_t Sigma^u_t G_t^T`** — fresh control
  noise mapped into state space — then carries the accumulation forward through `F Sigma F^T`.
  Uncertainty is only ever injected, never removed: this is the *predict* half of an EKF, with no
  measurement update. The `dt` and `dt^2/2` entries in `G` are what convert accel / turn-rate
  variance into position variance.
- `F` and `G` are evaluated at the **mean** state and **mean** control. That is the linearization,
  and it is the whole approximation.
- Mixture weights pass through unchanged into
  `GMM2D.from_log_pis_mus_cov_mats` ([unicycle.py:257](../../../trajectron/model/dynamics/unicycle.py#L257)).

`Unicycle` also applies a small learned initial-heading correction,
`phi_0 = atan2(v_0) + tanh(p0_model([x, phi_0]))` —
[unicycle.py:96](../../../trajectron/model/dynamics/unicycle.py#L96).

### 6b. Monte Carlo — `integrate_samples`

`a_sample = a_dist.rsample()` then roll through the **true nonlinear** `dynamic()` —
[unicycle.py:101-107](../../../trajectron/model/dynamics/unicycle.py#L101-L107). Also picks up the
accel / turn-rate clamps enforced inside `dynamic()`
([unicycle.py:50-53](../../../trajectron/model/dynamics/unicycle.py#L50-L53)).

**Non-obvious and worth stating explicitly:** the exported samples are drawn from the *assembled*
`a_dist`, not from the `a_t` values used during the rollout. Those were consumed by the GRU and
discarded. So the returned trajectory differs from the control path the network actually
conditioned on — same `mu, sigma, rho` per step, different realization of the noise.

### 6c. Why the two disagree — the practical headline

The analytic path linearizes around the mean trajectory and integrates *means*; the sample path
passes through exact nonlinear dynamics *and* the input clamps, and carries the compounded
feedback noise. On this project's data the analytic ellipses run up to ~2x wider than the sample
fan at low speed, where turn-rate variance produces strongly curved (non-affine) position spread
and the linearization is worst.

Consequence for the deck: **use samples for numbers, blobs for illustration.**

---

## 7. Training — no control labels exist anywhere

`train_loss` — [mgcvae.py:951](../../../trajectron/model/mgcvae.py#L951).

1. All 25 `z` are enumerated (`sample_q`,
   [discrete_latent.py:31](../../../trajectron/model/components/discrete_latent.py#L31)), so
   `num_components = 25`.
2. `p_y_xz` returns only the **analytic** `y_dist` in TRAIN mode.
3. [mgcvae.py:944](../../../trajectron/model/mgcvae.py#L944) scores ground-truth future
   **positions**; `GMM2D.log_prob` does `logsumexp(log_pis + component_log_p)` over the component
   axis ([gmm2d.py:113](../../../trajectron/model/components/gmm2d.py#L113)), computing

   ```
   log SUM_c  q(z_c | x, y) * N( y_t ; mu[c,t], Sigma[c,t] )
   ```

   i.e. the reconstruction term **marginalizes over `z` inside the log**, rather than being
   `E_q[ log p(y|x,z) ]`. This is why assembling the mixture is not optional during training.
4. Summed over the horizon, then `ELBO = log_likelihood - kl_weight * KL + mutual_inf_p`
   ([mgcvae.py:1003](../../../trajectron/model/mgcvae.py#L1003)).

**The conceptual punchline:** gradients reach `mu, sigma, rho` of the control heads *only* by
backpropagating position NLL through `integrate_distribution` — through the unicycle Jacobians
`F` and `G`. The "controls" are a learned latent representation whose physical meaning is imposed
entirely by the dynamics model. Nothing forces `mu_a` to be a plausible acceleration except that
implausible values integrate to bad positions.

Second-order but important: **the model was fit under the EKF covariance**, since training
likelihood uses the analytic `y_dist`. The `sigma` the heads learned is the one that made the
*linearized* position covariance match the data. The nonlinear sample fan is a different object
that no loss term ever saw — which is the mechanism behind the width gap in §6c, not a bug in
either path.

---

## 8. The flag matrix

Slide-ready. All flags live on `predict` / `sample_model`
([mgcvae.py:1084](../../../trajectron/model/mgcvae.py#L1084),
[batched_online_trajectron.py:416](../../../trajectron/model/online/batched_online_trajectron.py#L416)).

| pass | flags | what you get |
|---|---|---|
| sample fan | `z_mode=F, gmm_mode=F, full_dist=F` | `ns` trajectories; `z` drawn per sample, control noise at every step |
| Gaussian blobs | `z_mode=F, gmm_mode=T, full_dist=T` | 25 deterministic mean rollouts + analytic covariances, weighted by `p(z\|x)` |
| most likely | `z_mode=T, gmm_mode=T, full_dist=T` | see below — `z_mode` is ignored here |

These three are `DECODE_PASSES` in [online_engine.py:51-55](../src/online_engine.py#L51).

**What the "most likely" pass actually does:** `full_dist` wins the `if/elif`, so `z_mode` is a
no-op. It decodes all 25 modes with mean feedback, assembles the 25-component control GMM, then
calls `a_dist.mode()` ([mgcvae.py:880](../../../trajectron/model/mgcvae.py#L880)), which for
`components > 1` runs a meshgrid argmax search
([gmm2d.py:126-146](../../../trajectron/model/components/gmm2d.py#L126-L146)) — hence the
`assert samp == 1` and `num_samples=1`.

Two caveats: the search is **per horizon step independently**, so the result is a sequence of
per-step mixture modes — *not* the mode of the joint trajectory distribution, and not necessarily
consistent with any single latent mode. And it is the slow path: a Python double loop over
agents x horizon with a 0.01-resolution grid each time.

---

## 9. Ancestral sampling — why `full_dist=False` already samples the mixture

A likely audience question, worth pre-empting on a slide.

The model defines

```
p(y|x) = SUM_c  p(z_c | x) * p(y | x, z_c)
```

Drawing `c ~ Cat(p(z|x))` and then `y ~ p(y|x, z_c)` produces draws distributed exactly as
`p(y|x)` — that is what ancestral sampling means. So the `full_dist=False` path **is** sampling
from the 25-mode GMM. It simply never materializes the mixture, because it does not need to: the
mode is *realized* rather than enumerated, and the weights are carried by **sample frequency**
rather than explicit `log_pis`.

That is why `log_pis` is overwritten with ones in that branch — reweighting samples that are
already `p(z|x)`-distributed would double-count. It also means anything downstream that counts
samples (e.g. the proximity-risk metric) is a correct estimator with no reweighting needed.

Cost comparison: ancestral is 1 decoder rollout per sample; explicit enumeration is 25.

**Where the distinction genuinely bites:** ancestral sampling gives no coverage guarantee for
low-probability modes. If a mode with `p(z_c|x) = 0.02` is the one heading toward the ego, then
with 20 samples you miss it entirely about 2/3 of the time. Enumerate-and-weight has no such
variance. This is a real limitation to state honestly on a slide, not a footnote.

---

## 10. Pitfalls and dead code — good "things we learned" slides

### 10a. `p_traj` vs `p_prod` — a live but uncalled code path

Combining `full_dist=True` with `gmm_mode=False` would call `a_dist.rsample()` on a 25-component
mixture. **No call site in the repo does this** (verified: every `full_dist=True` site sets
`gmm_mode=True`; every `gmm_mode=False` site sets `full_dist=False`).

If it were called, it would not sample what you want. `pis_cat_dist = td.Categorical(logits=log_pis)`
with `log_pis` of shape `[ns, bs, ph, 25]` has `batch_shape = [ns, bs, ph]`, so
[gmm2d.py:84](../../../trajectron/model/components/gmm2d.py#L84) draws an **independent component
index at every horizon step**. The two distributions:

```
mixture over trajectories (wanted):
    p_traj(u_1..u_ph) = SUM_c  pi_c * PROD_t  N(u_t; mu[c,t], Sigma[c,t])

product of per-step marginals (what rsample gives):
    p_prod(u_1..u_ph) = PROD_t [ SUM_c  pi_c * N(u_t; mu[c,t], Sigma[c,t]) ]
```

A mixture of products versus a product of mixtures — sum and product swapped. **Every per-step
marginal is identical**; the joints are not. `p_traj` couples the steps through the shared `c`;
`p_prod` has no coupling at all.

Concrete failure, good as a slide graphic: mode A = turn left (`mu_dphi = +0.5`, `pi = 0.5`),
mode B = turn right (`mu_dphi = -0.5`, `pi = 0.5`).

```
p_traj draws:   A A A A A A  -> left turn
                B B B B B B  -> right turn

p_prod draws:   A B A A B A  -> turns cancel under integration -> goes STRAIGHT
```

Straight is the one behavior the model assigned almost no probability to. `p_prod` manufactures
mass in the gap between modes — the classic mode-averaging artifact. With `ph = 12` it draws from
`25^12 ~ 6e16` control-sequence combinations, of which the 25 coherent ones are a vanishing
fraction.

Two honest qualifications:
- `full_dist=False` is **unaffected** — with one component the categorical is degenerate and
  per-step reselection is a no-op. `p_traj` and `p_prod` coincide when `K = 1`. Every sampling call
  site in the repo is in that regime, so this has never corrupted a number here.
- `p_traj` is not the full story either. Even with `c` fixed, the stored parameters give a product
  of per-step Gaussians — `GMM2D` retains no cross-time covariance. What rescues it is that
  integration is a cumulative sum, so i.i.d. control noise still produces a strongly correlated
  *position* path. The within-mode approximation is benign; the across-mode one is not, because
  mode identity determines *where the trajectory goes*, not merely how it jitters.

### 10b. `GMM_components` must stay 1

The reshapes at [mgcvae.py:842-854](../../../trajectron/model/mgcvae.py#L842-L854) do
`mu_t.reshape(num_samples, num_components, -1, 2)` where `num_components` is the *latent* count.
With `GMM_components > 1` the head's extra components get silently folded into the batch axis
instead of being treated as mixture components. It does not raise — it produces a wrong-sized
batch. Any genuine per-step control mixture would require rewriting that block first.

### 10c. Local modification: unicycle dynamic limits

Upstream stored `hyperparams['dynamic'][node_type]['limits']` on the base class and never read it,
so the declared bounds were dead config — the decoder sampled `dphi` and `a` from unbounded
Gaussians and the rollout integrated `v <- v + a*dt` with nothing limiting either. One draw from
the Gaussian tail put a sample on a permanent runaway. Measured on `sweep_s1_rail`, ~0.1% of
samples exceeded 40 m/s and the worst reached 390 m/s, while median speed stayed correct and flat
across the horizon. Now clamped inside `dynamic()` —
[unicycle.py:45-53](../../../trajectron/model/dynamics/unicycle.py#L45-L53), with rationale at
[unicycle.py:13-25](../../../trajectron/model/dynamics/unicycle.py#L13-L25).

---

## 11. Suggested slide sequence

1. Title / problem framing: multimodal trajectory prediction, why a discrete latent
2. Pipeline diagram (build up over the next slides)
3. Encoder → `x`
4. The discrete latent: 25 modes, `p(z|x)`
5. **The decoder emits controls, not positions** — the `pred_state` trap
6. The per-step bivariate Gaussian + the autoregressive feedback subtlety
7. `full_dist=True`: 25 parallel independent rollouts (the "not this / this" slide)
8. Assembling the mixture; shapes table; `ph` lives inside the distribution
9. Dynamics integration: analytic (EKF) vs Monte Carlo, side by side
10. Why the blobs are wider than the fan — and which to trust
11. Training: no control labels, gradients through the Jacobians
12. The flag matrix / the three decode passes
13. Ancestral sampling: `full_dist=False` already samples the mixture; the rare-mode caveat
14. Pitfalls: `p_traj` vs `p_prod`, `GMM_components`, `z_mode` precedence
15. Takeaways

### Figures worth drawing
- The 25-parallel-rollouts picture (§4c) — the single highest-value diagram.
- Blob vs fan overlay at low speed, showing the ~2x width gap (§6c).
- The `A B A A B A` mode-hopping cartoon (§10a) — turns cancel, path goes straight.
- Covariance growth over the horizon, `Sigma_0 = 0` accumulating `G Sigma^u G^T` (§6a).

### Things to get right if the deck is challenged
- The decoder output is controls; `pred_state: position` is a naming artifact.
- The 25 modes never interact during the rollout; the mixture is an assembly step afterwards.
- The analytic covariance is EKF-linearized for `Unicycle` and exact only for `SingleIntegrator`.
- `full_dist=False` is *not* a single-mode approximation — it is ancestral sampling of the full
  mixture.
- Training likelihood uses the analytic distribution, not the samples.
