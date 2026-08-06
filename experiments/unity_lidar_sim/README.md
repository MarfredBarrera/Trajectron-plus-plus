# Unity-sim LiDAR prediction pipeline

Trajectron++ over ground-truth agent tracks exported from a Unity simulator: streaming
inference with per-timestep risk scoring, recorded to disk and visualized from the record.

All commands are run from this directory (`experiments/unity_lidar_sim/`), with the
`trajectron-gpu` conda environment; paths in the configs and in the arguments below are
relative to it.

```bash
# streaming: observe -> predict -> score, one timestep at a time
python src/unity_online.py --config configs/sweep_s1_rail_online.yaml --gpu 5 --style samples

# re-render a stored bundle (no model, no GPU)
python src/visualization/visualize.py --pred_file unity_out/<scene>/predictions.pkl

# build the risk figures from a stored run's recorded risk (no model, no GPU)
python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl --viz
# ...with the time-series agents in the intersection investigations' hues
python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl --junction_colors

# freeze a timestep and study the predicted distribution over the horizon (no model, no GPU)
python src/unity_online.py --config configs/intersection_probe.yaml --gpu 0 --style both
python src/investigations/gaussian_propagation.py   # in position space
python src/investigations/action_propagation.py     # in the decoder's control space
```

## Layout

    src/unity_online.py         the driver: observe -> predict -> score per timestep
    src/online_engine.py        OnlineEngine, wrapping BatchedOnlineTrajectron
    src/bundle.py               the on-disk prediction bundle: format + save/load
    src/repo_paths.py           puts trajectron/ and experiments/nuScenes/ on sys.path

    src/scene_conditioning/     log ingest, Environment/map construction, ego conditioning
    src/risk_eval/              proximity-risk metric, its overlay video, and its CLI
    src/visualization/          bundle rendering + the model-free re-render CLI
    src/investigations/         one-off studies of model behaviour (not part of the pipeline)

    configs/                    one YAML per scene/run (see configs/config.yaml for every key)
    unity_data/<scene>/         gt_agents.json, poses.csv, frames.csv
    unity_maps/                 drivable-area raster + its world->pixel registration
    unity_out/<scene>/          prediction bundles, risk CSVs, rendered frames and video

`src/` is the source root: the scripts under it are meant to be run directly
(`python src/unity_online.py ...`), which puts `src/` on `sys.path`, so modules import each
other as `bundle`, `risk_eval.proximity`, `visualization.traj_viz`. Nothing depends on the
working directory to resolve an import.

## Data flow

**Risk is scored online and recorded; visualizations are built from the record.** The
streaming driver scores each timestep at the moment it predicts it (`risk_eval.proximity`),
writes the per-sample result into the bundle (`risk_enter`) next to the predictions it
refers to, and exports the per-agent rows to CSV. Every figure — the overlay video, the
entering-trajectories-over-time plot — is then built by reading that file back, including
the ones the driver itself emits at the end of a run. So a plot made during the run and one
made from the stored bundle a week later come off the same code path and cannot disagree;
verified row-for-row against the CSV written inside the online loop.

The one deliberate exception is `--rescore`, which recomputes risk from the stored samples
at a different radius. That is for sweeping the threshold without re-running inference, and
it announces itself in the output so a rescored figure is never mistaken for the run's own
result.

## Investigations

`src/investigations/` holds focused studies of what the model does, kept apart from the
pipeline: they read a stored bundle, never run the model, and nothing in `src/` imports them.

### Gaussian propagation at the intersection

`configs/intersection_probe.yaml` + `src/investigations/gaussian_propagation.py`. Freezes a
timestep at the junction around (157, 71) — where the ego drives north while `-24740` turns
across it, `-23642` waits at it and `-24656` turns off it — and draws the whole predicted
distribution for that instant: every one of the 25 latent modes, at every horizon step, as a
1σ contour whose **colour is that mode's mixture weight** on a per-agent heat map (one
colorbar per agent), plus a track through each mode's own mean so a single mode can be
followed across the horizon. Translucent fills were tried first and don't work here for a
structural reason — overlapping patches composite, so the darkness at a point is the number
of ellipses stacked there rather than any one mode's weight, and two modes crossing produce
a third shape belonging to neither. Outlines on a weight heat map don't composite. Nothing
else is drawn — the map panel is the components, the history and the logged future, and
everything left off it is quantified in the companion diagnostics figure. Then repeats for a
series of timesteps. The probe config leaves `ego_conditioning: false`, so the ego is
not in the interaction graph and the distributions drawn are a function of the other agents
and the map alone.

Every run prints the junction roster and `--agents` switches agents on and off using any
column of it — the position, the short name, or the full id — so nothing has to be typed from
memory:

```
At the junction, nearest first  (--agents takes the #, the name, or the id):
  * #1     740    -24740     6.0 m away
  * #2     656    -24656     8.3 m away
  * #3     642    -23642    10.2 m away
    #4     304    -23304    11.1 m away
  * = drawn in this run (3 of 4)

python src/investigations/gaussian_propagation.py --agents 1,3       # by position
python src/investigations/gaussian_propagation.py --agents 740,642   # by short name
```

The roster is computed over the whole bundle rather than the frozen `--times`, and hue is
pinned to roster position, so #2 is the same vehicle in the same colour no matter which
timesteps or which subset of agents a run draws. (A leading `-` would be eaten by argparse,
so a full id is written without its sign: `--agents 24740`.)

Each frozen timestep also gets `propagation_t<t>.gif`: the same panel with the horizon as the
animated axis — one frame per horizon step, each step's contours added to the ones already
drawn and never dimmed or recoloured afterwards, and the logged future revealed marker by
marker alongside them. A mode is the same colour in frame 1 and frame 10, and the last frame
is the still panel. Simulation time stays frozen — what moves is how far ahead the model is
looking, which separates the contours the still panel has to draw in one go. `--anim
mp4|both|none` picks the format, `--anim_fps` / `--anim_hold` its pacing.

What it found on `sweep_s1_rail` (numbers in
`unity_out/intersection_probe/gaussian_propagation/propagation_summary.csv`):

* **The GMM and the sample fan are different objects, and the gap is measurable.**
  `Unicycle.integrate_distribution` rolls the *mean control* through the exact nonlinear
  dynamics but the covariance through an EKF linearization, while the sampling pass
  integrates each drawn control exactly. End-of-horizon RMS radius came out ×1.03–×1.99
  wider analytically than sampled, and the analytic mean sat up to 5.0 m off the sample mean.
  The gap is largest at low speed, where the unicycle linearization is worst — a vehicle
  stopped at the junction gets ellipses about twice as wide as any trajectory the sampler
  produces. The `max_a` / `max_heading_change` clamps in `Unicycle.dynamic` bound the samples
  but not the covariance recursion, which pushes the same way.
* **Most of the spread is blur within a mode, not disagreement between modes.** The
  between-mode share of variance at 5 s ran 1–42%, i.e. the 25 latent modes usually stay
  close together and the ellipses simply grow.
* **The latent posterior collapses when an agent commits to a manoeuvre.** Effective modes
  (perplexity of `p(z|scene)`) sat at 6–16 of 25 while agents ran straight, and dropped to
  2.0 for `-24740` at the timestep it began its turn and 1.4 for `-24656` at the timestep it
  began its own.
* **Uncertainty is mostly longitudinal**, i.e. about how far the agent gets, not about
  whether it leaves its lane — except mid-turn, where the two become comparable.
* **Mid-turn is where the mean is worst.** For `-24740` at t=48, halfway through its turn,
  the mean path continues along the instantaneous heading and overshoots the corner: 22.4 m
  from the logged future at 5 s, 3.7σ out.

### The same mixture in action space

`src/investigations/action_propagation.py`. Trajectron++ does not predict positions: for a
vehicle it predicts a Gaussian mixture over **controls** — heading rate dφ [rad/s] and
acceleration a [m/s²] — per horizon step, and `Unicycle.integrate_distribution` turns that
into the position ellipses everything else draws. This study freezes the same timesteps and
draws the mixture on that side of the integration: one control-plane panel per agent, the
mode's contour at every horizon step plus its track, the same hue per agent and the same
`--agents` handles as the position study, plus `action_t<t>.gif` unrolling it along the
horizon.

Two things are only visible here. First, what the model is uncertain *about*: horizontal
spread is doubt about the turn, vertical spread is doubt about the throttle, neither scaled by
how fast the agent happens to be moving. Second, the dynamic limits, which exist in this space
and nowhere else — `Unicycle.dynamic` clamps every control to `a ∈ [-5, 4]`,
`dφ ∈ [-0.7, 0.7]`. Those limits **are** the axes of the panel, so "is this control
executable" is read off the frame, and the mixture mass outside them is in the panel title and
the CSV.

One agent and the most likely mode by default (`--max_agents`, `--agents`, `--modes n`):
25 modes × 10 steps inside a box two units across is unreadable however it is coloured, and
what it adds is mostly modes the model barely believes. The default agent is roster #2, the
amber one. Colour is the mode's weight on the same per-agent heat map as the position study —
the same mode is the same colour in both figures, on both sides of the dynamics — and horizon
time is stroke weight plus the labelled ends of the mode's track.

What it found on `sweep_s1_rail` (numbers in
`unity_out/intersection_probe/action_propagation/action_summary.csv`):

* **A large share of the control mixture is outside what the integrator can execute.** Mass
  outside the clamp box ran 1–44% at the end of the horizon and peaked at 75% (`-23642`,
  t=56, first step). The clamp bounds the samples but not the covariance recursion, so that
  mass is inside the position ellipses and inside no trajectory the model can produce — the
  mechanism behind the ×1.03–×1.99 analytic-vs-sampled gap the position study measured.
* **The mixture mean itself leaves the box at the first horizon step** for two of the
  eighteen agent-timesteps studied, so this is not only a tail effect.
* **Uncertainty is far larger at the first horizon step than later** — σ(a) up to 9.3 m/s²
  and σ(dφ) up to 2.9 rad/s at h=1, settling to 0.5–2.4 and 0.2–1.7 by 5 s. The decoder's
  immediate control is its least confident output, which the position view hides because one
  step of a wild control barely moves a vehicle.

Producing it needed the control mixture to be kept: `ctrl_mus` / `ctrl_covs` per node and the
model's `dynamic` config in `meta` (see `src/bundle.py`). `MultimodalGenerativeCVAE.p_y_xz`
grew an `output_control_dist` flag — off by default, so nothing else changes — that returns
`a_dist` alongside `y_dist`, and the batched online model threads it into the `output_dists`
pass it was already running. **Bundles written before this lack `ctrl_*`; re-run the probe to
use this study.**
