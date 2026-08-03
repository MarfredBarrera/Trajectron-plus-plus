"""Streaming Trajectron++ inference for the Unity-sim pipeline.

`OnlineEngine` drives `BatchedOnlineTrajectron` over a scene one timestep at a time and
emits frame records in the on-disk bundle format (see src/bundle.py), so visualization and
risk_eval apply unchanged.

The offline alternative would hand `Trajectron.predict()` a finished log and let it re-read
each agent's whole history window at every timestep. This engine instead hands the model one
observation per agent per timestep and keeps the recurrent state between them, which is how a
perception -> prediction stack actually runs and what makes per-timestep risk scoring
meaningful rather than a post-hoc sweep over a completed log.

Streaming semantics
-------------------
* Only observations up to and including timestep t reach the model. Each agent's `history`
  is read back out of the model's own buffers, not from the log.
* Agents are discovered as they appear and aged out as they leave. Nothing about the agent
  set is known up front.
* `future` (the agent's logged ground truth) is attached to each record for evaluation and
  rendering only. It is read from the log *after* the model has predicted and never reaches
  the model.
"""
import json
import time

import numpy as np
import torch

import repo_paths  # noqa: F401  (sys.path side effect: trajectron/)
from model.online import BatchedOnlineTrajectron
from model.model_registrar import ModelRegistrar
from environment import Environment, Scene


# Floor on warm-up length. This is a convention, not a data constraint: unity_scene's
# build_environment precomputes velocity/acceleration for the whole track with derivative_of,
# so the state vector is already finite at t=0. Warming up over one timestep keeps the first
# predicted timestep at t=2, matching the batch driver's start, and gives the interaction
# graph one step of motion before anything is scored.
MIN_WARMUP_TIMESTEPS = 1

# Debugging/experimentation kill-switches: which decoder passes `step()` may run off its
# single encoder pass. Flip any of these to False to force-skip that pass regardless of what
# the caller asked for; the fields it would have filled are written as empty arrays so records
# keep the bundle format and the rest of the pipeline still loads them (it just has nothing to
# draw or score for the skipped pass).
#
# These are the *ceiling*, not the decision: which passes actually run is chosen per run by
# the `need_samples` / `need_dists` constructor arguments, which unity_online.py derives from
# --style. Leave them True unless you are deliberately bisecting a decoder pass -- a False
# here silently empties that field in the bundle no matter how the run was invoked.
#   DECODE_SAMPLES     -- Monte Carlo samples; the proximity-risk metric is defined on these.
#   DECODE_MOST_LIKELY -- single z_mode/gmm_mode trajectory ('ml').
#   DECODE_DISTS       -- analytic per-mode GMM ('dist_mus'/'dist_covs'/'dist_pis').
DECODE_SAMPLES = True
DECODE_MOST_LIKELY = True
DECODE_DISTS = True


def load_online_hyperparams(model_dir, config_name='config.json'):
    """Read a trained model's config.json and apply the eval-time overrides.

    Mirrors experiments/nuScenes/helper.load_model. Unlike `OnlineTrajectron`, the batched
    online model imposes no requirement on `dynamic_edges`.
    """
    with open(f'{model_dir}/{config_name}', 'r') as f:
        hyperparams = json.load(f)

    hyperparams['map_enc_dropout'] = 0.0
    hyperparams.setdefault('incl_robot_node', False)
    return hyperparams


def _online_environment(env):
    """The single-scene Environment the online model runs against.

    It carries the log scene's `dt` (dynamics integration) and map rasters plus the env-level
    attention radii / standardization, but deliberately no nodes: observations are streamed
    in one timestep at a time, so there is exactly one source of agent data and the model
    never has the option of reading ahead in it.
    """
    log_scene = env.scenes[0]
    online_scene = Scene(timesteps=log_scene.timesteps, map=log_scene.map, dt=log_scene.dt,
                         name=log_scene.name)
    online_scene.robot = log_scene.robot
    return Environment(node_type_list=env.node_type_list,
                       standardization=env.standardization,
                       scenes=[online_scene],
                       attention_radius=env.attention_radius,
                       robot_type=env.robot_type)


class OnlineEngine(object):
    """Drives BatchedOnlineTrajectron over a log one timestep at a time.

    :param unity_scene: a prepared `unity_scene.UnityScene` (supplies env, scene, ego poses).
    :param model_dir: directory holding config.json + model_registrar-<ts>.pt.
    :param model_ts: checkpoint iteration to load.
    :param device: torch device string ('cpu' or 'cuda:N').
    :param ph: prediction horizon in timesteps.
    :param num_samples: Monte Carlo samples drawn per agent per timestep.
    :param warmup_timesteps: observations streamed into the encoders before the first
        prediction; the first predicted timestep is `warmup_timesteps + 1`.
    :param min_history_timesteps: observations an agent needs before it is predicted for
        (1 = from its second observation).
    :param need_samples: when False the Monte Carlo pass is skipped and each record's
        `samples` is stored empty -- only useful for Gaussian-only rendering, since the
        proximity-risk metric is defined on samples.
    :param need_dists: when False the analytic per-mode GMM pass is skipped and each record's
        `dist_mus`/`dist_covs`/`dist_pis` are stored empty -- fine for sample-fan rendering,
        but a bundle written this way cannot be re-rendered with --style gaussian later.
    """

    def __init__(self, unity_scene, model_dir, model_ts, device, ph, num_samples,
                 warmup_timesteps=MIN_WARMUP_TIMESTEPS, min_history_timesteps=1,
                 need_samples=True, need_dists=True):
        if warmup_timesteps < MIN_WARMUP_TIMESTEPS:
            raise SystemExit(f'warmup_timesteps must be >= {MIN_WARMUP_TIMESTEPS} '
                             f'(see MIN_WARMUP_TIMESTEPS)')

        self.scene_data = unity_scene
        self.log_scene = unity_scene.scene
        self.robot_type = unity_scene.env.robot_type
        self.device = device
        self.ph = ph
        self.num_samples = num_samples
        self.need_samples = need_samples and DECODE_SAMPLES
        self.need_dists = need_dists and DECODE_DISTS
        self.init_timestep = warmup_timesteps

        self.hyperparams = load_online_hyperparams(model_dir)
        self.state = self.hyperparams['state']

        # The checkpoint and the scene have to agree about the ego: an incl_robot_node model
        # has an ego-plan encoder and expects a plan every timestep, and a model without one
        # has no weights to consume it. Checking here turns a silent shape mismatch deep in
        # the forward pass into a message that says which side to change.
        self.incl_robot_node = self.hyperparams['incl_robot_node']
        if self.incl_robot_node and unity_scene.robot is None:
            raise SystemExit(f'{model_dir} was trained with incl_robot_node: true -- set '
                             f'ego_conditioning: true in the config so the ego is built into '
                             f'the scene as the robot node')
        if unity_scene.robot is not None and not self.incl_robot_node:
            raise SystemExit(f'ego_conditioning is set, but {model_dir}/config.json has '
                             f'incl_robot_node: false -- that checkpoint has no ego-plan '
                             f'encoder. Use a checkpoint trained with incl_robot_node: true.')

        model_registrar = ModelRegistrar(model_dir, device)
        model_registrar.load_models(model_ts)
        self.model = BatchedOnlineTrajectron(model_registrar, self.hyperparams, device,
                                             min_history_timesteps=min_history_timesteps)
        self.model.set_environment(_online_environment(unity_scene.env))

        # (type name, id) -> log Node. `Scene.get_clipped_input_dict` hands the model a
        # single-timestep *copy* of each node, so the full logged track has to be looked up
        # here. Used only to read out ground-truth futures after the model has predicted;
        # never consulted by the model itself.
        self._log_nodes = {(n.type.name, n.id): n for n in self.log_scene.nodes}

        self._warm_up()

    def _warm_up(self):
        """Stream timesteps 0..init_timestep through the encoders without decoding.

        Every agent present in that window ends up with its LSTMs already holding context by
        the first predicted timestep, instead of starting from a zero hidden state while the
        batch driver encodes a full history window at the same point.
        """
        for t in range(self.init_timestep + 1):
            self._observe(t)

    def _observe(self, t):
        """Advance every agent's encoders by timestep `t`'s observation, without decoding."""
        self.model.incremental_forward(self.log_scene.get_clipped_input_dict(t, self.state),
                                       robot_plan=self._ego_plan(t))

    def _ego_plan(self, t):
        """The ego's planned state over [t, t+ph], or None when not conditioning on it.

        With `ego_path_mode: logged` this is the ego's logged future -- treated as its plan,
        which is what a vehicle predicting for its own benefit actually has.
        """
        if not self.incl_robot_node:
            return None
        return self.scene_data.ego_plan_state(t, self.ph, self.state[self.robot_type])

    @property
    def first_timestep(self):
        """First timestep this engine can predict at."""
        return self.init_timestep + 1

    # ------------------------------------------------------------------ #
    # One streaming step                                                  #
    # ------------------------------------------------------------------ #
    def step(self, t):
        """Feed timestep `t`'s observations to the model and predict from it.

        Returns a frame record in the bundle format (see src/bundle.py) with an added
        'runtime_s' field, or None if no agent was predictable at `t`.
        """
        start = time.time()
        self._observe(t)

        # Up to three decoder passes off one encoder pass, each mirroring the corresponding
        # offline predict() call flag for flag -- minus their
        # redundant re-encoding. Each runs only if this run needs its output (see the
        # need_* arguments) and its DECODE_* kill-switch is on.
        samples = most_likely = dists = None
        if self.need_samples:
            samples = self.model.sample_model(self.ph, self.num_samples,
                                              z_mode=False, gmm_mode=False, full_dist=False)
        if DECODE_MOST_LIKELY:
            most_likely = self.model.sample_model(self.ph, 1, z_mode=True, gmm_mode=True,
                                                  full_dist=True)
        if self.need_dists:
            # Deterministic per-latent-mode GMM (mean/covariance propagated analytically
            # through the dynamics model), for the Gaussian-blob viz -- no sample statistics.
            _, dists = self.model.sample_model(self.ph, 1, z_mode=False, gmm_mode=True,
                                               full_dist=True, output_dists=True)

        runtime = time.time() - start

        # Any enabled pass keys its output by the same predictable-agent set, so the first one
        # that ran defines the frame's agents. With every pass off there is nothing to record.
        predicted = next((d for d in (most_likely, samples, dists) if d is not None), None)
        if not predicted:
            return None

        nodes = [self._node_record(node, t, samples, most_likely, dists)
                 for node in sorted(predicted, key=lambda n: n.id)]
        return {'t': int(t), 'nodes': nodes, 'runtime_s': runtime}

    def _node_record(self, node, t, samples, most_likely, dists):
        """Serializable per-agent arrays, all in scene-local coords (bundle format).

        A pass that was toggled off leaves its fields as zero-length arrays of the right rank.
        """
        empty = np.empty((0,), dtype=np.float32)
        rec = {
            'id': node.id, 'type': node.type.name,
            'history': self.model.history(node),                             # (H, 2)
            'future': self._logged_future_of(node, t),                       # (F, 2)
            'samples': empty.reshape(0, self.ph, 2),                         # (S, ph, 2)
            'ml': empty.reshape(0, 2),                                       # (ph, 2)
            'dist_mus': empty.reshape(0, 0, 2),                              # (ph, K, 2)
            'dist_covs': empty.reshape(0, 0, 2, 2),                          # (ph, K, 2, 2)
            'dist_pis': empty.reshape(0, 0),                                 # (ph, K)
        }
        if samples is not None:
            rec['samples'] = np.asarray(samples[node][0], dtype=np.float32)
        if most_likely is not None:
            rec['ml'] = np.asarray(most_likely[node][0, 0], dtype=np.float32)
        if dists is not None:
            dist = dists[node]
            rec['dist_mus'] = np.asarray(dist['mus'][0, 0], dtype=np.float32)
            rec['dist_covs'] = np.asarray(dist['covs'][0, 0], dtype=np.float32)
            rec['dist_pis'] = np.asarray(dist['pis'][0, 0], dtype=np.float32)
        return rec

    def _logged_future_of(self, node, t):
        """The agent's logged ground-truth future over the horizon, for evaluation and
        rendering only -- pulled from the log after prediction, never fed to the model."""
        log_node = self._log_nodes.get((node.type.name, node.id))
        if log_node is None:
            return np.empty((0, 2), dtype=np.float32)
        fut = log_node.get(np.array([t + 1, t + self.ph]), {'position': ['x', 'y']})
        return np.asarray(fut[~np.isnan(fut).any(axis=1)], dtype=np.float32)

    def num_tracked(self):
        """(agents observed at the latest timestep, edges in the current interaction graph)."""
        return len(self.model.tracked_nodes), self.model.num_edges()
