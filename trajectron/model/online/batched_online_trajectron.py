"""Batched-online Trajectron++: streaming inference with agents in the batch dimension.

`BatchedOnlineTrajectron` consumes one timestep of observations at a time -- it never reads
ahead, and never re-reads an observation it has already encoded -- while keeping the offline
model's shape: one `BatchedOnlineMultimodalGenerativeCVAE` per NodeType, all agents of a type
encoded and decoded together.

Per timestep it:

  1. appends the new observations to per-agent ring buffers and ages out agents that have
     stopped being observed (`incremental_forward` -> `_observe`);
  2. rebuilds the interaction graph over the buffered window, so the edge addition/removal
     filters see the same temporal context they were trained with (`_update_scene_graph`);
  3. assembles one batch per NodeType -- current states, per-agent neighbour sets, dynamic
     edge masks, rotated map patches -- and advances every encoder by exactly one step
     (`_encode`);
  4. decodes on demand (`sample_model`), any number of times, from that one encoder pass.

What is carried over from `OnlineTrajectron`, and what is not
------------------------------------------------------------
Carried over: the ring buffers, the online scene-graph rebuild, and the agent lifecycle
(zero-initialized state on first appearance, a grace period of `len(edge_removal_filter)`
frames before an agent that stops being observed is evicted).

Not carried over: per-agent model objects. Recurrent state lives in a dict keyed by agent
inside each NodeType's model (`RecurrentStateStore`), so the batch can be rebuilt in any
order every frame while each agent keeps its own memory.

Two deliberate differences from `model/online/online_trajectron.py`:

  * An agent's own state is standardized with ``std[0:2] = attention_radius[(type, type)]``,
    as `model/dataset/preprocessing.get_node_timestep_data` does. `online_trajectron.py`
    computes that std and then never passes it to `standardize()`, so it falls back to the
    default position scale -- a quiet mismatch with how the weights were trained.
  * Edges are encoded for *every* edge type of a node type on every frame, again as offline.
    `online_mgcvae.py` only encodes edge types it has seen a neighbour for, which changes the
    input to the edge-influence combiner for isolated agents.

How close this is to the offline model
--------------------------------------
Exact, except for one thing streaming cannot do. Compared block by block against
`MultimodalGenerativeCVAE.obtain_encoded_tensors` on the same observations, the map encoding
and the ego-plan encoding agree to 0.0 -- neither carries state across timesteps. The two
*recurrent* encoders (node history, and the per-edge-type encoders) differ only by the
position reference frame that streaming forces, described in
`BatchedOnlineMultimodalGenerativeCVAE.encode_node_history_step`; re-referencing the offline
window reproduces the history encoding to 1e-7, and the edge encoders consume the same
per-timestep relative states, so they inherit it too.

End to end on sweep_s1_rail that is worth a median 0.38 m (mean 0.58 m) on the most-likely
path over a 5 s horizon, against a batch driver re-encoding a full window every timestep.

Conditioning on the ego's plan
-----------------------------
With an `incl_robot_node` checkpoint, pass the ego's planned state over [t, t+ph] to
`incremental_forward` and every agent's prediction is conditioned on where the ego intends
to go. One plan per timestep; it is made relative to each agent and rides in the batch
dimension alongside them, as offline. The ego should also be streamed in as an observation
with `is_robot=True` so it acts as a neighbour in the interaction graph -- it is then
automatically excluded from the predicted set.
"""
from collections import Counter

import numpy as np
import torch

from model.trajectron import Trajectron
from model.online.batched_online_mgcvae import BatchedOnlineMultimodalGenerativeCVAE
from environment import RingBuffer, TemporalSceneGraph, SceneGraph


class BatchedOnlineTrajectron(Trajectron):
    """Streaming Trajectron++ over a live stream of per-timestep observations.

    :param model_registrar: a loaded `ModelRegistrar` holding the trained weights.
    :param hyperparams: the model's `config.json` (with `map_enc_dropout` zeroed for eval).
    :param device: torch device string.
    :param min_history_timesteps: how many observations *before* the current one an agent
        needs before it is predicted for. 1 matches `Trajectron.predict`'s default, i.e. an
        agent is predicted from its second observation onwards; its encoders start advancing
        from the first either way. 0 predicts from the very first observation.
    """

    def __init__(self, model_registrar, hyperparams, device, min_history_timesteps=1):
        super(BatchedOnlineTrajectron, self).__init__(model_registrar=model_registrar,
                                                      hyperparams=hyperparams,
                                                      log_writer=None,
                                                      device=device)
        self.min_history_timesteps = min_history_timesteps
        self.incl_robot_node = bool(hyperparams.get('incl_robot_node'))
        # Enough history for the longest thing that looks backwards: the edge filters and the
        # trained history window. The encoders themselves need only the newest row -- the
        # buffer exists for the scene-graph rebuild and for reporting an agent's track.
        self.ring_capacity = max(len(hyperparams['edge_removal_filter']),
                                 len(hyperparams['edge_addition_filter']),
                                 hyperparams['maximum_history_length']) + 1
        # Frames an unobserved agent is kept (still ticking, no longer predicted) before its
        # state is thrown away. Matched to the edge removal filter so an agent survives
        # exactly as long as its edges take to decay.
        self.grace_period = len(hyperparams['edge_removal_filter'])

        self.node_data = dict()          # agent -> RingBuffer of observed state rows
        self.observation_counts = Counter()   # agent -> observations actually seen
        self.removed_nodes = Counter()   # agent -> frames since it was last observed
        self.scene_graph = None
        self._std_memo = dict()

    def __repr__(self):
        return (f'BatchedOnlineTrajectron(tracking {len(self.node_data)} agents '
                f'across {len(self.node_models_dict)} node type(s), device: {self.device})')

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #
    def set_environment(self, env):
        """Bind to an Environment holding exactly one scene and build one model per NodeType.

        The scene supplies `dt` (dynamics integration), the map rasters, and the env-level
        attention radii / standardization. It does *not* need to contain any nodes: agents
        arrive through `incremental_forward`, which is the whole point of streaming.
        """
        if len(env.scenes) != 1:
            raise ValueError('BatchedOnlineTrajectron needs an Environment with exactly one '
                             f'scene (got {len(env.scenes)})')
        self.env = env
        self.scene_graph = SceneGraph(edge_radius=env.attention_radius)
        self.node_data.clear()
        self.observation_counts.clear()
        self.removed_nodes.clear()
        self.node_models_dict.clear()
        self._std_memo.clear()

        edge_types = env.get_edge_types()
        for node_type in env.NodeType:
            if node_type in self.pred_state:
                self.node_models_dict[node_type] = BatchedOnlineMultimodalGenerativeCVAE(
                    env, node_type, self.model_registrar, self.hyperparams, self.device, edge_types)

    # ------------------------------------------------------------------ #
    # One timestep in                                                     #
    # ------------------------------------------------------------------ #
    def incremental_forward(self, new_inputs_dict, robot_plan=None, run_models=True):
        """Feed one timestep of observations and advance every encoder by one step.

        :param new_inputs_dict: {Node: (1, state_dim) array} for the agents observed at this
            timestep. Agents not in it are treated as unobserved this frame. The same shape
            `Scene.get_clipped_input_dict` produces, so a recorded scene can be replayed
            through this as if it were live. Include the ego here too (as a node with
            `is_robot=True`) when conditioning on its plan, so it acts as a neighbour.
        :param robot_plan: the ego's planned state over [t, t+ph], (ph+1, robot_state),
            unstandardized and in the same frame as the observations. Row 0 is the present.
            Required for `incl_robot_node` models, ignored otherwise.
        :param run_models: when False, buffers and the scene graph advance but no network
            runs. Only useful to skip past a prefix without paying for it -- note that the
            encoders then have no memory of the skipped timesteps.
        """
        if self.incl_robot_node and run_models and robot_plan is None:
            raise ValueError('this is an incl_robot_node model: incremental_forward needs '
                             'the ego plan (robot_plan) for every timestep it runs')
        with torch.no_grad():
            self._observe(new_inputs_dict)
            self._update_scene_graph()
            if run_models:
                self._encode(robot_plan)

    def _observe(self, new_inputs_dict):
        """Append this timestep's observations and age the agents that are missing from it."""
        for node, new_input in new_inputs_dict.items():
            if node not in self.node_data:
                self.node_data[node] = RingBuffer(capacity=self.ring_capacity,
                                                  dtype=(float, self.state_length[node.type]))
            self.node_data[node].append(new_input)
            self.observation_counts[node] += 1
            self.removed_nodes.pop(node, None)

        # Anything tracked but not observed this frame is one frame further from the last
        # time we saw it -- both agents that just went missing and ones already missing.
        self.removed_nodes.update(set(self.node_data) - set(new_inputs_dict))

        for node, age in list(self.removed_nodes.items()):
            if age >= self.grace_period:
                del self.node_data[node]
                del self.removed_nodes[node]
                self.observation_counts.pop(node, None)
                model = self.node_models_dict.get(node.type)
                if model is not None:
                    model.forget([node])

        # Survivors of the grace period get a NaN row, which becomes a zero observation once
        # standardized. Their LSTMs keep ticking on it, so an agent that reappears within the
        # grace period resumes with its memory intact rather than restarting from zero.
        for node in self.removed_nodes:
            self.node_data[node].append(np.full((1, self.node_data[node].shape[1]), np.nan))

    def _update_scene_graph(self):
        """Rebuild the interaction graph over the whole buffered window.

        The edge addition/removal filters are temporal convolutions, so the graph has to be
        derived from the buffered positions rather than from this frame alone -- that is what
        makes an edge fade in and out gradually instead of flickering.
        """
        temp_scene_dict = {node: np.asarray(buf)[:, 0:2] for node, buf in self.node_data.items()}
        if not temp_scene_dict:
            self.scene_graph = SceneGraph(edge_radius=self.env.attention_radius)
            return
        self.scene_graph = TemporalSceneGraph.create_from_temp_scene_dict(
            temp_scene_dict,
            self.env.attention_radius,
            duration=self.ring_capacity,
            edge_addition_filter=self.hyperparams['edge_addition_filter'],
            edge_removal_filter=self.hyperparams['edge_removal_filter'],
            online=True).to_scene_graph(t=self.ring_capacity - 1)

    # ------------------------------------------------------------------ #
    # Batch assembly                                                      #
    # ------------------------------------------------------------------ #
    def _self_std(self, node_type):
        """Standardization scale for an agent's own state.

        Positions are scaled by the agent's own attention radius rather than the dataset
        position std, matching `get_node_timestep_data`. `get_standardize_params` memoizes and
        returns the array it cached, so it is copied before being modified.
        """
        if node_type not in self._std_memo:
            _, std = self.env.get_standardize_params(self.state[node_type], node_type)
            std = np.array(std, dtype=float)
            std[0:2] = self.env.attention_radius[(node_type, node_type)]
            self._std_memo[node_type] = std
        return self._std_memo[node_type]

    def _edge_std(self, edge_type):
        """Standardization scale for a neighbour's state, seen across `edge_type`."""
        if edge_type not in self._std_memo:
            _, std = self.env.get_standardize_params(self.state[edge_type[1]], edge_type[1])
            std = np.array(std, dtype=float)
            std[0:2] = self.env.attention_radius[edge_type]
            self._std_memo[edge_type] = std
        return self._std_memo[edge_type]

    def _current_states(self):
        """{agent: (1, state_dim) current observation}, NaNs zeroed.

        Read straight out of the ring buffers, so nothing here touches the Node objects the
        observations came from. (`OnlineTrajectron` writes the buffers back into them via
        `Node.overwrite_data` because its encoders read from the Node; not needing that keeps
        a replayed Scene unmodified and makes agents from a live tracker just as usable.)
        """
        states = dict()
        for node, buf in self.node_data.items():
            x = np.array(np.asarray(buf)[-1:], dtype=float)
            x[np.isnan(x)] = 0.0
            states[node] = x
        return states

    def _batch_nodes(self, node_type):
        """(all agents of this type to encode, row indices of those to decode).

        Everything tracked is encoded, so no agent's memory goes stale. Only agents observed
        this frame and seen at least `min_history_timesteps + 1` times are decoded, matching
        which agents `Trajectron.predict` would return at the same point in the log.

        The ego is left out of the batch entirely when we are conditioning on its plan: it is
        not something to predict, and it still reaches the model as a graph neighbour, which
        reads its state from the buffers rather than from this batch. (`get_timesteps_data`
        drops it the same way, via `return_robot=not incl_robot_node`.)
        """
        nodes = sorted((n for n in self.node_data
                        if n.type == node_type and not (n.is_robot and self.incl_robot_node)),
                       key=lambda n: n.id)
        rows = [i for i, node in enumerate(nodes)
                if node not in self.removed_nodes
                and self.observation_counts[node] > self.min_history_timesteps]
        return nodes, rows

    def _edge_inputs(self, node_type, nodes, states):
        """Per edge type, every (agent, neighbour) pair flattened into one array.

        For each pair the neighbour's state is made relative to the agent and standardized --
        the operation `get_node_timestep_data` does one neighbour at a time -- but here the
        batch's whole ragged neighbour structure is expressed as `values` [P, neighbour_state]
        plus `rows` [P], so the model can reduce it with a single scatter-add.

        Returns ({edge_type: (values, rows)}, edge_masks [bs, 1]).
        """
        model = self.node_models_dict[node_type]
        ego = np.concatenate([states[node] for node in nodes], axis=0)     # [bs, state_dim]

        edge_inputs = dict()
        for edge_type in model.edge_types:
            other_type = edge_type[1]
            others = [n for n in self.node_data if n.type == other_type]
            other_index = {node: i for i, node in enumerate(others)}

            pair_rows, pair_cols = [], []
            for i, node in enumerate(nodes):
                for neighbor in self.scene_graph.get_neighbors(node, other_type):
                    pair_rows.append(i)
                    pair_cols.append(other_index[neighbor])

            width = model.neighbor_state_length(edge_type)
            if pair_rows:
                neighbor_states = np.concatenate([states[n] for n in others], axis=0)
                # The agent's own state, truncated or zero-padded to the neighbour's layout.
                # This assumes the leading state dimensions mean the same thing for both node
                # types, which is the assumption the offline preprocessing makes too.
                shared = min(ego.shape[-1], width)
                ego_padded = np.zeros((len(nodes), width))
                ego_padded[:, :shared] = ego[:, :shared]
                values = ((neighbor_states[pair_cols] - ego_padded[pair_rows])
                          / self._edge_std(edge_type))
                values = torch.tensor(values, dtype=torch.float, device=self.device)
                rows = torch.tensor(pair_rows, dtype=torch.long, device=self.device)
            else:
                values = torch.zeros((0, width), device=self.device)
                rows = torch.zeros(0, dtype=torch.long, device=self.device)
            edge_inputs[edge_type] = (values, rows)

        # One scaling per agent, over all of its edges regardless of type -- the same value
        # the offline model applies to every edge type of that agent.
        if self.scene_graph.edge_scaling is None:
            masks = np.ones((len(nodes), 1))
        else:
            masks = np.array([[min(1.0, float(np.sum(self.scene_graph.get_edge_scaling(node))))]
                              for node in nodes])
        return edge_inputs, torch.tensor(masks, dtype=torch.float, device=self.device)

    def _crop_maps(self, node_type, nodes, states):
        """Agent-aligned map patches for a whole batch, [bs, C, H, W], or None.

        One batched crop for all agents of a type, cropped on the CPU (where
        `GeometricMap.get_padded_map` allocates) and moved afterwards, as the offline
        dataloader does.
        """
        if not self.hyperparams['use_map_encoding'] or node_type not in self.hyperparams['map_encoder']:
            return None
        me_hyp = self.hyperparams['map_encoder'][node_type]
        scene_map = self.env.scenes[0].map[node_type]
        x = np.concatenate([states[node] for node in nodes], axis=0)

        heading_state_index = me_hyp.get('heading_state_index')
        if heading_state_index is None:
            rotation = None
        elif isinstance(heading_state_index, list):   # heading inferred from a velocity vector
            rotation = -np.arctan2(x[:, heading_state_index[1]],
                                   x[:, heading_state_index[0]]) * 180 / np.pi
        else:
            rotation = -x[:, heading_state_index] * 180 / np.pi
        # The map is rotated opposite to the agent so every patch is agent-aligned.
        patches = scene_map.get_cropped_maps_from_scene_map_batch(
            [scene_map] * len(nodes),
            scene_pts=torch.Tensor(x[:, :2]),
            patch_size=me_hyp['patch_size'],
            rotation=None if rotation is None else torch.Tensor(rotation))
        return patches.to(self.device)

    def _robot_plan(self, node_type, ego, robot_plan):
        """The ego's plan expressed relative to every agent in the batch.

        -> [bs, ph+1, robot_state], standardized. Positions are scaled by the
        agent-to-robot attention radius, everything else by the robot's own standardization,
        which is what `get_relative_robot_traj` does one agent at a time.

        :param ego: the batch's unstandardized current states [bs, state_dim].
        :param robot_plan: (ph+1, robot_state), unstandardized.
        """
        robot_type = self.env.robot_type
        key = ('robot', node_type)
        if key not in self._std_memo:
            _, std = self.env.get_standardize_params(self.state[robot_type], robot_type)
            std = np.array(std, dtype=float)
            std[0:2] = self.env.attention_radius[(node_type, robot_type)]
            self._std_memo[key] = std
        std = self._std_memo[key]

        plan = np.asarray(robot_plan, dtype=float)
        # Each agent's own current state, truncated or zero-padded to the robot's layout --
        # the offset the plan is measured from, so an agent sees the ego's plan in its frame.
        shared = min(ego.shape[-1], plan.shape[-1])
        ego_padded = np.zeros((ego.shape[0], plan.shape[-1]))
        ego_padded[:, :shared] = ego[:, :shared]

        relative = (plan[np.newaxis] - ego_padded[:, np.newaxis]) / std
        return torch.tensor(relative, dtype=torch.float, device=self.device)

    def _encode(self, robot_plan=None):
        """Advance every NodeType's encoders by this timestep's observations."""
        states = self._current_states()
        for node_type, model in self.node_models_dict.items():
            nodes, rows = self._batch_nodes(node_type)
            if not nodes:
                model.nodes, model.predict_nodes = [], []
                model.predict_rows = None
                continue

            raw = np.concatenate([states[node] for node in nodes], axis=0)   # [bs, state_dim]
            # Relative to the agent's own position, so the encoders see motion rather than
            # absolute map coordinates.
            rel = np.zeros_like(raw)
            rel[:, 0:2] = raw[:, 0:2]
            standardized = (raw - rel) / self._self_std(node_type)

            state_t = torch.tensor(raw, dtype=torch.float, device=self.device)
            state_st_t = torch.tensor(standardized, dtype=torch.float, device=self.device)
            edge_inputs, edge_masks = self._edge_inputs(node_type, nodes, states)
            maps = self._crop_maps(node_type, nodes, states)
            predict_rows = torch.tensor(rows, dtype=torch.long, device=self.device)
            robot = (self._robot_plan(node_type, raw, robot_plan)
                     if self.incl_robot_node else None)

            model.encoder_forward(nodes, state_t, state_st_t, edge_inputs, edge_masks, maps,
                                  predict_rows, robot=robot)

    # ------------------------------------------------------------------ #
    # Predictions out                                                     #
    # ------------------------------------------------------------------ #
    def sample_model(self, prediction_horizon, num_samples, z_mode=False, gmm_mode=False,
                     full_dist=True, all_z_sep=False, output_dists=False):
        """Decode the current encoder state. -> {agent: (1, num_samples, ph, 2)}.

        Call as often as you like after one `incremental_forward`; each call reuses that
        timestep's single encoder pass. Output shapes and the optional `output_dists` dict
        match `Trajectron.predict`, so a batch driver's post-processing applies unchanged.

        :param output_dists: also return {agent: {'mus', 'covs', 'pis'}}, the analytic GMM the
            model computed (mean/covariance propagated through the dynamics model) rather than
            statistics of the samples.
        """
        predictions_dict = dict()
        dists_dict = dict()
        with torch.no_grad():
            for node_type, model in self.node_models_dict.items():
                if not model.predict_nodes:
                    continue

                y_dist, predictions = model.decoder_forward(prediction_horizon, num_samples,
                                                            z_mode=z_mode, gmm_mode=gmm_mode,
                                                            full_dist=full_dist,
                                                            all_z_sep=all_z_sep)
                predictions_np = predictions.cpu().numpy()                    # [ns, bs, ph, 2]
                if output_dists:
                    mus_np = y_dist.mus.cpu().numpy()                         # [ns, bs, ph, K, 2]
                    covs_np = y_dist.get_covariance_matrix().cpu().numpy()
                    pis_np = torch.exp(y_dist.log_pis).cpu().numpy()          # [ns, bs, ph, K]

                for i, node in enumerate(model.predict_nodes):
                    predictions_dict[node] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                    if output_dists:
                        dists_dict[node] = {
                            'mus': np.transpose(mus_np[:, [i]], (1, 0, 2, 3, 4)),
                            'covs': np.transpose(covs_np[:, [i]], (1, 0, 2, 3, 4, 5)),
                            'pis': np.transpose(pis_np[:, [i]], (1, 0, 2, 3)),
                        }

        if output_dists:
            return predictions_dict, dists_dict
        return predictions_dict

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    @property
    def tracked_nodes(self):
        """Agents observed at the latest timestep.

        Excludes agents inside the grace period, and the ego when it is being conditioned on
        -- it is streamed in as an observation, but it is not one of the agents being
        tracked *for* prediction, so counting it would inflate the number reported.
        """
        return [node for node in self.node_data
                if node not in self.removed_nodes
                and not (node.is_robot and self.incl_robot_node)]

    def num_edges(self):
        """Edges in the current interaction graph."""
        return int(self.scene_graph.get_num_edges()) if self.scene_graph is not None else 0

    def history(self, node):
        """The agent's buffered track as the model holds it, (H, 2), oldest first.

        Capped at `ring_capacity`, and NaN rows (not yet filled, or a gap) are dropped. This
        is the raw-observation window, not the encoders' memory -- the history LSTM has
        recurrently consumed every observation since the agent appeared.
        """
        buf = np.asarray(self.node_data[node])[:, 0:2]
        return np.asarray(buf[~np.isnan(buf).any(axis=1)], dtype=np.float32)
