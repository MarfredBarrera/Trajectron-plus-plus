"""Batched-online CVAE: the offline model architecture, advanced one timestep at a time.

Trajectron++ ships two variants of the same network:

    MultimodalGenerativeCVAE (model/mgcvae.py)
        One model per *NodeType*, agents in the batch dimension. Fast, but it re-reads and
        re-encodes each agent's whole history window on every call, so it needs the history
        to still be lying around -- it replays a log rather than consuming a stream.

    OnlineMultimodalGenerativeCVAE (model/online/online_mgcvae.py)
        One model per *agent*, each holding its own LSTM hidden states and consuming a single
        new observation per timestep. Genuinely incremental, but `batch_size = 1` is hardcoded
        and the batch dimension is repurposed for candidate ego plans, so cost grows linearly
        with the number of agents -- measurably slower than the offline model on real scenes.

This module is the third combination: per-NodeType models with **agents in the batch
dimension** *and* **persistent per-agent recurrent state**, so every observation is encoded
exactly once and all agents of a type are encoded and decoded in one pass.

Only the two recurrent encoders (node history, and one edge encoder per edge type) need to
change; they are stepped by a single observation here instead of re-run over a window.
Everything downstream -- edge-influence attention, map encoder, latent, decoder, dynamics --
is the inherited offline implementation, unchanged and already batched.
"""
import numpy as np
import torch

from model.mgcvae import MultimodalGenerativeCVAE
from model.model_utils import ModeKeys
from environment.scene_graph import DirectedEdge


class RecurrentStateStore(object):
    """Per-agent LSTM state, addressed by agent rather than by batch row.

    An online batch is not stable: agents appear, disappear and get re-ordered between
    frames. Keeping ``(h, c)`` in a dict keyed by agent and gathering it into a single
    ``[1, bs, H]`` tensor per call means the batch can be assembled in any order and an
    agent's memory follows the agent, never a row index.

    Agents absent from the store start from a zero state -- exactly what ``nn.LSTM`` does
    when called without an initial state, so an agent's first observation is encoded the
    same way the offline model encodes the start of a history window.
    """

    def __init__(self, hidden_dim, device):
        self.hidden_dim = hidden_dim
        self.device = device
        self._h = dict()
        self._c = dict()
        self._zero = torch.zeros(hidden_dim, device=device)

    def gather(self, keys):
        """``(h, c)``, each ``[1, len(keys), H]``, in the order given. Unknown keys read zero."""
        h = torch.stack([self._h.get(k, self._zero) for k in keys], dim=0)
        c = torch.stack([self._c.get(k, self._zero) for k in keys], dim=0)
        return h.unsqueeze(0), c.unsqueeze(0)

    def scatter(self, keys, state):
        """Write an LSTM's returned ``(h, c)`` back to the agents that produced it."""
        h, c = state
        h, c = h.squeeze(0), c.squeeze(0)
        for i, key in enumerate(keys):
            self._h[key] = h[i]
            self._c[key] = c[i]

    def forget(self, keys):
        """Drop these agents' state. Called when an agent is evicted for good."""
        for key in keys:
            self._h.pop(key, None)
            self._c.pop(key, None)

    def __len__(self):
        return len(self._h)


class BatchedOnlineMultimodalGenerativeCVAE(MultimodalGenerativeCVAE):
    """One model per NodeType, stepped one observation at a time over a batch of agents.

    Usage per timestep is `encoder_forward` once, then `decoder_forward` as many times as
    you need distributions from it (samples / most-likely / analytic GMM all reuse the same
    encoder pass). The split mirrors OnlineMultimodalGenerativeCVAE's, which is what makes
    the "encode once, decode several ways" pattern possible at all -- the offline
    `predict()` re-encodes from scratch on every call.

    `incl_robot_node` models condition every agent's prediction on the ego's planned future,
    which fits the batch dimension directly -- one ego-plan row per agent, each expressed
    relative to that agent, as the offline model does. Exactly one plan per timestep is
    supported. (`OnlineMultimodalGenerativeCVAE` is the variant that cannot batch over agents
    at all: it spends the batch dimension on *candidate* ego plans instead, which is a
    different thing to want and rules out batching agents together.)
    """

    def __init__(self, env, node_type, model_registrar, hyperparams, device, edge_types):
        super(BatchedOnlineMultimodalGenerativeCVAE, self).__init__(env, node_type,
                                                                    model_registrar, hyperparams,
                                                                    device, edge_types,
                                                                    log_writer=None)
        self.history_states = RecurrentStateStore(hyperparams['enc_rnn_dim_history'], device)
        self.edge_states = {edge_type: RecurrentStateStore(hyperparams['enc_rnn_dim_edge'], device)
                            for edge_type in self.edge_types}

        # Set by encoder_forward, consumed by decoder_forward.
        self.nodes = []             # agents in batch-row order
        self.predict_nodes = []     # the subset that decoder_forward will produce output for
        self.predict_rows = None    # their row indices, as a LongTensor
        self.x = None               # CVAE condition tensor [bs, x_size]
        self.n_s_t0 = None          # standardized current state [bs, state_dim]
        self.state_t = None         # unstandardized current state [bs, state_dim]
        self.x_nr_t = None          # ego present state, per agent [bs, robot_state]
        self.y_r = None             # ego planned future, per agent [bs, ph, robot_state]

    # ------------------------------------------------------------------ #
    # Agent lifecycle                                                     #
    # ------------------------------------------------------------------ #
    def forget(self, nodes):
        """Erase all recurrent state belonging to `nodes`.

        Called when an agent has been gone long enough to be evicted. Until then its state is
        deliberately kept: an agent that flickers out of the tracker for a frame and returns
        should resume its own memory rather than restart from zero.
        """
        self.history_states.forget(nodes)
        for store in self.edge_states.values():
            store.forget(nodes)

    def num_tracked(self):
        """How many agents currently hold history-encoder state."""
        return len(self.history_states)

    # ------------------------------------------------------------------ #
    # Encoders (the only part that differs from the offline model)        #
    # ------------------------------------------------------------------ #
    def encode_node_history_step(self, nodes, state_st_t):
        """Advance each agent's history LSTM by one observation. -> [bs, enc_rnn_dim_history].

        The offline counterpart (`encode_node_history`) runs the same LSTM over a whole
        `[bs, mhl + 1, state]` window and keeps the output at each sequence's last valid
        index. Here the window is one step long and the "last valid index" is the only index,
        so the variable-length machinery -- and `first_history_indices` with it -- disappears:
        an agent that has been observed twice has simply had this called twice.

        The one thing an incremental encoder cannot reproduce
        ----------------------------------------------------
        The offline model standardizes a whole window against the agent's position *at t*, so
        past rows carry a displacement trail leading up to the present. An incremental encoder
        must standardize each observation when it arrives, against the only position it has --
        its own -- so the two position channels are always zero. Velocity, acceleration and
        heading are unaffected (their scales are fixed), and the newest row is identical
        either way, since offline it is the row the window is referenced to.

        This is inherent to streaming, not an implementation shortcut: re-referencing past
        inputs to the present position means re-running the recurrence, which is the cost
        that streaming exists to avoid. It is also the *only* remaining difference from the
        offline model -- feeding the offline encoder the same window with its position
        columns zeroed reproduces this method's output to 1e-7, and the non-recurrent blocks
        of x (map, ego plan) match to 0.0. `encode_edge_step` inherits the same effect, since
        it consumes both this standardized state and neighbour states referenced the same
        way. End to end on sweep_s1_rail it moves the most-likely path by a median of 0.38 m
        over a 5 s horizon.
        """
        lstm = self.node_modules[self.node_type + '/node_history_encoder']
        outputs, state = lstm(state_st_t.unsqueeze(1), self.history_states.gather(nodes))
        self.history_states.scatter(nodes, state)
        return outputs[:, 0, :]

    def encode_edge_step(self, nodes, edge_type, state_st_t, edge_input, edge_masks):
        """Advance each agent's edge LSTM for one edge type by one observation.

        :param edge_input: `(values, rows)` -- see `combine_neighbors`.
        :param edge_masks: dynamic-edge scaling per agent, [bs, 1].
        :return: [bs, enc_rnn_dim_edge], zeroed for agents with no active edge.
        """
        combined_neighbors = self.combine_neighbors(edge_input, len(nodes),
                                                    self.neighbor_state_length(edge_type))
        joint_history = torch.cat([combined_neighbors, state_st_t], dim=-1).unsqueeze(1)

        store = self.edge_states[edge_type]
        lstm = self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + '/edge_encoder']
        outputs, state = lstm(joint_history, store.gather(nodes))
        store.scatter(nodes, state)

        encoded = outputs[:, 0, :]
        if self.hyperparams['dynamic_edges'] == 'yes':
            encoded = encoded * edge_masks
        return encoded

    def combine_neighbors(self, edge_input, batch_size, state_length):
        """Reduce each agent's neighbours of one edge type to a single state vector.

        :param edge_input: `(values, rows)` where `values` is [P, neighbour_state] holding one
            row per (agent, neighbour) pair -- the neighbour's state already made relative to
            that agent and standardized -- and `rows` is [P] giving the batch row each pair
            belongs to. Flattening the batch's ragged neighbour sets this way lets the whole
            reduction run as one scatter-add instead of a Python loop per agent, which is the
            point of batching in the first place.
        :return: [bs, neighbour_state]; rows for agents with no neighbour of this type stay
            zero, matching the offline model's explicit zero padding for that case.
        """
        values, rows = edge_input
        combined = torch.zeros((batch_size, state_length), device=self.device)
        if values.shape[0] > 0:
            combined.index_add_(0, rows, values)

        method = self.hyperparams['edge_state_combine_method']
        if method == 'sum':
            return combined
        if method == 'mean':
            counts = torch.zeros(batch_size, device=self.device)
            counts.index_add_(0, rows, torch.ones(rows.shape[0], device=self.device))
            return combined / counts.clamp(min=1.0).unsqueeze(1)
        # 'max' is unreachable in practice: the offline implementation of it stacks the
        # (values, indices) tuple that torch.max returns and would fail the same way.
        raise NotImplementedError(f"edge_state_combine_method {method!r} is not supported by "
                                  f"the batched-online model (use 'sum' or 'mean')")

    def neighbor_state_length(self, edge_type):
        """Width of the state vector of the edge type's *destination* node type."""
        return int(np.sum([len(dims) for dims in self.state[edge_type[1]].values()]))

    # ------------------------------------------------------------------ #
    # One timestep: encode once, decode as often as needed                #
    # ------------------------------------------------------------------ #
    def encoder_forward(self, nodes, state_t, state_st_t, edge_inputs, edge_masks, maps,
                        predict_rows, robot=None):
        """Consume one timestep of observations and build the CVAE condition tensor `x`.

        Every argument is in batch-row order, row `i` belonging to `nodes[i]`.

        :param nodes: agents in this batch [bs]. Includes agents that are only being kept
            ticking (see `predict_rows`), because their recurrent state has to keep advancing.
        :param state_t: unstandardized current state [bs, state_dim], NaNs already zeroed.
        :param state_st_t: current state standardized relative to each agent's own position
            [bs, state_dim] -- the encoders' input, and the decoder's `n_s_t0`.
        :param edge_inputs: {edge_type: (values, rows)}, see `combine_neighbors`.
        :param edge_masks: dynamic-edge scaling per agent [bs, 1].
        :param maps: rotated map patches [bs, C, H, W], or None if this node type has no map
            encoder.
        :param predict_rows: LongTensor of the rows `decoder_forward` should produce output
            for. Agents outside it still have their encoders advanced but are not decoded --
            an agent that has just gone missing, or that has not been seen often enough yet,
            should keep its memory warm without costing a decoder pass (decoding dominates
            runtime by an order of magnitude).
        :param robot: the ego's plan over [t, t+ph] made relative to each agent and
            standardized, [bs, ph+1, robot_state]. Required for `incl_robot_node` models,
            ignored otherwise. Row 0 is the ego's present state, rows 1..ph its plan.
        """
        mode = ModeKeys.PREDICT
        self.nodes = list(nodes)
        self.predict_rows = predict_rows
        self.predict_nodes = [self.nodes[i] for i in predict_rows.tolist()]
        self.state_t = state_t
        self.n_s_t0 = state_st_t

        node_history_encoded = self.encode_node_history_step(nodes, state_st_t)

        # Concatenation order must match the offline model's, since these weights were
        # trained against that layout: edge influence, history, [robot future], map.
        x_concat_list = list()
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = [self.encode_edge_step(nodes, edge_type, state_st_t,
                                                        edge_inputs[edge_type], edge_masks)
                                  for edge_type in self.edge_types]
            x_concat_list.append(self.encode_total_edge_influence(mode, node_edges_encoded,
                                                                  node_history_encoded, len(nodes)))
        x_concat_list.append(node_history_encoded)
        if self.hyperparams['incl_robot_node']:
            if robot is None:
                raise ValueError(f'{self.node_type} model has incl_robot_node set but no ego '
                                 f'plan was supplied to encoder_forward')
            self.x_nr_t, self.y_r = robot[:, 0, :], robot[:, 1:, :]
            x_concat_list.append(self.encode_robot_future(mode, self.x_nr_t, self.y_r))
        if maps is not None:
            # Dropout is a no-op outside training mode, so it is left out rather than
            # threaded through with training=False.
            x_concat_list.append(self.node_modules[self.node_type + '/map_encoder'](maps * 2. - 1.,
                                                                                    False))

        self.x = torch.cat(x_concat_list, dim=1)

    def decoder_forward(self, prediction_horizon, num_samples, z_mode=False, gmm_mode=False,
                        full_dist=True, all_z_sep=False):
        """Decode the rows selected by the last `encoder_forward`. -> (GMM2D, samples).

        Safe to call repeatedly after one `encoder_forward` -- nothing here writes back to the
        encoders, so drawing samples, the most-likely path and the analytic GMM from a single
        timestep costs one encoder pass and three decoder passes.

        `full_dist` defaults to True here, matching `Trajectron.predict`, because it changes
        which branch of `DiscreteLatent.sample_p` runs and therefore what "most likely" means.
        (`OnlineTrajectron.sample_model` defaults it to False, which quietly disagrees with
        the offline driver.)
        """
        mode = ModeKeys.PREDICT
        rows = self.predict_rows
        x = self.x[rows]
        n_s_t0 = self.n_s_t0[rows]

        # The dynamics integrator is stateful. Setting its initial condition here rather than
        # in encoder_forward keeps it in step with the rows actually being decoded.
        # TODO: generalize away from a fixed position/velocity column layout, as upstream does.
        self.dynamic.set_initial_condition({'pos': self.state_t[rows, 0:2],
                                            'vel': self.state_t[rows, 2:4]})

        x_nr_t = None if self.x_nr_t is None else self.x_nr_t[rows]
        y_r = None if self.y_r is None else self.y_r[rows]

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples, mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)
        return self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z, prediction_horizon,
                           num_samples, num_components, gmm_mode)
