"""The trajectory-crossing risk metric.

For each predicted agent at each frame, the risk is the fraction of that agent's sampled
future trajectories that geometrically cross the ego's projected path over the horizon:

    risk(agent, t) = (# samples whose polyline crosses the ego path) / (# samples)

The ego "projected path" is stored per frame as `ego_path` by the drivers -- with
`ego_path_mode: logged` the ego's actual logged future, with `projected` a constant-velocity
dead-reckoning (causal). A crossing counts only if it is a time-coincident conflict within
the horizon: the agent and the ego reach the crossing point within `time_window` horizon
steps of each other (both polylines are time-aligned, index h = horizon step h). Set
time_window -1 to ignore timing (pure geometric crossing). Both the agent's and the ego's
current positions are prepended so near-term crossings count.

`evaluate_frame` is called from inside the online loop (unity_online.run), once per agent per
timestep, so this module stays numpy-only -- no matplotlib, no bundle I/O. The overlay video
lives in render.py and the CLI in cli.py.
"""
import csv

import numpy as np

from visualization.colors import color_for, precompute_order


def _cross(u, v):
    """2D scalar cross product u x v, broadcasting over leading axes."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _segments_cross(a1, a2, b1, b2):
    """Proper segment-intersection test (excludes collinear/endpoint-touch), broadcasting.
    a1,a2 endpoints of segment A; b1,b2 of segment B; each (..., 2). Returns bool (...)."""
    d1 = _cross(b2 - b1, a1 - b1)
    d2 = _cross(b2 - b1, a2 - b1)
    d3 = _cross(a2 - a1, b1 - a1)
    d4 = _cross(a2 - a1, b2 - a1)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _polyline_segs(poly):
    """(P, 2) polyline -> (P-1, 2, 2) array of [start, end] segments."""
    return np.stack([poly[:-1], poly[1:]], axis=1)


def _time_gate(n_agent_seg, n_ego_seg, time_window):
    """(I, J) bool: may agent segment i be paired with ego segment j under `time_window`?"""
    if time_window < 0:
        return np.ones((n_agent_seg, n_ego_seg), dtype=bool)
    i = np.arange(n_agent_seg)[:, None]
    j = np.arange(n_ego_seg)[None, :]
    return np.abs(i - j) <= time_window


# Cap on the (samples x agent segments x ego segments) pair grid held in memory at once.
# The vectorized test allocates a handful of float temporaries of that shape, so the sample
# axis is chunked to keep peak memory bounded for large --num_samples or long horizons.
_MAX_PAIRS_PER_CHUNK = 4_000_000


def crossing_mask(samples_w, ego_poly, time_window=1):
    """Which of an agent's sample trajectories cross the ego path *at a coincident time*
    within the prediction horizon.

    Both polylines are time-aligned: index 0 = current position ('now'), index h = the
    position at horizon step h. A crossing counts only if the agent and the ego reach the
    crossing at close steps -- agent segment i (time interval [i, i+1]) must intersect ego
    segment j with |i - j| <= time_window. So the two are near the crossing point at
    roughly the same time (a genuine space-time conflict), not merely on overlapping paths.

    Every (sample, agent segment, ego segment) pair is tested in one broadcast rather than
    sample-by-sample: the per-pair work is a few multiplies, so a Python loop over it spends
    essentially all its time in call overhead. This runs inside the online loop once per
    agent per timestep (unity_online.run), where it used to dominate the step.

    :param samples_w: (S, P, 2) sample polylines in world coords (P = ph + 1).
    :param ego_poly: (E, 2) ego path polyline in world coords.
    :param time_window: allowed step offset between the agent's and ego's crossing segments.
        0 = strictly simultaneous; larger = more temporal slack; < 0 disables the gate
        (pure geometric crossing, timing ignored).
    :return: bool (S,) -- True where sample s has a time-coincident crossing.
    """
    samples_w = np.asarray(samples_w)
    ns = len(samples_w)
    out = np.zeros(ns, dtype=bool)

    if ns == 0 or samples_w.ndim != 3:
        return out
    ego_seg = _polyline_segs(np.asarray(ego_poly))        # (E-1, 2, 2)
    n_ego = len(ego_seg)                                  # J
    n_ag = samples_w.shape[1] - 1                         # I
    if n_ego == 0 or n_ag <= 0:
        return out

    e1 = ego_seg[:, 0]                                    # (J, 2)
    e2 = ego_seg[:, 1]
    gate = _time_gate(n_ag, n_ego, time_window)           # (I, J)
    if not gate.any():
        return out

    chunk = max(1, _MAX_PAIRS_PER_CHUNK // (n_ag * n_ego))
    for lo in range(0, ns, chunk):
        block = samples_w[lo:lo + chunk]                  # (s, P, 2)
        a1 = block[:, :-1, None, :]                       # (s, I, 1, 2)
        a2 = block[:, 1:, None, :]
        hit = _segments_cross(a1, a2, e1, e2)             # (s, I, J)
        out[lo:lo + chunk] = (hit & gate).any(axis=(1, 2))
    return out


CSV_FIELDS = ['t', 'agent', 'type', 'n_samples', 'n_cross', 'risk']


def evaluate_frame(rec, off, order, time_window=1):
    """Crossing risk for the agents of a single frame record.

    Split out of `evaluate` so a streaming driver (unity_online.py) can score a timestep
    the moment the model produces it, with no access to later frames.

    :param rec: one bundle frame record ({'t', 'nodes', 'ego', 'ego_path', ...}).
    :param off: (2,) scene-local -> world offset, i.e. meta['x_min'/'y_min'].
    :param order: node_id -> colour index map, mutated in place as agents are first seen
        (so colours stay stable across frames without knowing the agent set up front).
    :return: (rows, frame_entry) where frame_entry is (ego_poly, {node_id: (mask,
        samples_world, colour)}) for the visualizer, or None if the frame has no ego path.
    """
    ego_path = rec.get('ego_path')
    if not ego_path:
        return [], None
    ego = np.asarray(rec['ego'][0], dtype=float) if rec.get('ego') else None
    ego_poly = np.asarray(ego_path, dtype=float)                  # (E, 2) world
    if ego is not None:
        ego_poly = np.vstack([ego, ego_poly])                    # prepend ego 'now'
    rows, fr = [], {}
    for nd in rec['nodes']:
        s = np.asarray(nd['samples'], dtype=float)               # (S, ph, 2) scene-local
        if s.size == 0:
            continue
        cur = np.asarray(nd['history'], dtype=float)[-1]         # scene-local current pos
        poly = np.concatenate([np.broadcast_to(cur, (s.shape[0], 1, 2)), s], axis=1)
        samples_w = poly + off                                   # -> world
        mask = crossing_mask(samples_w, ego_poly, time_window=time_window)
        rows.append({'t': rec['t'], 'agent': nd['id'], 'type': nd['type'],
                     'n_samples': int(s.shape[0]), 'n_cross': int(mask.sum()),
                     'risk': float(mask.mean())})
        fr[nd['id']] = (mask, samples_w, color_for(nd['id'], order))
    return rows, (ego_poly, fr)


def evaluate(bundle, time_window=1):
    """Compute crossing risk for every (frame, agent) with samples + an ego path.
    `time_window` gates the temporal coincidence of the crossing (see crossing_mask).
    Returns (rows, per_frame) where rows is a list of dicts and per_frame maps t ->
    {node_id: (crossing_mask, samples_world, agent_color_idx)} for the visualizer."""
    meta = bundle['meta']
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    order = precompute_order(bundle['frames'])
    rows, per_frame = [], {}
    for rec in bundle['frames']:
        frame_rows, entry = evaluate_frame(rec, off, order, time_window=time_window)
        if entry is None:
            continue
        rows.extend(frame_rows)
        per_frame[rec['t']] = entry
    return rows, per_frame


def summarize(rows):
    if not rows:
        print('No (frame, agent) pairs with samples + ego_path found.')
        print('  -> regenerate the bundle with samples: unity_predict.py --style samples (or both)')
        return
    risks = np.array([r['risk'] for r in rows])
    n_pos = int((risks > 0).sum())
    print(f'\n=== crossing-risk summary  ({len(rows)} frame-agent pairs) ===')
    print(f'  mean risk           : {risks.mean():.3f}')
    print(f'  max risk            : {risks.max():.3f}')
    print(f'  pairs with risk > 0 : {n_pos}  ({100.0 * n_pos / len(rows):.1f}%)')
    print(f'  pairs with risk >0.5: {int((risks > 0.5).sum())}')
    print('  highest-risk pairs:')
    for r in sorted(rows, key=lambda r: -r['risk'])[:8]:
        print(f'    t={r["t"]:>3}  agent {r["agent"]:<10} {r["type"]:<12} '
              f'risk={r["risk"]:.3f}  ({r["n_cross"]}/{r["n_samples"]})')


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote per-agent risk -> {path}')
