"""The ego-proximity risk metric.

At each simulation timestep the ego carries a disc of radius `radius` metres. For each
predicted agent, the risk is the fraction of that agent's sampled future trajectories that
reach into that disc at some point over the prediction horizon:

    risk(agent, t) = (# samples that come within `radius` of the ego at t) / (# samples)

The disc is centred on the ego's position *at t* and does not move over the horizon -- it is
the ego's current keep-out region, evaluated against everywhere the agent might go. It does
move from frame to frame, following the ego through the run, so each timestep is scored
against where the ego actually is then. This is deliberately not time-aligned: a sample
counts whether it arrives at horizon step 1 or step ph.

A sample's polyline is measured by *segment* distance, not just at its sampled points, so a
trajectory that passes through the disc between two horizon steps still counts. The agent's
current position is prepended, so the path is continuous from where the agent is now -- an
agent already inside the disc therefore scores 1.0. `dist_now` in the output makes that case
visible rather than silent.

`evaluate_frame` is called from inside the online loop (unity_online.run), once per timestep,
so this module stays numpy-only -- no matplotlib, no bundle I/O. The overlay video lives in
render.py and the CLI in cli.py.
"""
import csv

import numpy as np


DEFAULT_RADIUS = 25.0        # metres; the ego's keep-out disc


def closest_approach(samples_w, ego_xy):
    """Each sample trajectory's closest approach to a point, in metres.

    Distance is to the polyline, not to its vertices: every segment is tested with a
    point-to-segment distance, so a trajectory whose sampled points straddle the disc
    without landing in it is still measured correctly. That matters at a 0.5 s step, where
    an agent can cover a good fraction of the disc between two horizon steps.

    :param samples_w: (S, P, 2) sample polylines in world coords (P = ph + 1, index 0 = now).
    :param ego_xy: (2,) the disc centre in world coords.
    :return: float (S,) -- min distance from each sample's path to `ego_xy`.
    """
    samples_w = np.asarray(samples_w, dtype=float)
    if samples_w.ndim != 3 or samples_w.shape[0] == 0:
        return np.zeros(len(samples_w) if samples_w.ndim else 0)
    p = np.asarray(ego_xy, dtype=float)
    if samples_w.shape[1] < 2:                       # a single point, no segments
        return np.linalg.norm(samples_w[:, 0] - p, axis=-1)

    a = samples_w[:, :-1]                            # (S, I, 2) segment starts
    b = samples_w[:, 1:]                             # (S, I, 2) segment ends
    d = b - a
    dd = np.einsum('...i,...i->...', d, d)           # (S, I) squared segment length
    # Project the point onto each segment, clamped to it. A zero-length segment (a stationary
    # agent) would divide by zero, so guard the denominator and clamp to its start point.
    u = np.where(dd > 0, np.einsum('...i,...i->...', p - a, d) / np.where(dd > 0, dd, 1.0), 0.0)
    closest = a + np.clip(u, 0.0, 1.0)[..., None] * d
    return np.linalg.norm(closest - p, axis=-1).min(axis=1)


def entry_mask(samples_w, ego_xy, radius=DEFAULT_RADIUS):
    """Which of an agent's sample trajectories enter the ego's disc.

    :param samples_w: (S, P, 2) sample polylines in world coords (P = ph + 1).
    :param ego_xy: (2,) ego position at this timestep, world coords.
    :param radius: disc radius in metres.
    :return: (mask, dists) -- bool (S,) True where the sample reaches the disc, and the
        float (S,) closest approach of each sample.
    """
    dists = closest_approach(samples_w, ego_xy)
    return dists <= radius, dists


CSV_FIELDS = ['t', 'agent', 'type', 'n_samples', 'n_enter', 'risk', 'dist_now', 'min_dist']


def evaluate_frame(rec, off, radius=DEFAULT_RADIUS):
    """Proximity risk for the agents of a single frame record.

    Split out of `evaluate` so a streaming driver (unity_online.py) can score a timestep the
    moment the model produces it, with no access to later frames.

    :param rec: one bundle frame record ({'t', 'nodes', 'ego', ...}).
    :param off: (2,) scene-local -> world offset, i.e. meta['x_min'/'y_min'].
    :param radius: disc radius in metres.
    :return: (rows, frame_entry) where frame_entry is (ego_xy, {node_id: (mask,
        samples_world)}) for the visualizer, or None if the frame has no ego pose. Colour is
        not decided here: it is an identity scheme the *figure* picks (visualization.colors),
        and the metric has no business carrying one through the online loop.
    """
    if not rec.get('ego'):
        return [], None
    ego_xy = np.asarray(rec['ego'][0], dtype=float)               # (2,) world
    rows, fr = [], {}
    for nd in rec['nodes']:
        s = np.asarray(nd['samples'], dtype=float)               # (S, ph, 2) scene-local
        if s.size == 0:
            continue
        cur = np.asarray(nd['history'], dtype=float)[-1]         # scene-local current pos
        poly = np.concatenate([np.broadcast_to(cur, (s.shape[0], 1, 2)), s], axis=1)
        samples_w = poly + off                                   # -> world
        mask, dists = entry_mask(samples_w, ego_xy, radius=radius)
        rows.append({'t': rec['t'], 'agent': nd['id'], 'type': nd['type'],
                     'n_samples': int(s.shape[0]), 'n_enter': int(mask.sum()),
                     'risk': float(mask.mean()),
                     'dist_now': round(float(np.linalg.norm(cur + off - ego_xy)), 3),
                     'min_dist': round(float(dists.min()), 3)})
        fr[nd['id']] = (mask, samples_w)
    return rows, (ego_xy, fr)


def record_frame(rec, rows, entry):
    """Write a scored frame's risk back into the frame record, so it lands in the bundle.

    Risk is computed once, online, at the moment of prediction; everything downstream reads
    it back off disk rather than recomputing it. The per-agent CSV keeps only the counts,
    which is not enough to draw the overlay -- that needs to know *which* samples entered --
    so the per-sample mask is stored here alongside the samples it refers to.

    Mutates `rec` in place and returns it. `rows`/`entry` are what `evaluate_frame` returned
    for this same record.
    """
    if entry is None:
        return rec
    _, fr = entry
    by_id = {r['agent']: r for r in rows}
    for nd in rec['nodes']:
        scored = fr.get(nd['id'])
        if scored is None:
            continue
        r = by_id.get(nd['id'], {})
        nd['risk_enter'] = np.asarray(scored[0], dtype=bool)      # (S,) entered the disc
        nd['risk_dist_now'] = r.get('dist_now')
        nd['risk_min_dist'] = r.get('min_dist')
    return rec


def has_recorded_risk(bundle):
    """True if this bundle carries the risk its run already scored (see `record_frame`)."""
    return any('risk_enter' in nd for rec in bundle['frames'] for nd in rec['nodes'])


def load_recorded(bundle):
    """Read back the risk a run recorded, with no geometry recomputed.

    Returns the same `(rows, per_frame)` pair as `evaluate`, so a visualizer consumes either
    interchangeably -- but every risk decision here was made online, at prediction time. The
    only arithmetic is the scene-local -> world shift the renderer needs to draw samples.
    """
    meta = bundle['meta']
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    rows, per_frame = [], {}
    for rec in bundle['frames']:
        if not rec.get('ego'):
            continue
        ego_xy = np.asarray(rec['ego'][0], dtype=float)
        fr = {}
        for nd in rec['nodes']:
            if 'risk_enter' not in nd:
                continue
            mask = np.asarray(nd['risk_enter'], dtype=bool)
            s = np.asarray(nd['samples'], dtype=float)
            if s.size == 0 or mask.size == 0:
                continue
            cur = np.asarray(nd['history'], dtype=float)[-1]
            samples_w = np.concatenate(
                [np.broadcast_to(cur, (s.shape[0], 1, 2)), s], axis=1) + off
            rows.append({'t': rec['t'], 'agent': nd['id'], 'type': nd['type'],
                         'n_samples': int(mask.size), 'n_enter': int(mask.sum()),
                         'risk': float(mask.mean()),
                         'dist_now': nd.get('risk_dist_now'),
                         'min_dist': nd.get('risk_min_dist')})
            fr[nd['id']] = (mask, samples_w)
        if fr:
            per_frame[rec['t']] = (ego_xy, fr)
    return rows, per_frame


def evaluate(bundle, radius=DEFAULT_RADIUS):
    """Compute proximity risk for every (frame, agent) with samples + an ego pose.
    Returns (rows, per_frame) where rows is a list of dicts and per_frame maps t ->
    (ego_xy, {node_id: (entry_mask, samples_world)}) for the visualizer."""
    meta = bundle['meta']
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    rows, per_frame = [], {}
    for rec in bundle['frames']:
        frame_rows, entry = evaluate_frame(rec, off, radius=radius)
        if entry is None:
            continue
        rows.extend(frame_rows)
        per_frame[rec['t']] = entry
    return rows, per_frame


def summarize(rows, radius=DEFAULT_RADIUS):
    if not rows:
        print('No (frame, agent) pairs with samples + an ego pose found.')
        print('  -> regenerate the bundle with samples: unity_online.py --style samples (or both)')
        return
    risks = np.array([r['risk'] for r in rows])
    n_pos = int((risks > 0).sum())
    n_inside = int(sum(r['dist_now'] <= radius for r in rows))
    print(f'\n=== ego-proximity risk summary  ({len(rows)} frame-agent pairs, R = {radius:g} m) ===')
    print(f'  mean risk            : {risks.mean():.3f}')
    print(f'  max risk             : {risks.max():.3f}')
    print(f'  pairs with risk > 0  : {n_pos}  ({100.0 * n_pos / len(rows):.1f}%)')
    print(f'  pairs with risk >0.5 : {int((risks > 0.5).sum())}')
    print(f'  pairs already inside : {n_inside}  (agent within R at t, so risk is 1.0 by '
          f'construction)')
    print('  highest-risk pairs:')
    for r in sorted(rows, key=lambda r: (-r['risk'], r['min_dist']))[:8]:
        print(f'    t={r["t"]:>3}  agent {r["agent"]:<10} {r["type"]:<12} '
              f'risk={r["risk"]:.3f}  ({r["n_enter"]}/{r["n_samples"]})  '
              f'now={r["dist_now"]:.1f}m  closest={r["min_dist"]:.1f}m')


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote per-agent risk -> {path}')
