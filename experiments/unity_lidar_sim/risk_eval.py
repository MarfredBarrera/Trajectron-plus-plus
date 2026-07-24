#!/usr/bin/env python3
"""Naive trajectory-crossing risk metric.

For each predicted agent at each frame, the risk is the fraction of that agent's sampled
future trajectories that geometrically cross the ego's projected path over the horizon:

    risk(agent, t) = (# samples whose polyline crosses the ego path) / (# samples)

The ego "projected path" is the ego's actual logged future over the next ph timesteps
(stored per frame as `ego_path` by unity_predict.py). A crossing counts only if it is a
time-coincident conflict within the horizon: the agent and the ego reach the crossing
point within `--time_window` horizon steps of each other (both polylines are time-aligned,
index h = horizon step h). Set --time_window -1 to ignore timing (pure geometric crossing).
Both the agent's and the ego's current positions are prepended so near-term crossings count.

Requires a bundle produced with samples (unity_predict.py --style samples|both) and with
ego_path present. Outputs a per-(frame, agent) CSV + summary, and optionally a GIF that
draws the ego path and colours each agent's crossing samples red.

    python risk_eval.py --pred_file unity_out/sweep_s1_rail/predictions.pkl
    python risk_eval.py --pred_file unity_out/sweep_s1_rail/predictions.pkl --viz
"""
import os
import csv
import argparse
import numpy as np

from traj_viz import (load_bundle, _build_map_rgba, _draw_map_background,
                      _color_for, _precompute_order, _assemble,
                      ego_transform, _identity_transform)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


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


def crossing_mask(samples_w, ego_poly, time_window=1):
    """Which of an agent's sample trajectories cross the ego path *at a coincident time*
    within the prediction horizon.

    Both polylines are time-aligned: index 0 = current position ('now'), index h = the
    position at horizon step h. A crossing counts only if the agent and the ego reach the
    crossing at close steps -- agent segment i (time interval [i, i+1]) must intersect ego
    segment j with |i - j| <= time_window. So the two are near the crossing point at
    roughly the same time (a genuine space-time conflict), not merely on overlapping paths.

    :param samples_w: (S, P, 2) sample polylines in world coords (P = ph + 1).
    :param ego_poly: (E, 2) ego path polyline in world coords.
    :param time_window: allowed step offset between the agent's and ego's crossing segments.
        0 = strictly simultaneous; larger = more temporal slack; < 0 disables the gate
        (pure geometric crossing, timing ignored).
    :return: bool (S,) -- True where sample s has a time-coincident crossing.
    """
    ego_seg = _polyline_segs(ego_poly)          # (E-1, 2, 2)
    ne = len(ego_seg)
    out = np.zeros(len(samples_w), dtype=bool)
    if ne == 0:
        return out
    e1, e2 = ego_seg[:, 0], ego_seg[:, 1]
    for s in range(len(samples_w)):
        seg = _polyline_segs(samples_w[s])       # (P-1, 2, 2)
        for i in range(len(seg)):                # agent segment = horizon step i->i+1
            if time_window < 0:
                lo, hi = 0, ne
            else:
                lo, hi = max(0, i - time_window), min(ne, i + time_window + 1)
            if lo >= hi:
                continue
            if _segments_cross(seg[i, 0], seg[i, 1], e1[lo:hi], e2[lo:hi]).any():
                out[s] = True
                break
    return out


def evaluate(bundle, time_window=1):
    """Compute crossing risk for every (frame, agent) with samples + an ego path.
    `time_window` gates the temporal coincidence of the crossing (see crossing_mask).
    Returns (rows, per_frame) where rows is a list of dicts and per_frame maps t ->
    {node_id: (crossing_mask, samples_world, agent_color_idx)} for the visualizer."""
    meta = bundle['meta']
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    order = _precompute_order(bundle['frames'])
    rows, per_frame = [], {}
    for rec in bundle['frames']:
        ego_path = rec.get('ego_path')
        if not ego_path:
            continue
        ego = np.asarray(rec['ego'][0], dtype=float) if rec.get('ego') else None
        ego_poly = np.asarray(ego_path, dtype=float)                  # (E, 2) world
        if ego is not None:
            ego_poly = np.vstack([ego, ego_poly])                    # prepend ego 'now'
        fr = {}
        for nd in rec['nodes']:
            s = np.asarray(nd['samples'], dtype=float)               # (S, ph, 2) scene-local
            if s.size == 0:
                continue
            cur = np.asarray(nd['history'], dtype=float)[-1]         # scene-local current pos
            poly = np.concatenate([np.broadcast_to(cur, (s.shape[0], 1, 2)), s], axis=1)
            samples_w = poly + off                                   # -> world
            mask = crossing_mask(samples_w, ego_poly, time_window=time_window)
            prob = float(mask.mean())
            rows.append({'t': rec['t'], 'agent': nd['id'], 'type': nd['type'],
                         'n_samples': int(s.shape[0]), 'n_cross': int(mask.sum()),
                         'risk': prob})
            fr[nd['id']] = (mask, samples_w, _color_for(nd['id'], order))
        per_frame[rec['t']] = (ego_poly, fr)
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
        w = csv.DictWriter(f, fieldnames=['t', 'agent', 'type', 'n_samples', 'n_cross', 'risk'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote per-agent risk -> {path}')


def render(bundle, per_frame, out_dir, fps=2.0, fmt='gif', ego_frame=False, zoom=None):
    """One PNG per frame + a GIF and/or MP4 (`fmt` in {gif, mp4, both}): map, ego path
    (bold red), crossing samples red, the rest faded in the agent colour, each agent
    labelled with its crossing risk. `ego_frame` renders in the ego's frame (ego at the
    origin, heading up), cropped to +/- `zoom` metres."""
    meta = bundle['meta']
    map_rgba, map_bounds = _build_map_rgba(meta)
    if zoom is None:
        zoom = meta.get('zoom', 60.0)
    frame_dir = os.path.join(out_dir, 'frames_risk_ego' if ego_frame else 'frames_risk')
    os.makedirs(frame_dir, exist_ok=True)
    paths = []
    for rec in bundle['frames']:
        t = rec['t']
        if t not in per_frame:
            continue
        ego_poly, fr = per_frame[t]
        ego = rec.get('ego')
        use_ego = ego_frame and ego is not None
        if use_ego:
            ego_pos, ego_yaw = np.asarray(ego[0], dtype=float), float(ego[1])
            transform = ego_transform(ego_pos, ego_yaw)
        else:
            transform = _identity_transform

        def T(pts):                              # world -> plot coords (identity or ego)
            a = np.asarray(pts, dtype=float)
            return transform(a.reshape(-1, 2)).reshape(a.shape)

        fig, ax = plt.subplots(figsize=(12, 12) if use_ego else (11, 13))
        ax.set_facecolor('#FFFFFF')
        _draw_map_background(ax, map_rgba, map_bounds,
                             ego=(ego_pos, ego_yaw) if use_ego else None,
                             crop_radius=zoom * 1.6 if use_ego else None)
        # ego path
        ep = T(ego_poly)
        ax.plot(ep[:, 0], ep[:, 1], '-', color='#D62728', lw=2.6, zorder=700,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='w'), pe.Normal()])
        ax.scatter([ep[0, 0]], [ep[0, 1]], marker='^', s=180, color='#D62728',
                   edgecolors='k', linewidths=1.0, zorder=760)
        for nd in rec['nodes']:
            if nd['id'] not in fr:
                continue
            mask, samples_w, c = fr[nd['id']]
            samples_p = T(samples_w)
            cur = samples_p[0, 0]
            if (~mask).any():
                ax.add_collection(LineCollection(samples_p[~mask], colors=c, linewidths=0.5,
                                                 alpha=0.05, zorder=400))
            if mask.any():
                ax.add_collection(LineCollection(samples_p[mask], colors='#D62728',
                                                 linewidths=0.6, alpha=0.14, zorder=450))
            ax.scatter([cur[0]], [cur[1]], s=30, color=c, edgecolors='k', linewidths=0.6,
                       zorder=650)
            ax.annotate(f'{mask.mean():.2f}', (cur[0], cur[1]), textcoords='offset points',
                        xytext=(6, 6), fontsize=8, color='k', zorder=660,
                        path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        if use_ego:
            ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
            ax.set_xlabel('ego x (right) [m]'); ax.set_ylabel('ego y (forward) [m]')
            title = f'Ego-crossing risk (ego frame)  |  t = {t}  ({t * meta.get("dt", 0.5):.1f}s)'
        else:
            ax.set_xlim(meta['xlim']); ax.set_ylim(meta['ylim'])
            ax.set_xlabel('map x [m]'); ax.set_ylabel('map y [m]')
            title = f'Ego-crossing risk  |  t = {t}  ({t * meta.get("dt", 0.5):.1f}s)'
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        handles = [Line2D([], [], color='#D62728', lw=2.6, label='Ego projected path'),
                   Line2D([], [], color='#D62728', lw=1.0, alpha=0.8, label='Crossing samples'),
                   Line2D([], [], color='#888', lw=1.0, alpha=0.7, label='Non-crossing samples')]
        leg = ax.legend(handles=handles, loc='upper right', fontsize=11, frameon=True, facecolor='white')
        leg.set_zorder(1000)
        out = os.path.join(frame_dir, f'risk_{t:04d}.png')
        fig.savefig(out, dpi=90, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        paths.append(out)
    if paths:
        print(f'Wrote {len(paths)} risk frames')
        return _assemble(paths, out_dir, 'risk', ego_frame, fps, fmt)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred_file', required=True, help='prediction bundle (.pkl) with samples + ego_path')
    p.add_argument('--out_dir', default=None, help='where to write CSV / frames (default: next to pred_file)')
    p.add_argument('--viz', action='store_true', help='also render the crossing-risk video')
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'],
                   help='video output format(s) for the risk visualization')
    p.add_argument('--ego_frame', action='store_true',
                   help='render in the ego frame (ego at origin, heading up); outputs risk_ego.*')
    p.add_argument('--zoom', type=float, default=None,
                   help='ego-frame half-window in metres (default: bundle zoom)')
    p.add_argument('--fps', type=float, default=2.0, help='video playback frame rate')
    p.add_argument('--time_window', type=int, default=1,
                   help='temporal slack (in horizon steps) for a crossing to count: agent and '
                        'ego must reach the crossing within this many steps of each other. '
                        '0 = strictly simultaneous; -1 = ignore timing (pure geometric crossing)')
    args = p.parse_args()

    bundle = load_bundle(args.pred_file)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pred_file))
    os.makedirs(out_dir, exist_ok=True)
    print(f'Loaded {len(bundle["frames"])} frames from {args.pred_file}  '
          f'(scene {bundle["meta"].get("scene")})')

    print(f'  time_window = {args.time_window} step(s)'
          + ('  (strictly simultaneous)' if args.time_window == 0
             else '  (timing ignored, pure geometric)' if args.time_window < 0
             else f'  (~±{args.time_window * bundle["meta"].get("dt", 0.5):.1f}s slack)'))
    rows, per_frame = evaluate(bundle, time_window=args.time_window)
    summarize(rows)
    if rows:
        write_csv(rows, os.path.join(out_dir, 'risk_crossings.csv'))
    if args.viz:
        render(bundle, per_frame, out_dir, fps=args.fps, fmt=args.format,
               ego_frame=args.ego_frame, zoom=args.zoom)


if __name__ == '__main__':
    main()
