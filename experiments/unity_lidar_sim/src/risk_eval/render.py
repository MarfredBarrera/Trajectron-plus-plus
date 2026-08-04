"""The proximity-risk overlay video: the ego's disc, entering samples, per-agent risk labels.

Split from the metric (proximity.py) so scoring a frame inside the online loop does not pull
in matplotlib. Consumes the `per_frame` map that `proximity.evaluate`/`evaluate_frame`
produce, so a live run and a re-score of a stored bundle render identically.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from risk_eval.proximity import DEFAULT_RADIUS
from visualization.traj_viz import (_build_map_rgba, _draw_map_background, _assemble,
                                    ego_transform, _identity_transform)


def render(bundle, per_frame, out_dir, fps=2.0, fmt='gif', ego_frame=False, zoom=None,
           radius=DEFAULT_RADIUS):
    """One PNG per frame + a GIF and/or MP4 (`fmt` in {gif, mp4, both}): map, the ego's
    keep-out disc (bold red), samples that reach into it in red, the rest faded in the agent
    colour, each agent labelled with its proximity risk. Only what the metric actually reads
    is drawn -- the ego's future path is stored in the bundle but deliberately not shown.
    `ego_frame` renders in the ego's frame (ego at the origin, heading up), cropped to
    +/- `zoom` metres."""
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
        ego_xy, fr = per_frame[t]
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
        # The ego's keep-out disc. Both transforms are rigid, so a circle stays a circle and
        # only its centre needs mapping.
        c = T(ego_xy)
        ax.add_patch(Circle(c, radius, fill=True, facecolor='#D62728', alpha=0.07, zorder=690))
        ax.add_patch(Circle(c, radius, fill=False, edgecolor='#D62728', lw=2.4, zorder=700,
                            path_effects=[pe.Stroke(linewidth=4.2, foreground='w'), pe.Normal()]))
        ax.scatter([c[0]], [c[1]], marker='^', s=180, color='#D62728',
                   edgecolors='k', linewidths=1.0, zorder=760)
        for nd in rec['nodes']:
            if nd['id'] not in fr:
                continue
            mask, samples_w, col = fr[nd['id']]
            samples_p = T(samples_w)
            cur = samples_p[0, 0]
            if (~mask).any():
                ax.add_collection(LineCollection(samples_p[~mask], colors=col, linewidths=0.5,
                                                 alpha=0.05, zorder=400))
            if mask.any():
                ax.add_collection(LineCollection(samples_p[mask], colors='#D62728',
                                                 linewidths=0.6, alpha=0.14, zorder=450))
            ax.scatter([cur[0]], [cur[1]], s=30, color=col, edgecolors='k', linewidths=0.6,
                       zorder=650)
            # Three decimals: the risk is a fraction of `num_samples` entering the disc, so at
            # the usual 1000 samples the thousandths place is exactly the metric's resolution.
            # Two decimals quantized most agents in a low-risk scene to a flat 0.00.
            ax.annotate(f'{mask.mean():.3f}', (cur[0], cur[1]), textcoords='offset points',
                        xytext=(6, 6), fontsize=8, color='k', zorder=660,
                        path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        if use_ego:
            ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
            ax.set_xlabel('ego x (right) [m]'); ax.set_ylabel('ego y (forward) [m]')
            title = (f'Ego-proximity risk, R = {radius:g} m (ego frame)  |  '
                     f't = {t}  ({t * meta.get("dt", 0.5):.1f}s)')
        else:
            ax.set_xlim(meta['xlim']); ax.set_ylim(meta['ylim'])
            ax.set_xlabel('map x [m]'); ax.set_ylabel('map y [m]')
            title = (f'Ego-proximity risk, R = {radius:g} m  |  '
                     f't = {t}  ({t * meta.get("dt", 0.5):.1f}s)')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        handles = [Line2D([], [], color='#D62728', lw=2.4, label=f'Ego disc (R = {radius:g} m)'),
                   Line2D([], [], color='#D62728', lw=1.0, alpha=0.8, label='Entering samples'),
                   Line2D([], [], color='#888', lw=1.0, alpha=0.7, label='Non-entering samples')]
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
