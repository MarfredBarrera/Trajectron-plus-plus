"""The same frozen-timestep mixture as `gaussian_propagation`, one step upstream: in the
space the decoder actually emits, before the dynamics turn it into positions.

Why look here at all
--------------------
Trajectron++ does not predict positions. For a vehicle it predicts a Gaussian mixture over
CONTROLS -- heading rate dphi [rad/s] and acceleration a [m/s^2] -- per horizon step, and
`Unicycle.integrate_distribution` rolls that mixture through the dynamics to get the position
ellipses everything else in this repo draws. Two things are only visible on this side of that
integration:

  * What the model is actually uncertain about. A position cloud that grows into a fan can
    come from uncertainty about speed, about turning, or about both, and the shapes it makes
    are dominated by the agent's current velocity and heading. In control space the question
    is asked directly: the horizontal spread is doubt about the turn, the vertical spread is
    doubt about the throttle, and neither is scaled by how fast the agent happens to be going.

  * The dynamic limits, which live in this space and nowhere else. `Unicycle.dynamic` clamps
    every control to hyperparams['dynamic'][<type>]['limits'] (for this checkpoint: a in
    [-5, 4], dphi in [-0.7, 0.7]) before integrating. The clamp bounds the samples but not
    the covariance recursion -- `compute_jacobian` and `compute_control_jacobian` are still
    evaluated at the raw mean control -- so any mixture mass outside the box is mass the
    position ellipses account for and no trajectory the model can produce ever realises. The
    limits ARE the axes of every panel here, and the mass outside them is reported per
    timestep, in the panel title and in the CSV.

What it writes, per frozen timestep, into <bundle dir>/action_propagation/:

  action_t<t>.png   one control-plane panel per agent: the most likely latent mode's 1-sigma
                    contour at every horizon step, and its track. Colour is the mode's weight,
                    on the same per-agent heat map as the position study -- the same mode is
                    the same colour in both figures -- and horizon time is stroke weight,
                    thin near to thick far. `--modes n` draws the n heaviest instead of one;
                    all 25 at all 10 steps in a box two units across is unreadable however it
                    is coloured, and most of what it adds is modes the model barely believes.

  action_t<t>.gif   the same panel unrolled along the horizon, one frame per step, the axes
                    held fixed so the growth is the thing that moves.

and action_summary.csv across all of them, which is computed from the FULL mixture -- the
drawing cutoff never reaches the numbers.

One agent per figure by default (`--max_agents`, `--agents`), since a control panel small
enough to tile is too small to read; the default is roster #2, the amber agent.

Reads a stored bundle only -- no model, no GPU -- but needs one written after the control
mixture was plumbed through (`ctrl_mus` / `ctrl_covs`; see src/bundle.py), by a run launched
with `--style gaussian` or `--style both`.

Usage (from experiments/unity_lidar_sim/):
    python src/unity_online.py --config configs/intersection_probe.yaml --gpu 0 --style both
    python src/investigations/action_propagation.py
    python src/investigations/action_propagation.py --agents 1,3 --times 44,56
"""
import os
import sys
import csv
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import multivariate_normal

from bundle import load_bundle
from visualization.traj_viz import assemble_gif, assemble_mp4, ellipse, mode_cmap
from investigations.gmm_stats import effective_modes
from visualization.colors import DEFAULT_CENTER, closest_approach, junction_palette
from investigations.gaussian_propagation import (
    INK, MUTED, _add_mode_colorbars, _style_axes, junction_roster,
    short_names, resolve_agents, print_roster, mode_weights)


# What the two control axes mean, per dynamics model. The bundle records the model's
# `dynamic` config (name and limits per node type), so the axes and the clamp box come from
# the run itself rather than from an assumption about which checkpoint produced it.
CONTROL_AXES = {
    'Unicycle': (('dφ [rad/s]', 'min_heading_change', 'max_heading_change'),
                 ('a [m/s²]', 'min_a', 'max_a')),
    'SingleIntegrator': (('a$_x$ [m/s²]', None, None), ('a$_y$ [m/s²]', None, None)),
}
DEFAULT_AXES = (('u$_0$', None, None), ('u$_1$', None, None))


def control_axes(meta, node_type):
    """((label, min_key, max_key) per axis, {limit key: value}) for a node type's dynamics."""
    dyn = (meta.get('dynamic') or {}).get(node_type, {})
    return CONTROL_AXES.get(dyn.get('name'), DEFAULT_AXES), (dyn.get('limits') or {})


def clamp_box(axes, limits):
    """(lo, hi) per control axis from the dynamics limits; NaN where the config sets none."""
    return np.array([[limits.get(lo, np.nan), limits.get(hi, np.nan)]
                     for _, lo, hi in axes], float)


# --------------------------------------------------------------------------- #
# The numbers this study adds to the position one                              #
# --------------------------------------------------------------------------- #
def mass_outside(mus, covs, pis, box):
    """Mixture probability that the control falls outside `box`, per horizon step.

    The exact rectangle probability of each component -- `Unicycle.dynamic` clamps per axis
    independently, and the components are correlated, so a product of per-axis marginals
    would be wrong in both directions. Axes the config leaves unbounded are treated as
    infinite, which is what the clamp does with them.
    """
    lo = np.where(np.isnan(box[:, 0]), -np.inf, box[:, 0])
    hi = np.where(np.isnan(box[:, 1]), np.inf, box[:, 1])
    if not np.isfinite(lo).any() and not np.isfinite(hi).any():
        return np.zeros(len(mus))
    out = np.zeros(len(mus))
    for h in range(len(mus)):
        inside = 0.0
        for k in range(mus.shape[1]):
            rv = multivariate_normal(mean=mus[h, k], cov=covs[h, k], allow_singular=True)
            # P(lo < x < hi) by inclusion-exclusion on the 2-D CDF
            p = (rv.cdf(hi) - rv.cdf([lo[0], hi[1]])
                 - rv.cdf([hi[0], lo[1]]) + rv.cdf(lo))
            inside += pis[h, k] * float(np.clip(p, 0.0, 1.0))
        out[h] = 1.0 - inside
    return out


def analyse_controls(node, meta, dt):
    """Per-agent control-space arrays: the mixture, its moments, and the clamp box."""
    mus = np.asarray(node['ctrl_mus'], float)
    if mus.size == 0:
        raise ValueError(f"node {node['id']} carries no control GMM -- the bundle predates "
                         f"`ctrl_mus`/`ctrl_covs`, or the run was not --style gaussian/both")
    covs = np.asarray(node['ctrl_covs'], float)
    pis = np.asarray(node['dist_pis'], float)   # shared with the position mixture, by design
    axes, limits = control_axes(meta, node['type'])
    box = clamp_box(axes, limits)

    sigma = np.sqrt(np.einsum('hk,hkii->hi', pis, covs)
                    + np.einsum('hk,hki,hki->hi', pis, mus - np.einsum('hk,hkd->hd', pis, mus)
                                [:, None, :], mus - np.einsum('hk,hkd->hd', pis, mus)[:, None, :]))
    return {'id': node['id'], 'type': node['type'], 'mus': mus, 'covs': covs, 'pis': pis,
            'mean': np.einsum('hk,hkd->hd', pis, mus), 'sigma': sigma,
            'axes': axes, 'limits': limits, 'box': box,
            'outside': mass_outside(mus, covs, pis, box),
            'perplexity': effective_modes(pis)}


# --------------------------------------------------------------------------- #
# The figure                                                                   #
# --------------------------------------------------------------------------- #
def control_view(agents, scale=1.0):
    """Axis limits: the clamp box itself, optionally widened by `scale`.

    The panel bounds ARE the dynamic limits, so "is this control executable" is read off the
    frame instead of from a rectangle somewhere inside it, and the whole panel is spent on the
    region the integrator can realise. Mass outside is clipped -- there is a lot of it (see
    `mass_outside`, reported per timestep and in the CSV) and it is not worth the pixels: a
    few near-zero-weight modes sit two orders of magnitude past the limit, and framing for
    them shrinks everything that matters to a dot.

    An axis the config leaves unbounded has no box, so it falls back to mean +/- 2 sigma.
    """
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    box_lo = np.full(2, np.inf)
    box_hi = np.full(2, -np.inf)
    for a in agents:
        box_lo = np.minimum(box_lo, np.where(np.isnan(a['box'][:, 0]), np.inf, a['box'][:, 0]))
        box_hi = np.maximum(box_hi, np.where(np.isnan(a['box'][:, 1]), -np.inf, a['box'][:, 1]))
        lo = np.minimum(lo, (a['mean'] - 2 * a['sigma']).min(axis=0))
        hi = np.maximum(hi, (a['mean'] + 2 * a['sigma']).max(axis=0))
    box_lo = np.where(np.isfinite(box_lo), box_lo, lo)
    box_hi = np.where(np.isfinite(box_hi), box_hi, hi)
    mid, half = 0.5 * (box_lo + box_hi), 0.5 * scale * np.maximum(box_hi - box_lo, 1e-3)
    return mid - half, mid + half


def top_modes(a, n):
    """The `n` heaviest latent modes, strongest first."""
    pis, _ = mode_weights(a)
    return list(np.argsort(pis)[::-1][:max(n, 1)]), pis


def _draw_agent(ax, a, cmap, norm, steps, view, n_modes, n_stds, dtv, label_ends=True):
    """One agent's control mixture over `steps`, in one panel.

    Only the heaviest `n_modes` are drawn. Every mode at every step -- 25 x 10 contours in a
    box two units across -- is unreadable however it is coloured, and the modes that survive
    thresholding are mostly ones the model barely believes: the most likely mode alone answers
    "what is the model actually proposing", which is what this panel is for.

    Colour is the mode's mixture weight, on the same per-agent heat map as the position study,
    so one mode is one colour in both figures and on both sides of the dynamics -- colour
    means likelihood everywhere or it means nothing. That leaves horizon time to stroke
    weight (thin near, thick far), the track through the mode's own means, and the labels at
    the two ends.
    """
    ph = len(a['mus'])
    steps = list(steps)
    for rank, k in enumerate(top_modes(a, n_modes)[0]):
        c = cmap(norm(float(mode_weights(a)[0][k])))
        pts = a['mus'][steps, k]
        ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=1.0, alpha=0.55, zorder=500)
        for h in steps:
            lw = 0.7 + 1.1 * (h / max(ph - 1, 1))
            for ns in n_stds:
                ax.add_patch(ellipse(a['mus'][h, k], a['covs'][h, k], ns, facecolor='none',
                                      edgecolor=c, lw=lw, zorder=300 + h))
            ax.scatter(*a['mus'][h, k], s=14, color=c, edgecolors='none', zorder=600 + h)
        # Only on the heaviest mode: one pair of end labels per panel, not one per mode. And
        # only where the two ends are far enough apart to read as two -- a mode that holds its
        # control over the horizon puts both labels on the same point.
        span = np.linalg.norm(np.asarray(view[1]) - np.asarray(view[0]))
        ends = [steps[-1]] if (len(steps) < 2 or np.linalg.norm(
            a['mus'][steps[-1], k] - a['mus'][steps[0], k]) < 0.06 * span) else [steps[0],
                                                                                 steps[-1]]
        if label_ends and rank == 0:
            for h, txt in ((h, f'{(h + 1) * dtv:.1f}s') for h in ends):
                ax.annotate(txt, a['mus'][h, k], textcoords='offset points', xytext=(6, -10),
                            fontsize=9, color=MUTED, zorder=950,
                            path_effects=[pe.Stroke(linewidth=2.4, foreground='w'), pe.Normal()])
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(view[0][0], view[1][0])
    ax.set_ylim(view[0][1], view[1][1])
    ax.set_xlabel(a['axes'][0][0])
    ax.set_ylabel(a['axes'][1][0])
    _style_axes(ax)


def _action_figure(t, agents, hues, meta, steps, subtitle, view, n_modes, n_stds):
    """One panel per agent; the still and every animation frame come through here."""
    dtv = meta.get('dt', 0.5)
    n = len(agents)
    steps = list(steps)
    fig, axes = plt.subplots(1, n, squeeze=False, figsize=(5.6 * n + 1.3, 5.9))
    fig.subplots_adjust(left=0.10 / n + 0.02, right=0.855, top=0.855, bottom=0.10, wspace=0.24)
    cmaps = [mode_cmap(hue) for hue in hues]
    norms = [mode_weights(a)[1] for a in agents]
    for ax, a, cm, nm in zip(axes[0], agents, cmaps, norms):
        _draw_agent(ax, a, cm, nm, steps, view, n_modes, n_stds, dtv)
        pis = top_modes(a, n_modes)[1]
        heavy = ', '.join(f'π={pis[k]:.2f}' for k in top_modes(a, n_modes)[0])
        # ax.set_title(f"{a['id']}   {heavy}   {100 * a['outside'][steps[-1]]:.0f}% out",
        #              fontsize=11.5, color=INK)
    _add_mode_colorbars(fig, agents, cmaps, norms)
    fig.suptitle(f't = {t}  ·  {t * dtv:.1f} s\n{subtitle}', fontsize=13, color=INK)
    return fig


def draw_action(t, agents, hues, meta, out_path, view, n_modes=1, n_stds=(1.0,)):
    """The still: the whole horizon in one panel per agent."""
    ph = meta.get('ph', 10)
    fig = _action_figure(t, agents, hues, meta, range(ph),
                         f'{"most likely mode" if n_modes == 1 else f"top {n_modes} modes"}'
                         f'  ·  axes = control limits', view, n_modes, n_stds)
    fig.savefig(out_path, dpi=120, facecolor='white')
    plt.close(fig)
    return out_path


def draw_action_animation(t, agents, hues, meta, out_base, view, n_modes=1, n_stds=(1.0,),
                          fps=2.0, fmt='gif', hold=3, keep_frames=False):
    """The loop: one frame per horizon step, accumulating, axes fixed."""
    dtv, ph = meta.get('dt', 0.5), meta.get('ph', 10)
    frame_dir = f'{out_base}_frames'
    os.makedirs(frame_dir, exist_ok=True)
    paths = []
    for h in range(ph):
        fig = _action_figure(t, agents, hues, meta, range(h + 1),
                             f'step {h + 1}/{ph}  ·  {(h + 1) * dtv:.1f} s ahead  ·  '
                             f'axes = control limits', view, n_modes, n_stds)
        out = os.path.join(frame_dir, f'step_{h + 1:02d}.png')
        fig.savefig(out, dpi=100, facecolor='white')
        plt.close(fig)
        paths.append(out)

    seq = paths + [paths[-1]] * max(0, hold)
    outputs = []
    if fmt in ('gif', 'both'):
        outputs.append(assemble_gif(seq, out_base + '.gif', fps))
    if fmt in ('mp4', 'both'):
        outputs.append(assemble_mp4(seq, out_base + '.mp4', fps))
    if not keep_frames:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return outputs


# --------------------------------------------------------------------------- #
SUMMARY_FIELDS = ['t', 't_sec', 'id', 'eff_modes',
                  'dphi_mean_h1', 'dphi_sigma_h1', 'dphi_mean_end', 'dphi_sigma_end',
                  'a_mean_h1', 'a_sigma_h1', 'a_mean_end', 'a_sigma_end',
                  'outside_h1', 'outside_end', 'outside_max', 'mean_outside_limits_steps']


def summary_row(t, dtv, a):
    end = len(a['mean']) - 1
    box = a['box']
    lo = np.where(np.isnan(box[:, 0]), -np.inf, box[:, 0])
    hi = np.where(np.isnan(box[:, 1]), np.inf, box[:, 1])
    # how often the mixture *mean* itself is out of bounds: not a tail, the centre
    mean_out = int(np.sum(np.any((a['mean'] < lo) | (a['mean'] > hi), axis=1)))
    return {'t': t, 't_sec': round(t * dtv, 2), 'id': a['id'],
            'eff_modes': round(float(a['perplexity'][0]), 2),
            'dphi_mean_h1': round(float(a['mean'][0, 0]), 4),
            'dphi_sigma_h1': round(float(a['sigma'][0, 0]), 4),
            'dphi_mean_end': round(float(a['mean'][end, 0]), 4),
            'dphi_sigma_end': round(float(a['sigma'][end, 0]), 4),
            'a_mean_h1': round(float(a['mean'][0, 1]), 4),
            'a_sigma_h1': round(float(a['sigma'][0, 1]), 4),
            'a_mean_end': round(float(a['mean'][end, 1]), 4),
            'a_sigma_end': round(float(a['sigma'][end, 1]), 4),
            'outside_h1': round(float(a['outside'][0]), 4),
            'outside_end': round(float(a['outside'][end]), 4),
            'outside_max': round(float(a['outside'].max()), 4),
            'mean_outside_limits_steps': mean_out}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', default='unity_out/intersection_probe/predictions_online.pkl',
                   help='prediction bundle written by a --style gaussian/both run')
    p.add_argument('--out_dir', default=None,
                   help='output directory (default: <bundle dir>/action_propagation)')
    p.add_argument('--times', default='40,44,48,52,56,60',
                   help='comma-separated timesteps to freeze')
    # Roster #2 is the amber agent -- hue follows roster position (see gaussian_propagation),
    # so naming the position names the colour, and this study's figures stay the amber ones.
    p.add_argument('--agents', default='2',
                   help='which agents to draw, comma-separated: a position in the roster this '
                        'run prints (2), a short name / unique tail of the id (656), or the '
                        'full id. Default = 2, the amber agent')
    p.add_argument('--center', default=','.join(str(c) for c in DEFAULT_CENTER),
                   help='intersection centre "x,y" in map coords (selection only)')
    p.add_argument('--select_radius', type=float, default=25.0,
                   help='how close to the centre an agent must come to be studied [m]')
    p.add_argument('--max_agents', type=int, default=1,
                   help='cap on agents drawn (default 1: one agent per figure keeps the '
                        'control panels large enough to read). --agents picks which')
    p.add_argument('--n_std', default='1', help='contour(s) to draw, in sigmas')
    p.add_argument('--modes', type=int, default=1,
                   help='how many latent modes to draw, heaviest first (default 1: the most '
                        'likely mode). Every mode still counts in the reported numbers')
    p.add_argument('--view_scale', type=float, default=1.0,
                   help='axis limits as a multiple of the dynamic limits (1 = the limits are '
                        'the axes; >1 to see how far outside them the mixture reaches)')
    p.add_argument('--anim', default='gif', choices=['gif', 'mp4', 'both', 'none'],
                   help='per-timestep animation of the horizon (one frame per horizon step)')
    p.add_argument('--anim_fps', type=float, default=2.0, help='animation frame rate')
    p.add_argument('--anim_hold', type=int, default=3,
                   help='extra frames to hold the end of the horizon before the loop restarts')
    p.add_argument('--keep_frames', action='store_true',
                   help='keep the per-step PNGs the animation is assembled from')
    args = p.parse_args()

    bundle = load_bundle(args.bundle)
    meta, frames = bundle['meta'], bundle['frames']
    dtv = meta.get('dt', 0.5)
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    center = np.array([float(v) for v in args.center.split(',')])
    times = [int(v) for v in args.times.split(',')]
    n_stds = tuple(float(v) for v in str(args.n_std).split(','))
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.bundle), 'action_propagation')
    os.makedirs(out_dir, exist_ok=True)

    by_t = {f['t']: f for f in frames}
    missing = [t for t in times if t not in by_t]
    if missing:
        raise SystemExit(f'timestep(s) {missing} are not in {args.bundle} '
                         f'(it holds t={frames[0]["t"]}..{frames[-1]["t"]})')

    # agent selection is shared with the position study, so #2 and "740" mean the same
    # vehicle in the same colour in both sets of figures
    roster = junction_roster(frames, off, center, args.select_radius)
    if not roster:
        raise SystemExit(f'no agent comes within {args.select_radius:g} m of {tuple(center)} '
                         f'-- widen --select_radius')
    roster_ids = [i for i, _ in roster]
    all_ids = sorted(closest_approach(frames, off, center))
    at_times = set(closest_approach(frames, off, center, set(times)))
    names = short_names(roster_ids)
    if args.agents:
        ids = resolve_agents(args.agents, roster_ids, names, all_ids)
        names.update(short_names(sorted(set(roster_ids) | set(ids))))
    else:
        ids = [i for i in roster_ids if i in at_times][:args.max_agents]
        if not ids:
            raise SystemExit(f'none of the {len(roster)} agents at the junction is predicted '
                             f'at t in {times} -- pick other --times')
    hue_of = junction_palette(frames, off, center, args.select_radius)
    print_roster(roster, names, set(ids), at_times)

    # one view for every panel and every frame: the control plane does not move with the
    # agent, so a shared frame makes the timesteps directly comparable
    per_t = {}
    for t in times:
        present = {nd['id']: nd for nd in by_t[t]['nodes']}
        try:
            agents = [analyse_controls(present[i], meta, dtv) for i in ids if i in present]
        except (ValueError, KeyError):
            raise SystemExit(
                f'{args.bundle} carries no control mixture. Bundles written before `ctrl_mus`'
                f'/`ctrl_covs` existed hold only the integrated position GMM, which cannot be '
                f'turned back into controls -- re-run the probe to record it:\n'
                f'  python src/unity_online.py --config configs/intersection_probe.yaml '
                f'--gpu 0 --style both')
        if agents:
            per_t[t] = agents
        else:
            print(f'  t={t}: none of the studied agents is predicted here -- skipped')
    if not per_t:
        raise SystemExit('nothing to draw')
    view = control_view([a for agents in per_t.values() for a in agents], scale=args.view_scale)

    rows, outputs = [], []
    for t, agents in per_t.items():
        hues = [hue_of[a['id']] for a in agents]
        rows += [summary_row(t, dtv, a) for a in agents]
        outputs.append(draw_action(t, agents, hues, meta,
                                   os.path.join(out_dir, f'action_t{t:03d}.png'), view,
                                   n_modes=args.modes, n_stds=n_stds))
        if args.anim != 'none':
            outputs += draw_action_animation(t, agents, hues, meta,
                                             os.path.join(out_dir, f'action_t{t:03d}'), view,
                                             n_modes=args.modes, n_stds=n_stds,
                                             fps=args.anim_fps, fmt=args.anim,
                                             hold=args.anim_hold,
                                             keep_frames=args.keep_frames)
        print(f'  t={t:>3} ({t * dtv:>4.1f}s): ' + '   '.join(
            f"{a['id']} dφ={a['mean'][-1, 0]:+.2f}±{a['sigma'][-1, 0]:.2f}rad/s "
            f"a={a['mean'][-1, 1]:+.2f}±{a['sigma'][-1, 1]:.2f}m/s² "
            f"outside={100 * a['outside'][-1]:4.1f}%" for a in agents))

    csv_path = os.path.join(out_dir, 'action_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(outputs)} figures/animations and {csv_path}')
    worst = max(rows, key=lambda r: r['outside_max'])
    print(f'  most mass outside the clamp box: {100 * worst["outside_max"]:.1f}% '
          f'({worst["id"]} at t={worst["t"]}) -- that mass is in the position covariance '
          f'but in no trajectory the integrator can produce (see the module docstring)')


if __name__ == '__main__':
    main()
