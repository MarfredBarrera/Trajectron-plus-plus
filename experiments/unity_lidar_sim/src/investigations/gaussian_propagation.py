"""Freeze a timestep and watch the predicted Gaussian mixture propagate over the horizon.

The videos the pipeline normally produces animate `t`, which is exactly the wrong axis for
asking *what the prediction is*: the distribution at a single `t` flashes past in one frame.
This study does the opposite -- it holds `t` still and animates the *horizon* instead, so the
whole predicted distribution for one instant is visible at once (as a still) and also step by
step (as a short loop), and then repeats that for a series of `t` so the change between
instants is a comparison between panels rather than a memory of a video.

It is scoped to the four-way intersection at ~(157, 71) in `sweep_s1_rail`, where the ego
drives north through the junction while other vehicles turn across it and wait at it. The ego
is deliberately not part of the model's scene graph here (see configs/intersection_probe.yaml,
`ego_conditioning: false`): the distributions drawn are what the model predicts from the
other agents and the map alone.

Per frozen timestep it writes two figures and one animation:

  propagation_t<t>.png  the intersection, with every latent mode's 1-sigma contour at every
                        horizon step, plus that mode's own track through the horizon. Each
                        mode is drawn as an outline whose COLOUR is its mixture weight, on a
                        per-agent heat map with its own colorbar -- so a mode is one colour
                        from the first horizon step to the last, and where two modes cross,
                        the crossing still reads as two modes. (Translucent fills were tried
                        first and failed structurally: overlapping patches composite, so the
                        darkness at a point was the number of ellipses stacked there rather
                        than any one mode's weight.) Nothing else is drawn: no samples, no
                        most-likely path, no mixture mean or moment-matched ellipse -- the
                        panel is the model's components, the agent's history and its logged
                        future, and that is all. Everything omitted here is quantified in the
                        diagnostics figure instead.

  propagation_t<t>.gif  the same panel unrolled along the horizon: one frame per horizon
                        step, each step's contours added to the ones already drawn and never
                        dimmed or recoloured afterwards, with the logged future advancing
                        marker by marker alongside them. `t` is still frozen -- the animated
                        axis is how far ahead the prediction reaches, not simulation time --
                        so the splitting of the modes reads as motion instead of as overlap,
                        and the final frame is the .png above.

  diagnostics_t<t>.png  the same distribution as numbers: the within/between variance split,
                        longitudinal vs lateral spread in the agent's own heading frame, the
                        analytic mixture against the sampled one, the logged future's error
                        against the predicted spread, and the latent posterior's weights.

and across timesteps, growth_vs_time.png (how the spread and its composition change as the
agents approach, enter and leave the junction) plus propagation_summary.csv.

Reads a stored bundle only -- no model, no GPU. The bundle must come from a run launched with
`--style gaussian` or `--style both`, otherwise it holds no GMM to draw.

Usage (from experiments/unity_lidar_sim/):
    python src/unity_online.py --config configs/intersection_probe.yaml --gpu 0 --style both
    python src/investigations/gaussian_propagation.py
    python src/investigations/gaussian_propagation.py --anim both --anim_fps 3

Every run prints the junction roster -- position, short name, full id -- and `--agents` takes
any of the three, so an agent can be switched on or off without typing a simulator id:

    python src/investigations/gaussian_propagation.py --agents 1,3       # by position
    python src/investigations/gaussian_propagation.py --agents 740,642   # by short name
    python src/investigations/gaussian_propagation.py --times 44,48,52 --agents 740

Positions and colours come from the roster, which is computed over the whole bundle, so #2 is
the same vehicle in the same colour whatever `--times` or `--agents` are set to.
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
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from bundle import load_bundle
from visualization.colors import (DEFAULT_CENTER, closest_approach,  # noqa: F401
                                  junction_palette, junction_roster)
from visualization.traj_viz import (_build_map_rgba, _draw_map_background,
                                   assemble_gif, assemble_mp4, ellipse, mode_cmap)
from investigations.gmm_stats import analyse_node


# Agent identity is hue, fixed by roster position and never recycled. On the map panel a
# mode's *shade* within that hue is its mixture weight (a heat map per agent, see
# `mode_cmap`), and horizon time is carried by the mode's own track rather than by colour;
# in the diagnostics and growth figures, where there is one curve per agent, the hue is used
# flat. Every agent is also labelled, so identity never rests on colour alone.
# The hues, the junction and the roster rule itself live in visualization.colors, so figures
# outside this study (the risk time series) can draw the same agent in the same colour.
INK, MUTED, GRID = '#1A1A1A', '#5A5A5A', '#D8D8D8'
EGO_COLOR = '#8A8A8A'


def _hue_ramp(hex_color, n, lightest=0.72):
    """`n` shades of one hue, light -> full strength: the sequential ramp for horizon time."""
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    ws = np.linspace(lightest, 0.0, n) if n > 1 else np.zeros(1)
    return [tuple(base + (1.0 - base) * w) for w in ws]


# `mode_cmap` (weight -> colour, per agent) and `ellipse` (n-sigma contour) live in
# visualization.traj_viz: the renderer draws its Gaussian style the same way this study does,
# so the two share one definition rather than drifting apart. Each agent's colorbar here is
# drawn with its own π ticks, so the per-agent scale is never left implicit.


def _style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


# --------------------------------------------------------------------------- #
# Selection                                                                     #
# --------------------------------------------------------------------------- #
# `closest_approach` / `junction_roster` are imported from visualization.colors: the roster is
# what fixes agent hue, so the ordering rule and the palette have to be the same object.
def short_names(ids, min_len=3):
    """{id: shortest tail of the id that is unique among `ids`} -- e.g. -24740 -> '740'.

    The bundle's agent ids are the simulator's, five or six digits of no mnemonic value, and
    typing one into `--agents` correctly is harder than it should be for a flag whose whole
    job is picking who to look at. The last few digits are enough to be unambiguous here, and
    the roster printed at startup shows each agent's short name next to its full id.
    """
    ids = list(ids)
    for n in range(min_len, max((len(i) for i in ids), default=min_len) + 1):
        names = {i: i[-n:] for i in ids}
        if len(set(names.values())) == len(ids):
            return names
    return {i: i for i in ids}


TAIL_MIN = 3   # shortest id tail --agents will match on; below this, digits are too generic


def resolve_agents(spec, roster_ids, names, all_ids):
    """Turn a `--agents` token list into agent ids.

    Each token may be a roster position (`2`), an agent's short name or any unique tail of
    its id (`740`), or the full id, with or without its minus sign (argparse claims a leading
    `-` for itself, so `24740` is the form that always works). Roster positions win over
    tails when a token could be read either way -- they are what the startup listing numbers
    -- and the clash is reported rather than silently resolved, with the name to use instead.
    """
    picked = []
    for tok in (s.strip() for s in spec.split(',')):
        if not tok:
            continue
        by_tail = ([i for i in roster_ids if i == tok or i.endswith(tok)] or
                   [i for i in all_ids if i == tok or i.endswith(tok)]
                   if len(tok.lstrip('-')) >= TAIL_MIN else [])
        if tok.isdigit() and 1 <= int(tok) <= len(roster_ids):
            pick = roster_ids[int(tok) - 1]
            clash = [i for i in by_tail if i != pick]
            if clash:
                print(f'  note: --agents {tok} taken as roster #{tok} ({pick}); write '
                      f'"{names.get(clash[0], clash[0])}" for {clash[0]}')
        elif len(by_tail) == 1:
            pick = by_tail[0]
        elif by_tail:
            raise SystemExit(f'--agents {tok} is ambiguous: matches '
                             f'{", ".join(by_tail)}. Use a longer tail or the full id.')
        elif len(tok.lstrip('-')) < TAIL_MIN:
            raise SystemExit(f'--agents {tok} is neither a roster position (1..'
                             f'{len(roster_ids)}) nor long enough to name an agent: '
                             f'use at least {TAIL_MIN} digits, e.g. {names[roster_ids[0]]}.')
        else:
            raise SystemExit(f'--agents {tok} matches no agent in this bundle. Run without '
                             f'--agents to list who is at the junction.')
        if pick not in picked:
            picked.append(pick)
    return picked


def print_roster(roster, names, studied, at_times):
    """The startup listing: how a short name or a position maps onto an agent id."""
    print('At the junction, nearest first  (--agents takes the #, the name, or the id):')
    for n, (i, d) in enumerate(roster, 1):
        note = '' if i in at_times else '   (not at these timesteps)'
        print(f'  {"*" if i in studied else " "} #{n}  {names[i]:>6}  {i:>8}   '
              f'{d:5.1f} m away{note}')
    print(f'  * = drawn in this run ({len(studied)} of {len(roster)}); '
          f'--agents picks a different set, --max_agents a different cap')


# --------------------------------------------------------------------------- #
# Figure 1: the distribution on the map                                        #
# --------------------------------------------------------------------------- #
def mode_weights(a):
    """The latent posterior of agent `a` as one weight per mode, and its heat-map scale.

    `dist_pis` is a function of the encoder alone, so it is identical at every horizon step
    (see investigations/gmm_stats): row 0 is the whole story, and taking it once means a mode
    keeps exactly one colour from the start of the horizon to the end -- which is what makes
    a mode followable across steps and across frames of the animation.
    """
    pis = np.asarray(a['pis'], float)[0]
    return pis, matplotlib.colors.Normalize(vmin=0.0, vmax=max(float(pis.max()), 1e-9))


def _draw_modes(ax, a, cmap, norm, steps, pi_threshold, n_stds, ph, track=True):
    """Agent `a`'s latent modes over `steps` of the horizon: one outline per mode per step.

    Weight is COLOUR, not opacity. Filled translucent ellipses were unreadable here for a
    structural reason: 25 modes x 10 steps of overlapping patches composite, so the darkness
    at a point is the number of ellipses stacked there rather than any one mode's weight, and
    two modes crossing produce a third, darker shape that belongs to neither. Opaque contours
    on a per-agent heat map (`mode_cmap`) do not composite: a mode is one colour, it is that
    colour wherever it goes, and where two modes cross the crossing stays legible as two.

    That leaves horizon time without a colour channel, so each mode also gets a `track`: a
    polyline through its own mean, from the agent's current position out to the last step
    drawn. The track is the answer to "where did this ellipse come from" -- following one mode
    through the horizon is reading one line, not matching shades between clouds.

    Modes are drawn weakest-first so the ones carrying the belief end up on top, and stroke
    weight grows along the horizon so the far end of a track reads as the far end.
    """
    pis, _ = mode_weights(a)
    steps = list(steps)
    for rank, k in enumerate(np.argsort(pis)):
        pi = float(pis[k])
        if pi < pi_threshold:
            continue
        c = cmap(norm(pi))
        z = 300 + rank
        if track and len(steps) > 1:
            # a chain of that mode's centres, dotted at each step: deliberately not a plain
            # line, which at this density would read as one more contour.
            pts = np.vstack([a['pos'], a['mus'][steps, k]])
            ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=1.0, alpha=0.95, zorder=z,
                    marker='o', ms=2.4, mfc=c, mec='none')
        for h in steps:
            lw = 0.6 + 1.0 * (h / max(ph - 1, 1))
            for ns in n_stds:
                ax.add_patch(ellipse(a['mus'][h, k], a['covs'][h, k], ns,
                                      facecolor='none', edgecolor=c, lw=lw, zorder=z))


def _add_mode_colorbars(fig, agents, cmaps, norms):
    """One thin colorbar per agent, top to bottom in panel order: colour -> mixture weight.

    Placed at fixed figure coordinates rather than stolen from the axes, so that the still
    and every frame of the animation get identical geometry (a GIF needs identical frames,
    and the two should be comparable anyway).
    """
    n = max(len(agents), 1)
    span = 0.74 / n
    for i, (a, cm, nm) in enumerate(zip(agents, cmaps, norms)):
        cax = fig.add_axes([0.895, 0.14 + (n - 1 - i) * span, 0.013, span * 0.72])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax)
        cb.set_label(f"{a['id']}  π", fontsize=8.5, color=MUTED)
        cb.ax.tick_params(labelsize=7.5, colors=MUTED)
        cb.outline.set_visible(False)


def _draw_context(ax, frame, agents, hues, center, view, gt_steps=None):
    """Everything on the map panel that does not depend on the horizon step: the ego, and
    each agent's history, current position and logged future.

    `gt_steps` truncates the logged future to that many steps (the animation reveals it one
    step at a time, so the truth walks through the distribution instead of giving away where
    it ends before the modes have got there); None draws all of it.
    """
    # # ego: context only. It is not an agent here -- nothing about these distributions was
    # # conditioned on it (configs/intersection_probe.yaml sets ego_conditioning: false).
    # ego = frame.get('ego')
    # if ego is not None:
    #     path = np.asarray(frame.get('ego_path') or [], float)
    #     if path.size:
    #         ax.plot(path[:, 0], path[:, 1], ':', color=EGO_COLOR, lw=1.4, zorder=200)
    #     ax.scatter(*ego[0], marker='^', s=190, color=EGO_COLOR, edgecolors='k',
    #                linewidths=0.8, zorder=800)

    for hue, a in zip(hues, agents):
        ax.plot(a['history'][:, 0], a['history'][:, 1], '-', color=hue, lw=2.0, alpha=0.55,
                zorder=510)
        fut = a['future'] if gt_steps is None else a['future'][:gt_steps]
        if len(fut):
            ax.plot(*np.vstack([a['pos'], fut]).T, '--', color='w', lw=1.8,
                    zorder=600, path_effects=[pe.Stroke(linewidth=3.4, foreground='k'),
                                              pe.Normal()])
            if gt_steps is not None:
                ax.scatter(*fut[-1], s=46, marker='o', facecolor='w', edgecolors='k',
                           linewidths=1.0, zorder=660)
        ax.scatter(*a['pos'], s=52, color=hue, edgecolors='k', linewidths=0.7, zorder=650)

    ax.set_xlim(center[0] - view, center[0] + view)
    ax.set_ylim(center[1] - view, center[1] + view)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    _style_axes(ax)


def _legend_handles(pi_threshold, n_stds, upto=None):
    contours = '/'.join(f'{n:g}' for n in n_stds)
    return [
        Ellipse((0, 0), 1, 1, facecolor='none', edgecolor=MUTED, lw=1.2,
                label=f'modes ≥{pi_threshold:g}, {contours}σ  ·  colour = weight'),
        Line2D([], [], color=MUTED, lw=1.0, label='mode mean'),
        # Line2D([], [], color='w', ls='--', lw=1.8, label='logged future (not an input)',
        #        path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]),
        # Line2D([], [], marker='^', color=EGO_COLOR, ls=':', ms=11, mec='k',
        #        label='ego + its path (outside the model)'),
    ]


def _propagation_panel(frame, agents, hues, meta, center, view, steps, subtitle,
                       pi_threshold, n_stds, map_bg, gt_steps=None):
    """Build the map panel for `steps` of the horizon. Returns the figure.

    The still and every frame of the animation come through here, so the two cannot drift
    apart: they differ only in which steps are passed and what the second title line says.
    Fixed margins rather than `bbox_inches='tight'` -- tight boxes change size with the
    title, and a GIF needs every frame identical.
    """
    dtv, ph = meta.get('dt', 0.5), meta.get('ph', 10)
    steps = list(steps)
    fig, ax = plt.subplots(figsize=(12.6, 10.5))
    fig.subplots_adjust(left=0.075, right=0.865, top=0.905, bottom=0.115)
    ax.set_facecolor('white')
    _draw_map_background(ax, *map_bg)
    _draw_context(ax, frame, agents, hues, center, view, gt_steps=gt_steps)

    cmaps = [mode_cmap(hue) for hue in hues]
    norms = [mode_weights(a)[1] for a in agents]
    for a, cm, nm in zip(agents, cmaps, norms):
        _draw_modes(ax, a, cm, nm, steps, pi_threshold, n_stds, ph)

        # Horizon time, at a few steps along the cloud: colour is spent on the weight, so
        # this and the mode tracks are what say how far ahead a contour is. Anchored to the
        # (undrawn) mixture mean, which is the centre of that step's modes; a label is
        # dropped where the agent has barely moved from its current position, or from where
        # the last label went, rather than stacking several on one point.
        placed = [a['pos']]
        for h in sorted({*steps[::4][1:], steps[-1]}):
            if min(np.linalg.norm(a['mean'][h] - p) for p in placed) < 0.09 * view:
                continue
            placed.append(a['mean'][h])
            ax.annotate(f'{(h + 1) * dtv:.1f}s', a['mean'][h], textcoords='offset points',
                        xytext=(6, -12), fontsize=8.5, color=MUTED, zorder=900,
                        path_effects=[pe.Stroke(linewidth=2.2, foreground='w'), pe.Normal()])

    _add_mode_colorbars(fig, agents, cmaps, norms)
    t = frame['t']
    ax.set_title(f't = {t}  ·  {t * dtv:.1f} s' + (f'\n{subtitle}' if subtitle else ''),
                 fontsize=13.5, color=INK)
    # below the panel, not on it: at this zoom an agent can be anywhere in the frame, and a
    # legend box parked in a corner covers whichever one is there
    leg = ax.legend(handles=_legend_handles(pi_threshold, n_stds,
                                            upto=None if len(steps) == ph else len(steps)),
                    loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=2, fontsize=9.5,
                    frameon=False)
    leg.set_zorder(1000)
    return fig


def draw_propagation(frame, agents, hues, meta, center, view, out_path,
                     pi_threshold=0.01, n_stds=(1.0,), map_bg=(None, None)):
    """One frozen timestep: every mode, at every horizon step, over the intersection.

    Deliberately spare. The only trajectory-shaped things on the page are the agent's history,
    its logged future and one track per latent mode; everything else -- samples, most-likely
    path, mixture mean, the moment-matched ellipse -- is left off so that what remains is what
    the model actually emits: 25 weighted Gaussians per horizon step. The numbers behind those
    omissions, the within/between variance split included, are all in the diagnostics figure.
    """
    ph = meta.get('ph', 10)
    fig = _propagation_panel(frame, agents, hues, meta, center, view, range(ph),
                             '',
                             pi_threshold, n_stds, map_bg)
    fig.savefig(out_path, dpi=120, facecolor='white')
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 1b: the same panel, unrolled along the horizon                        #
# --------------------------------------------------------------------------- #
def draw_propagation_animation(frame, agents, hues, meta, center, view, out_base,
                               pi_threshold=0.01, n_stds=(1.0,), map_bg=(None, None),
                               fps=2.0, fmt='gif', hold=3, keep_frames=False):
    """The frozen-timestep panel with the horizon as the animated axis: one frame per
    horizon step, written as a GIF and/or MP4 next to the still.

    The still panel above shows the whole horizon at once, which is what makes it a
    *distribution* -- but it also puts ten steps of contours on the page in one go, and where
    modes separate slowly that overlap hides which contour came from which step. Here the
    cloud is built up a step at a time, so the modes are seen splitting apart rather than
    inferred from a finished overlay. Simulation time is still frozen: nothing in the scene
    moves, only how far ahead the model is looking.

    Steps already drawn are left exactly as they were -- no fading, no de-emphasis. A mode's
    colour is its weight and nothing else, so it is the same colour in frame 1 and frame 10,
    and the last frame therefore *is* the still panel above.
    """
    dtv, ph = meta.get('dt', 0.5), meta.get('ph', 10)
    frame_dir = f'{out_base}_frames'
    os.makedirs(frame_dir, exist_ok=True)

    paths = []
    for h in range(ph):
        fig = _propagation_panel(frame, agents, hues, meta, center, view, range(h + 1),
                                 f'step {h + 1}/{ph}  ·  {(h + 1) * dtv:.1f} s ahead',
                                 pi_threshold, n_stds, map_bg, gt_steps=h + 1)
        out = os.path.join(frame_dir, f'step_{h + 1:02d}.png')
        fig.savefig(out, dpi=100, facecolor='white')
        plt.close(fig)
        paths.append(out)

    # the end of the horizon is the frame worth looking at, and it is also the one the loop
    # throws away first: hold it for a few frames before restarting.
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
# Figure 2: the same distribution as numbers                                   #
# --------------------------------------------------------------------------- #
def draw_diagnostics(frame, agents, hues, meta, out_path):
    """Per-agent rows: variance split, heading-frame spread, analytic vs sampled, truth vs
    spread, weight spectrum."""
    dtv, ph = meta.get('dt', 0.5), meta.get('ph', 10)
    hz = np.arange(1, ph + 1) * dtv
    n = len(agents)
    ncol = 5
    fig, axes = plt.subplots(n, ncol, figsize=(4.7 * ncol, 4.1 * n), squeeze=False)

    for row, (hue, a) in enumerate(zip(hues, agents)):
        ax = axes[row][0]
        ax.plot(hz, a['sigma'], '-', color=hue, lw=2.2, label='total')
        ax.plot(hz, a['sigma_within'], '--', color=hue, lw=1.6, alpha=0.85,
                label='within modes')
        ax.plot(hz, a['sigma_between'], ':', color=hue, lw=2.0, alpha=0.95,
                label='between modes')
        ax.set_ylabel('√tr(C) [m]')
        ax.set_title(f"{a['id']}  ·  variance split", fontsize=11, color=INK)
        ax.legend(fontsize=9, frameon=False)

        ax = axes[row][1]
        ax.plot(hz, a['sigma_long'], '-', color=hue, lw=2.2, label='longitudinal')
        ax.plot(hz, a['sigma_lat'], '--', color=hue, lw=1.8, label='lateral')
        ax.set_ylabel('σ [m]')
        ax.set_title('heading frame', fontsize=11, color=INK)
        ax.legend(fontsize=9, frameon=False)

        # The ellipses vs the fan: same decoder, EKF-linearized covariance against the exact
        # nonlinear rollout of sampled controls. See investigations/gmm_stats.
        ax = axes[row][2]
        ax.plot(hz, a['sigma'], '-', color=hue, lw=2.2, label='analytic (EKF)')
        ax.plot(hz, a['sigma_samples'], '--', color=INK, lw=1.8,
                label=f"sampled ({len(a['samples'])})")
        ax.plot(hz, a['mean_gap'], ':', color=MUTED, lw=1.8, label='mean gap')
        ax.set_ylabel('radius / gap [m]')
        ratio = a['sigma'][-1] / max(a['sigma_samples'][-1], 1e-9)
        ax.set_title(f'analytic vs sampled  ·  ×{ratio:.2f} at end', fontsize=11, color=INK)
        ax.legend(fontsize=9, frameon=False)

        ax = axes[row][3]
        ax.fill_between(hz, 0, 2 * a['sigma'], color=hue, alpha=0.12, lw=0,
                        label='2× RMS')
        ax.fill_between(hz, 0, a['sigma'], color=hue, alpha=0.22, lw=0,
                        label='1× RMS')
        ax.plot(hz, a['gt_error'], '-', color=INK, lw=2.0, label='truth error')
        ax.set_ylabel('displacement [m]')
        finite = np.isfinite(a['gt_mahalanobis'])
        # Mahalanobis, not the band: the band is the isotropic RMS radius, while the truth can
        # miss along the narrow axis of a very elongated covariance.
        tail = f"{a['gt_mahalanobis'][finite][-1]:.1f}σ" if finite.any() else 'n/a'
        ax.set_title(f'truth vs spread  ·  {tail} at end', fontsize=11, color=INK)
        ax.legend(fontsize=9, frameon=False)

        ax = axes[row][4]
        pis = np.sort(a['pis'][0])[::-1]
        ax.bar(np.arange(1, len(pis) + 1), pis, color=hue, width=0.72)
        ax.set_ylabel('π')
        ax.set_xlabel('mode (sorted)')
        ax.set_title(f"latent posterior  ·  {a['perplexity'][0]:.1f}/{len(pis)} modes",
                     fontsize=11, color=INK)

        for col in range(4):
            axes[row][col].set_xlabel('s ahead')
        for col in range(ncol):
            _style_axes(axes[row][col])
            axes[row][col].set_ylim(bottom=0)

    t = frame['t']
    fig.suptitle(f't = {t}  ·  {t * dtv:.1f} s  ·  mixture composition', fontsize=14,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110, facecolor='white')
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 3: across the frozen timesteps                                        #
# --------------------------------------------------------------------------- #
def draw_growth(per_agent, hues, meta, out_path):
    """For each agent, one curve per frozen timestep: total spread, and how much of its
    variance is disagreement between modes rather than blur within them."""
    dtv = meta.get('dt', 0.5)
    n = len(per_agent)
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 3.9 * n), squeeze=False)

    for row, ((aid, series), hue) in enumerate(zip(per_agent.items(), hues)):
        ramp = _hue_ramp(hue, len(series), lightest=0.62)
        for (t, a), c in zip(series, ramp):
            hz = np.arange(1, len(a['sigma']) + 1) * dtv
            axes[row][0].plot(hz, a['sigma'], '-', color=c, lw=1.9,
                              label=f't={t}')
            frac = a['sigma_between'] ** 2 / np.clip(a['sigma'] ** 2, 1e-12, None)
            axes[row][1].plot(hz, 100 * frac, '-', color=c, lw=1.9)
        axes[row][0].set_ylabel('RMS radius [m]')
        axes[row][0].set_title(f'{aid}  ·  spread per timestep', fontsize=11, color=INK)
        axes[row][0].legend(fontsize=8.5, frameon=False, ncol=2)
        axes[row][1].set_ylabel('between-mode share [%]')
        axes[row][1].set_title(f'{aid}  ·  between vs within', fontsize=11, color=INK)
        axes[row][1].set_ylim(0, 100)
        for col in range(2):
            axes[row][col].set_xlabel('s ahead')
            _style_axes(axes[row][col])
        axes[row][0].set_ylim(bottom=0)

    fig.suptitle('spread vs simulation time', fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, facecolor='white')
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
SUMMARY_FIELDS = ['t', 't_sec', 'id', 'speed_mps', 'eff_modes',
                  'sigma_h1', 'sigma_mid', 'sigma_end', 'sigma_within_end',
                  'sigma_between_end', 'between_share_end', 'sigma_long_end',
                  'sigma_lat_end', 'sigma_sampled_end', 'analytic_over_sampled_end',
                  'mean_gap_end', 'gt_err_end', 'gt_mahalanobis_end']


def summary_row(t, dtv, a):
    ph = len(a['sigma'])
    mid, end = ph // 2 - 1, ph - 1
    return {'t': t, 't_sec': round(t * dtv, 2), 'id': a['id'],
            'speed_mps': round(a['speed'], 2),
            'eff_modes': round(float(a['perplexity'][0]), 2),
            'sigma_h1': round(float(a['sigma'][0]), 3),
            'sigma_mid': round(float(a['sigma'][mid]), 3),
            'sigma_end': round(float(a['sigma'][end]), 3),
            'sigma_within_end': round(float(a['sigma_within'][end]), 3),
            'sigma_between_end': round(float(a['sigma_between'][end]), 3),
            'between_share_end': round(float(a['sigma_between'][end] ** 2
                                             / max(a['sigma'][end] ** 2, 1e-12)), 3),
            'sigma_long_end': round(float(a['sigma_long'][end]), 3),
            'sigma_lat_end': round(float(a['sigma_lat'][end]), 3),
            'sigma_sampled_end': round(float(a['sigma_samples'][end]), 3),
            'analytic_over_sampled_end': round(float(a['sigma'][end]
                                                     / max(a['sigma_samples'][end], 1e-9)), 3),
            'mean_gap_end': round(float(a['mean_gap'][end]), 3),
            'gt_err_end': round(float(a['gt_error'][end]), 3),
            'gt_mahalanobis_end': round(float(a['gt_mahalanobis'][end]), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', default='unity_out/intersection_probe/predictions_online.pkl',
                   help='prediction bundle written by a --style gaussian/both run')
    p.add_argument('--out_dir', default=None,
                   help='output directory (default: <bundle dir>/gaussian_propagation)')
    p.add_argument('--times', default='40,44,48,52,56,60',
                   help='comma-separated timesteps to freeze')
    p.add_argument('--agents', default=None,
                   help='which agents to draw, comma-separated. Each entry is a position in '
                        'the roster this run prints (2), a short name / unique tail of the id '
                        '(740), or the full id (-24740). Default = the nearest --max_agents '
                        'at the junction')
    p.add_argument('--center', default=','.join(str(c) for c in DEFAULT_CENTER),
                   help='intersection centre "x,y" in map coords')
    p.add_argument('--view', type=float, default=45.0, help='half-width of the map panel [m]')
    p.add_argument('--select_radius', type=float, default=25.0,
                   help='how close to the centre an agent must come to be studied [m]')
    p.add_argument('--max_agents', type=int, default=3, help='cap on agents studied')
    p.add_argument('--n_std', default='1',
                   help='ellipse contour(s), in sigmas; comma-separated for several '
                        '(e.g. --n_std 1,2 to match the traj_viz gaussian style)')
    p.add_argument('--pi_threshold', type=float, default=0.01,
                   help='hide latent modes whose weight is below this (0 = draw all 25). '
                        'Purely a drawing cutoff -- the hidden modes still carry their '
                        'weight in every number the study reports')
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
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.bundle), 'gaussian_propagation')
    os.makedirs(out_dir, exist_ok=True)
    # the drivable raster is ~15 MP: build the shading overlay once, not once per panel --
    # with the animation there are now ph frames per frozen timestep, not one.
    map_bg = _build_map_rgba(meta)

    by_t = {f['t']: f for f in frames}
    missing = [t for t in times if t not in by_t]
    if missing:
        raise SystemExit(f'timestep(s) {missing} are not in {args.bundle} '
                         f'(it holds t={frames[0]["t"]}..{frames[-1]["t"]})')

    roster = junction_roster(frames, off, center, args.select_radius)
    if not roster:
        raise SystemExit(f'no agent comes within {args.select_radius:g} m of '
                         f'{tuple(center)} -- widen --select_radius')
    roster_ids = [i for i, _ in roster]
    all_ids = sorted(closest_approach(frames, off, center))
    at_times = set(closest_approach(frames, off, center, set(times)))
    names = short_names(roster_ids)

    if args.agents:
        ids = resolve_agents(args.agents, roster_ids, names, all_ids)
        names.update(short_names(sorted(set(roster_ids) | set(ids))))
    else:
        # nearest first, but only agents the frozen timesteps actually have a prediction for
        ids = [i for i in roster_ids if i in at_times][:args.max_agents]
        if not ids:
            raise SystemExit(f'none of the {len(roster)} agents at the junction is predicted '
                             f'at t in {times} -- pick other --times')
    # Hue follows roster position, not the order the agents were asked for: toggling one
    # agent off with --agents must not recolour the others, or two runs of the same junction
    # cannot be laid side by side.
    hue_of = junction_palette(frames, off, center, args.select_radius)
    print_roster(roster, names, set(ids), at_times)
    outside = [i for i in ids if i not in roster_ids]
    if outside:
        print(f'  also drawing {", ".join(outside)} (outside the '
              f'{args.select_radius:g} m selection radius)')

    rows, per_agent, outputs = [], {i: [] for i in ids}, []
    for t in times:
        frame = by_t[t]
        present = {nd['id']: nd for nd in frame['nodes']}
        agents = [analyse_node(present[i], dtv, off) for i in ids if i in present]
        if not agents:
            print(f'  t={t}: none of the studied agents is predicted here -- skipped')
            continue
        hues = [hue_of[a['id']] for a in agents]
        for a in agents:
            per_agent[a['id']].append((t, a))
            rows.append(summary_row(t, dtv, a))
        outputs.append(draw_propagation(frame, agents, hues, meta, center, args.view,
                                        os.path.join(out_dir, f'propagation_t{t:03d}.png'),
                                        pi_threshold=args.pi_threshold, n_stds=n_stds,
                                        map_bg=map_bg))
        outputs.append(draw_diagnostics(frame, agents, hues, meta,
                                        os.path.join(out_dir, f'diagnostics_t{t:03d}.png')))
        if args.anim != 'none':
            outputs += draw_propagation_animation(
                frame, agents, hues, meta, center, args.view,
                os.path.join(out_dir, f'propagation_t{t:03d}'),
                pi_threshold=args.pi_threshold, n_stds=n_stds, map_bg=map_bg,
                fps=args.anim_fps, fmt=args.anim, hold=args.anim_hold,
                keep_frames=args.keep_frames)
        print(f'  t={t:>3} ({t * dtv:>4.1f}s): ' + '   '.join(
            f"{a['id']} σ(5s)={a['sigma'][-1]:5.1f}m "
            f"between={100 * a['sigma_between'][-1] ** 2 / max(a['sigma'][-1] ** 2, 1e-12):4.0f}% "
            f"modes={a['perplexity'][0]:4.1f}" for a in agents))

    per_agent = {i: s for i, s in per_agent.items() if s}
    if per_agent:
        outputs.append(draw_growth(per_agent, [hue_of[i] for i in per_agent], meta,
                                   os.path.join(out_dir, 'growth_vs_time.png')))

    csv_path = os.path.join(out_dir, 'propagation_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(outputs)} figures/animations and {csv_path}')
    ratios = [r['analytic_over_sampled_end'] for r in rows
              if np.isfinite(r['analytic_over_sampled_end'])]
    gaps = [r['mean_gap_end'] for r in rows if np.isfinite(r['mean_gap_end'])]
    if ratios:
        print(f'  end-of-horizon spread, analytic / sampled: x{min(ratios):.2f}..x{max(ratios):.2f}'
              f'   mean offset up to {max(gaps):.1f} m'
              f'  (EKF linearization vs exact rollout -- see investigations/gmm_stats)')


if __name__ == '__main__':
    main()
