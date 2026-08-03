"""Total entering trajectories over time: the scene-level view of proximity risk.

The per-agent CSV answers "which agent is risky at t"; this answers "how much of the
predicted distribution is heading into the ego's disc at all, right now". For each
timestep it sums `n_enter` over every predicted agent, so the series counts sampled
trajectories, not agents:

    entering(t) = sum over agents of (# samples reaching the ego's disc at t)

The constant denominator (agents x samples-per-agent) is drawn as a reference line, so the
height of the curve can be read as a fraction of the whole predicted distribution without a
second axis.

Kept apart from render.py because that module is the per-frame overlay video; this is one
static summary figure over the whole run.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator


# Chart tokens. One series, so one categorical hue; text never wears the series colour.
# Verified against the #fcfcfb surface: series 4.30:1 (needs 3:1 as a graphical mark),
# muted ink 7.73:1 (needs 4.5:1 as text). Light surface only -- these figures are written to
# disk and read alongside the other pipeline PNGs, all of which commit to a white background.
SURFACE = '#fcfcfb'
SERIES = '#2a78d6'
INK_MUTED = '#52514e'
GRID = '#d8d7d2'          # major rules, at the labelled ticks
GRID_MINOR = '#ebeae6'    # minor rules, one per timestep -- must stay under the data


def aggregate(rows):
    """Per-timestep totals from the per-agent risk rows.
    Returns (t, entering, total, n_agents) as parallel arrays sorted by timestep."""
    by_t = {}
    for r in rows:
        t = int(r['t'])
        e, n, k = by_t.get(t, (0, 0, 0))
        by_t[t] = (e + int(r['n_enter']), n + int(r['n_samples']), k + 1)
    ts = sorted(by_t)
    return (np.array(ts),
            np.array([by_t[t][0] for t in ts]),
            np.array([by_t[t][1] for t in ts]),
            np.array([by_t[t][2] for t in ts]))


def plot_entry_counts(rows, out_path, radius, dt=0.5):
    """Line+area of the total entering trajectories per timestep -> `out_path` (PNG).

    Deliberately bare: the only prose is the two axis labels, so the figure drops into a
    document that supplies its own caption. The numbers behind it go to CSV alongside
    (`write_counts_csv`), which is where any per-timestep detail belongs.

    Returns the path, or None if `rows` is empty."""
    if not rows:
        print('No risk rows to plot.')
        return None
    ts, enter, total, _ = aggregate(rows)
    x = ts * dt                                        # timestep index -> seconds

    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)

    # The full sample count sets the ceiling, so the curve's height still reads as a share
    # of the whole predicted distribution -- without a reference line to have to explain.
    ax.set_ylim(0, (total.max() if len(total) else 1) * 1.04)

    ax.fill_between(x, enter, color=SERIES, alpha=0.16, lw=0, zorder=4)
    ax.plot(x, enter, color=SERIES, lw=2.0, solid_capstyle='round', zorder=5)

    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel('simulation time [s]', fontsize=10, color=INK_MUTED)
    # The radius rides on the axis label rather than a title, so the figure still says what
    # it counts while all prose stays on the two axes.
    ax.set_ylabel(f'trajectories entering {radius:g} m disc', fontsize=10, color=INK_MUTED)
    # Two-level grid: labelled seconds as major rules, one minor rule per timestep so a
    # single point can be traced back to the timestep that produced it. The minor rules are
    # deliberately near-surface -- 59 of them at major weight would read as a hatch.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 5, 10], integer=True))
    ax.xaxis.set_minor_locator(MultipleLocator(dt))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='major', color=GRID, lw=0.8, zorder=0)
    ax.grid(which='minor', color=GRID_MINOR, lw=0.6, zorder=0)
    ax.tick_params(axis='x', which='major', colors=INK_MUTED, labelsize=9,
                   length=5, width=0.9, color=GRID)
    ax.tick_params(axis='x', which='minor', length=2.5, width=0.7, color=GRID)
    ax.tick_params(axis='y', which='both', colors=INK_MUTED, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(lambda v, _: f'{int(v):,}')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote risk time series -> {out_path}')
    return out_path


def write_counts_csv(rows, path):
    """The plotted series as a table, so the figure is never the only way to read it."""
    import csv
    ts, enter, total, n_ag = aggregate(rows)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'n_agents', 'n_entering', 'n_samples', 'fraction'])
        for i, t in enumerate(ts):
            w.writerow([int(t), int(n_ag[i]), int(enter[i]), int(total[i]),
                        round(float(enter[i]) / total[i], 6) if total[i] else 0.0])
    print(f'Wrote risk time series table -> {path}')
    return path
