"""Entering trajectories over time, split by agent: the scene-level view of proximity risk.

The per-agent CSV answers "which agent is risky at t"; this answers "how much of the
predicted distribution is heading into the ego's disc at all, right now", and which agents
that total is made of. For each timestep it sums `n_enter` over every predicted agent, so the
series counts sampled trajectories, not agents:

    entering(t) = sum over agents of (# samples reaching the ego's disc at t)

The figure stacks the per-agent counts, so the bands are the individual contributions and
their top edge -- drawn as a line -- is exactly that cumulative total. One chart answers both
questions, and the stack cannot disagree with the total the way two panels could.

The constant denominator (agents x samples-per-agent) sets the y ceiling, so the height of
the curve can be read as a fraction of the whole predicted distribution without a second
axis.

Kept apart from render.py because that module is the per-frame overlay video; this is one
static summary figure over the whole run. Agent colours are not this module's to invent: the
caller passes the identity map (`visualization.colors`), so a band is the same colour as that
agent's samples in the video -- or, with `--junction_colors`, the same colour as that agent
in the intersection investigation's figures.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

from visualization.colors import color_for


# Chart tokens. Series hues come from the agent palette (visualization.colors) so identity is
# shared with the overlay video; text never wears a series colour. Verified against the
# #fcfcfb surface: muted ink 7.73:1 and ink 12.6:1 (both need 4.5:1 as text), and the palette
# fills carry a surface-coloured separator rather than relying on hue contrast alone. Light
# surface only -- these figures are written to disk and read alongside the other pipeline
# PNGs, all of which commit to a white background.
SURFACE = '#fcfcfb'
INK = '#2b2a26'           # the total, which must stay legible over any band beneath it
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


def aggregate_by_agent(rows):
    """Per-agent entering counts on the common timestep axis.

    Agents come and go over a run, so each series is zero-filled at the timesteps where its
    agent was absent -- which is also what it means for the stack: an agent that is not
    predicted contributes nothing. The series therefore sum to `aggregate`'s totals exactly.

    Returns (t, {agent_id: (T,) counts}) with the timesteps sorted.
    """
    ts = sorted({int(r['t']) for r in rows})
    at = {t: i for i, t in enumerate(ts)}
    per = {}
    for r in rows:
        s = per.get(r['agent'])
        if s is None:
            s = per[r['agent']] = np.zeros(len(ts), dtype=int)
        s[at[int(r['t'])]] += int(r['n_enter'])
    return np.array(ts), per


def plot_entry_counts(rows, out_path, radius, dt=0.5, agent_colors=None):
    """Per-agent stack + cumulative total of entering trajectories -> `out_path` (PNG).

    Deliberately bare: the only prose is the two axis labels and the agent ids in the legend,
    so the figure drops into a document that supplies its own caption. The numbers behind it
    go to CSV alongside (`write_counts_csv`), which is where any per-timestep detail belongs.

    :param agent_colors: {agent id: colour} -- which identity scheme the bands are drawn
        under, from `visualization.colors` (`palette_for` to match the overlay video,
        `junction_palette` to match the intersection investigation). Falls back to the
        renderer's palette in the order agents appear in `rows`.
    :return: the path, or None if `rows` is empty.
    """
    if not rows:
        print('No risk rows to plot.')
        return None
    ts, enter, total, _ = aggregate(rows)
    _, per_agent = aggregate_by_agent(rows)
    x = ts * dt                                        # timestep index -> seconds

    # Agents that never enter the disc would be invisible bands with a legend entry each, so
    # they are dropped: the legend then lists exactly the agents the figure shows. The rest
    # stack biggest-first, which keeps the busiest band against the flat baseline.
    if agent_colors is None:
        order = {}
        agent_colors = {r['agent']: color_for(r['agent'], order) for r in rows}
    agents = sorted((a for a, s in per_agent.items() if s.any()),
                    key=lambda a: -int(per_agent[a].sum()))

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

    if agents:
        # Surface-coloured separators keep adjacent bands apart where two agents carry
        # similar hues; the fills are dropped in opacity so the grid still reads through.
        ax.stackplot(x, [per_agent[a] for a in agents], labels=[str(a) for a in agents],
                     colors=[agent_colors[a] for a in agents],
                     alpha=0.72, edgecolor=SURFACE, lw=0.6, zorder=4)
    # The stack's top edge is the total by construction, so drawing it costs nothing and
    # gives the scene-level series a hard line to be read off.
    ax.plot(x, enter, color=INK, lw=1.6, solid_capstyle='round', label='total', zorder=6)

    if agents:
        handles, labels = ax.get_legend_handles_labels()
        # Total first: it is the series the other entries decompose.
        ordered = [(h, l) for h, l in zip(handles, labels) if l == 'total'] + \
                  [(h, l) for h, l in zip(handles, labels) if l != 'total']
        # Above the axes rather than inside them: a busy run fills the upper left, and a
        # legend that has to dodge the data is a legend that sometimes lands on it.
        ax.legend([h for h, _ in ordered], [l for _, l in ordered],
                  loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=min(len(ordered), 7),
                  fontsize=9, frameon=False, handlelength=1.4, handleheight=0.9,
                  columnspacing=1.4, borderpad=0.0, labelcolor=INK_MUTED)

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
