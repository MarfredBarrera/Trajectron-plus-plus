"""Stable per-agent colours.

Split out of traj_viz so the risk metric can label agents consistently with the renderer
without importing matplotlib: `risk_eval.proximity` runs inside the online loop, where a
plotting dependency has no business being.

Two identity schemes live here, and both hand back the same thing -- a {agent id: colour}
map, so a figure only has to be told which scheme it is drawing under:

* `palette_for` -- order of appearance in the bundle. The renderer's scheme: no scene
  knowledge needed, so it works for any run.
* `junction_palette` -- the intersection investigation's scheme: the agents that come nearest
  a junction wear `AGENT_HUES` in that order, so agent #1 is navy in every figure of that
  study. Everyone else takes a colour no roster agent is using.

The junction ordering (`closest_approach`, `junction_roster`) is here rather than in
investigations/ so the risk figures can match those hues without importing the study code.
"""
import numpy as np


# distinct colors cycled per agent
AGENT_COLORS = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0', '#E8A33D',
                '#7B68EE', '#2ECC71', '#E74C3C', '#1ABC9C', '#F39C12', '#9B59B6']

# Agent identity in the intersection investigation, by roster position (see
# `junction_roster`). Kept in this order and never recycled: the study's prose names agents by
# their colour ("the amber agent"), so #2 has to stay amber everywhere it is drawn.
AGENT_HUES = ['#375397', '#E8A33D', '#00857A', '#F05F78']

# Intersection where the ego meets the turning and waiting vehicles, in map coordinates.
DEFAULT_CENTER = (157.0, 71.0)
DEFAULT_RADIUS = 25.0        # how close an agent must come to that centre to be on the roster


def color_for(node_id, order):
    """Colour of `node_id`, assigning it the next index in `order` if it is new.

    `order` is mutated, so a streaming caller that has never seen the full agent set still
    gets colours that stay put as agents appear.
    """
    if node_id not in order:
        order[node_id] = len(order)
    return AGENT_COLORS[order[node_id] % len(AGENT_COLORS)]


def precompute_order(frames):
    """Global node_id -> color index by first appearance (frame, then node order), so every
    frame -- including ones rendered in parallel worker processes -- colors agents the same."""
    order = {}
    for rec in frames:
        for nd in rec['nodes']:
            color_for(nd['id'], order)
    return order


def palette_for(frames):
    """{agent id: colour} under the renderer's scheme -- the colours the video draws."""
    order = precompute_order(frames)
    return {i: color_for(i, order) for i in order}


# --------------------------------------------------------------------------- #
# The intersection investigation's scheme                                       #
# --------------------------------------------------------------------------- #
def closest_approach(frames, off, center, times=None):
    """{agent id: how close it gets to `center`}, in metres, over `times` (default: all)."""
    best = {}
    for f in frames:
        if times is not None and f['t'] not in times:
            continue
        for nd in f['nodes']:
            d = float(np.linalg.norm(np.asarray(nd['history'], float)[-1] + off - center))
            best[nd['id']] = min(best.get(nd['id'], np.inf), d)
    return best


def junction_roster(frames, off, center, radius):
    """[(id, distance)] for the agents that come within `radius` of `center`, nearest first.

    Deliberately computed over the whole bundle rather than over a frozen set of timesteps:
    the roster is what the studies' `--agents` counts against, so #2 has to mean the same
    vehicle in every run on this bundle. Scoping it to selected timesteps would renumber the
    agents whenever that selection changed, which is exactly the thing that makes a short
    handle useless.
    """
    best = closest_approach(frames, off, center)
    return [(i, d) for d, i in sorted((d, i) for i, d in best.items() if d <= radius)]


def junction_palette(frames, off, center=DEFAULT_CENTER, radius=DEFAULT_RADIUS):
    """{agent id: colour} under the intersection investigation's scheme.

    Roster agents wear `AGENT_HUES` by position, so a figure drawn through here shows the same
    vehicle in the same colour as gaussian_propagation.py and action_propagation.py do. The
    agents that never reach the junction are not in that study at all, so they take the next
    colour from the general palette that no roster agent is using -- never a hue that would
    read as "the amber agent" somewhere else.

    :param off: (2,) scene-local -> world offset, i.e. meta['x_min'/'y_min'].
    """
    center = np.asarray(center, dtype=float)
    roster = junction_roster(frames, off, center, radius)
    pal = {i: AGENT_HUES[k % len(AGENT_HUES)] for k, (i, _) in enumerate(roster)}
    spare = [c for c in AGENT_COLORS if c not in set(pal.values())] or AGENT_COLORS
    # Distance order again for the rest, so the map depends on the scene and not on the order
    # a caller happened to hand the agents over in.
    rest = sorted(((d, i) for i, d in closest_approach(frames, off, center).items()
                   if i not in pal))
    for k, (_, i) in enumerate(rest):
        pal[i] = spare[k % len(spare)]
    return pal
