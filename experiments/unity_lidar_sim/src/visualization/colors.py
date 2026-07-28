"""Stable per-agent colours.

Split out of traj_viz so the risk metric can label agents consistently with the renderer
without importing matplotlib: `risk_eval.crossing` runs inside the online loop, where a
plotting dependency has no business being.
"""

# distinct colors cycled per agent
AGENT_COLORS = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0', '#E8A33D',
                '#7B68EE', '#2ECC71', '#E74C3C', '#1ABC9C', '#F39C12', '#9B59B6']


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
