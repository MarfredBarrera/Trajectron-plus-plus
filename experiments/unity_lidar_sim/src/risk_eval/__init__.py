"""Evaluating a prediction bundle for conflict with the ego.

    crossing   the crossing-risk metric: fraction of an agent's sampled trajectories that
               cross the ego's path at a coincident time. numpy-only; also used live by
               the online driver, which scores each timestep as it predicts it.
    render     the crossing-risk overlay video (matplotlib).
    cli        entry point: score (and optionally render) a stored bundle.

Import the submodule you need rather than the package: pulling the metric in should not
cost you matplotlib.
"""
