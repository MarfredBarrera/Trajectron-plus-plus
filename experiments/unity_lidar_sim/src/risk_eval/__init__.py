"""Evaluating a prediction bundle for conflict with the ego.

    proximity  the ego-proximity risk metric: fraction of an agent's sampled trajectories
               that reach into a disc of fixed radius around the ego's position at that
               timestep. numpy-only; also used live by the online driver, which scores
               each timestep as it predicts it.
    render     the proximity-risk overlay video (matplotlib).
    cli        entry point: score (and optionally render) a stored bundle.

Import the submodule you need rather than the package: pulling the metric in should not
cost you matplotlib.
"""
