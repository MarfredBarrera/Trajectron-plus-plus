"""One-off investigations: focused studies of model behaviour, not part of the pipeline.

Each module here answers a specific question about what the predictions are doing. Most read
a stored bundle (src/bundle.py) rather than running the model, so a study can be re-run,
re-cropped and re-plotted without touching a GPU.

    gmm_stats             -- statistics of the decoder's Gaussian mixture (no matplotlib)
    gaussian_propagation  -- freeze a timestep, draw how the mixture propagates over the
                             horizon, repeat over timesteps
    action_propagation    -- the same mixture in control space, before the dynamics
    flicker               -- how much the predicted reach pulsates frame to frame, measured
                             identically on a Unity bundle and on nuScenes
    synthetic_input       -- the same statistic on a hand-built constant-velocity agent, to
                             separate input-noise amplification from encoder instability

The last two run the model (there is no stored bundle for nuScenes, and none at all for a
synthetic scene), so they need the checkpoint and a GPU; `flicker --skip_nuscenes` does not.
"""
