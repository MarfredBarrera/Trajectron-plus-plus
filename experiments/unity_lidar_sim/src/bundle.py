"""The prediction bundle: the on-disk hand-off between prediction, evaluation and rendering.

`unity_online.py` predicts and scores risk in the same online pass and writes both here.
Everything downstream -- the risk CLI and the visualizers -- reads this file, so figures are
built from the recorded run rather than from anything recomputed afterwards.

Bundle format (pickle):
    {'meta':  {source, scene, dt, ph, num_samples, x_min, y_min, xlim, ylim,
               zoom, gif_prefix, risk_radius},
     'frames': [ {'t': int,
                  'ego': [[ex, ey], yaw] or None,      # world coords; enables ego view
                  'ego_path': [[x, y], ...] or None,   # context only; the metric ignores it
                  'nodes': [ {'id', 'type',
                              'history' (H,2), 'future' (F,2),
                              'samples' (S,ph,2), 'ml' (ph,2),
                              'dist_mus' (ph,K,2), 'dist_covs' (ph,K,2,2), 'dist_pis' (ph,K),
                              # risk, as scored online at this timestep (risk_eval.proximity):
                              'risk_enter' (S,) bool, 'risk_dist_now', 'risk_min_dist'},
                             ... ]}, ... ]}

The `risk_*` fields are what makes the visualizers pure consumers: `risk_enter` says which
samples entered the ego's disc, which the per-agent CSV cannot express. A bundle written
before online scoring simply lacks them, and the risk CLI falls back to rescoring.

All node arrays are in SCENE-LOCAL coords; meta['x_min'/'y_min'] shift them to world.
Whether a frame is drawn in world or ego view is a *render-time* choice -- the ego pose is
always stored, so the visualizer can toggle it without re-running the model. A decoder pass
that was switched off (see online_engine's DECODE_* toggles) leaves its arrays zero-length.

This module deliberately depends on nothing but pickle: writing a bundle at the end of a run
should not pull in matplotlib, and neither should reading one.
"""
import pickle


def save_bundle(path, meta, frames):
    with open(path, 'wb') as f:
        pickle.dump({'meta': meta, 'frames': frames}, f)


def load_bundle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
