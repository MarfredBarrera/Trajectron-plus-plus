"""
Run a pre-trained Trajectron++ model on FutureDet / Unity-sim ground-truth agent
tracks and visualize the predicted trajectory *distribution*.

This is the batch (offline) driver: it replays a finished log through
`Trajectron.predict()`, which re-reads and re-encodes each agent's full history window
at every timestep. For the streaming counterpart -- one LSTM state per agent advanced by
a single observation per timestep, with risk scored as the run proceeds -- see
unity_online.py. Scene ingest, map loading and Environment construction are shared
between the two and live in scene_conditioning/unity_scene.py.

Scene, model, and render settings live in a YAML config (default configs/config.yaml,
see that file); the command line only takes the knobs that actually change
run-to-run.

Usage (from experiments/unity_lidar_sim/):
    python src/unity_predict.py
    python src/unity_predict.py --config configs/sweep_s1_rail.yaml --gpu 0
    python src/unity_predict.py --style gaussian --format mp4
See `python src/unity_predict.py -h`. To re-render a stored bundle without
re-running the model, use src/visualization/visualize.py instead.
"""
import os
import argparse

import numpy as np
import torch

import repo_paths  # noqa: F401  (sys.path side effect: trajectron/, experiments/nuScenes/)
from helper import load_model
from utils import prediction_output_to_trajectories
from bundle import save_bundle
from visualization.traj_viz import render_bundle
from scene_conditioning.unity_scene import load_config, prepare_scene, resolve_device


# --------------------------------------------------------------------------- #
# Prediction engine (model -> serializable per-timestep arrays)                #
# --------------------------------------------------------------------------- #
def predict_frame(eval_stg, scene, t, ph, num_samples, min_history_timesteps=1,
                  need_samples=True):
    """Run the predict() calls at one timestep and extract a serializable record:
    per node, history / GT future / most-likely path / analytic GMM params (all in
    scene-local coords). Returns None if no node is predictable at t.

    `need_samples`: when False (gaussian-only viz) the expensive Monte Carlo sampling
    pass is skipped and each node's `samples` is stored empty -- history/future are
    sourced from the most-likely pass instead, which carries the same GT arrays."""
    with torch.no_grad():
        preds = None
        if need_samples:
            preds = eval_stg.predict(scene, np.array([t]), ph, num_samples=num_samples,
                                     min_history_timesteps=min_history_timesteps,
                                     z_mode=False, gmm_mode=False, full_dist=True)
        preds_mm = eval_stg.predict(scene, np.array([t]), ph, num_samples=1,
                                    min_history_timesteps=min_history_timesteps,
                                    z_mode=True, gmm_mode=True)
        # Deterministic per-latent-mode GMM (mean/cov propagated analytically through the
        # dynamics model), for the Gaussian-blob viz -- no sample statistics involved.
        _, dists_d = eval_stg.predict(scene, np.array([t]), ph, num_samples=1,
                                      min_history_timesteps=min_history_timesteps,
                                      z_mode=False, gmm_mode=True, full_dist=True,
                                      output_dists=True)
    if not preds_mm:
        return None
    # history/future are identical across passes; take them from the most-likely one so
    # this works whether or not the sampling pass ran.
    predmm_d, hist_d, fut_d = prediction_output_to_trajectories(preds_mm, scene.dt, 10, ph, map=None)
    pred_d = None
    if preds:
        pred_d, _, _ = prediction_output_to_trajectories(preds, scene.dt, 10, ph, map=None)
    tk = list(predmm_d.keys())[0]
    dtk = list(dists_d.keys())[0]
    nodes = []
    for node in sorted(predmm_d[tk].keys(), key=lambda n: n.id):
        dist = dists_d[dtk][node]
        samples = (np.asarray(pred_d[tk][node][0], dtype=np.float32) if pred_d is not None
                   else np.empty((0, ph, 2), dtype=np.float32))
        nodes.append({
            'id': node.id, 'type': node.type.name,
            'history': np.asarray(hist_d[tk][node], dtype=np.float32),
            'future': np.asarray(fut_d[tk][node], dtype=np.float32),
            'samples': samples,                                             # (S, ph, 2)
            'ml': np.asarray(predmm_d[tk][node][0, 0], dtype=np.float32),    # (ph, 2)
            'dist_mus': np.asarray(dist['mus'][0, 0], dtype=np.float32),    # (ph, K, 2)
            'dist_covs': np.asarray(dist['covs'][0, 0], dtype=np.float32), # (ph, K, 2, 2)
            'dist_pis': np.asarray(dist['pis'][0, 0], dtype=np.float32),   # (ph, K)
        })
    return {'t': int(t), 'nodes': nodes}


def run_predictions(eval_stg, scene, timesteps, ph, num_samples,
                    min_history_timesteps=1, ego_fn=None, need_samples=True, ego_path_fn=None):
    """Predict over a list of timesteps -> list of frame records. `ego_fn(t)` (optional)
    returns (pos, yaw) of the ego at t (stored so the visualizer can toggle ego view).
    `ego_path_fn(t)` (optional) returns the ego's actual future path [[x,y], ...] over the
    horizon in world coords (for the trajectory-crossing risk metric).
    `need_samples` is threaded to predict_frame to gate the Monte Carlo sampling pass."""
    frames = []
    for i, t in enumerate(timesteps):
        rec = predict_frame(eval_stg, scene, t, ph, num_samples, min_history_timesteps,
                            need_samples=need_samples)
        if rec is not None:
            ego = None if ego_fn is None else ego_fn(t)
            rec['ego'] = None if ego is None else [np.asarray(ego[0], dtype=float).tolist(),
                                                   float(ego[1])]
            rec['ego_path'] = None if ego_path_fn is None else ego_path_fn(t)
            frames.append(rec)
        print(f'  predict [{i + 1}/{len(timesteps)}] t={t}', end='\r', flush=True)
    print()
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/config.yaml',
                   help='YAML config with scene/model/render settings (see configs/config.yaml)')
    p.add_argument('--gpu', type=int, default=-1, help='CUDA device index to run on (e.g. --gpu 2); -1 = CPU')
    p.add_argument('--style', default='samples', choices=['samples', 'gaussian', 'both'],
                   help='distribution render style: sample fan, Gaussian blobs, or both')
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'], help='output video format(s)')
    args = p.parse_args()
    cfg = load_config(args.config)

    scene_data = prepare_scene(cfg)
    out_dir = scene_data.out_dir
    pred_file = cfg['pred_file'] or os.path.join(out_dir, 'predictions.pkl')
    scene = scene_data.scene

    device = resolve_device(args.gpu)
    print(f'Loading model from {cfg["model_dir"]} (ts={cfg["model_ts"]}) on {device}...')
    eval_stg, hyp = load_model(cfg['model_dir'], scene_data.env, ts=cfg['model_ts'], device=device)
    # The checkpoint and the scene have to agree about the ego -- see the same check in
    # online_engine.OnlineEngine. With them in agreement, ego-plan conditioning needs nothing
    # further here: `get_timesteps_data` reads scene.robot and drops it from the predicted set.
    if hyp.get('incl_robot_node') != (scene_data.robot is not None):
        raise SystemExit(f'ego_conditioning is {cfg["ego_conditioning"]} but '
                         f'{cfg["model_dir"]}/config.json has incl_robot_node: '
                         f'{hyp.get("incl_robot_node")} -- the two must match')

    timesteps = [cfg['single_t']] if cfg['single_t'] is not None \
        else list(range(2, scene_data.n_timesteps - 1, cfg['frame_stride']))
    # 'gaussian' renders only the analytic GMM, so skip the expensive sampling pass.
    need_samples = args.style in ('samples', 'both')
    frames = run_predictions(eval_stg, scene, timesteps, cfg['ph'], cfg['num_samples'],
                             ego_fn=scene_data.ego_pose, need_samples=need_samples,
                             ego_path_fn=lambda t: scene_data.ego_logged_path(t, cfg['ph']))

    meta = scene_data.bundle_meta()
    save_bundle(pred_file, meta, frames)
    print(f'Saved {len(frames)} prediction frames -> {pred_file}')

    # ------- optional rendering from the just-computed bundle -------
    if not cfg['no_viz']:
        render_bundle({'meta': meta, 'frames': frames}, out_dir, ego_frame=cfg['ego_frame'],
                      fps=cfg['fps'], zoom=cfg['zoom'], single=cfg['single_t'] is not None,
                      fmt=args.format, style=args.style, workers=cfg['workers'])


if __name__ == '__main__':
    main()
