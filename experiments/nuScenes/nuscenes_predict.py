"""
Run Trajectron++ on the *nuScenes* processed data (the same Environment pickle the
qualitative notebook uses) and store the predicted trajectory distribution to disk,
reusing the prediction engine from unity_predict.py and the renderer from traj_viz.py.

Unlike the Unity path, nuScenes scenes already ship as a built Environment/Scene graph
(with real HD-map context) -- so there is nothing to build: load the pickle, pick a
scene, run the model, save a prediction bundle, and (optionally) render it. `--ego_frame`
uses the scene's real `ego` robot node as the origin.

Prediction and visualization are separable (same as unity_predict.py):
    python nuscenes_predict.py --scene 103 --gpu 0            # predict + render
    python nuscenes_predict.py --scene 103 --gpu 0 --no_viz   # predict only -> predictions.pkl
    python nuscenes_predict.py --scene 103 --viz_only         # render stored preds (no model)
    python nuscenes_predict.py --list                         # list scenes in the pickle
Or render a stored bundle with the standalone visualizer: python visualize.py --pred_file ...
"""
import os
import sys
import argparse

import numpy as np
import dill

sys.path.append('../../trajectron')
sys.path.append('../unity_lidar_sim')
from helper import load_model
from unity_predict import resolve_device, run_predictions
from traj_viz import save_bundle, load_bundle, render_bundle


def robot_pose_at(scene, t):
    """(pos, yaw) of the scene's ego/robot node at timestep t, in scene-local coords;
    None if the robot isn't present at t."""
    r = scene.robot
    if r is None or not (r.first_timestep <= t <= r.last_timestep):
        return None
    pos = r.get(np.array([t]), {'position': ['x', 'y']})[0]
    yaw = r.get(np.array([t]), {'heading': ['°']})[0, 0]
    return pos, yaw


def scene_extent(scene, pad=15.0):
    xs, ys = [], []
    for n in scene.nodes:
        p = n.data.position
        xs += [np.nanmin(p.x), np.nanmax(p.x)]
        ys += [np.nanmin(p.y), np.nanmax(p.y)]
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_pkl', default='../processed/nuScenes_test_mini_full.pkl')
    p.add_argument('--scene', default=None, help='scene name or index (default: first scene)')
    p.add_argument('--list', action='store_true', help='list scenes in the pickle and exit')
    p.add_argument('--model_dir', default='./models/int_ee_me')
    p.add_argument('--model_ts', type=int, default=12)
    p.add_argument('--ph', type=int, default=6)
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--frame_stride', type=int, default=1)
    p.add_argument('--single_t', type=int, default=None)
    p.add_argument('--out_dir', default='nuscenes_out')
    p.add_argument('--ego_frame', action='store_true',
                   help='render in the ego robot node coordinate frame (origin=ego, heading up)')
    p.add_argument('--zoom', type=float, default=60.0, help='ego-frame half-window in metres')
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'], help='output video format(s)')
    p.add_argument('--style', default='samples', choices=['samples', 'gaussian', 'both'],
                   help='distribution render style: sample fan, Gaussian blobs, or both')
    p.add_argument('--gpu', type=int, default=-1, help='CUDA device index to run on (e.g. --gpu 2); -1 = CPU')
    p.add_argument('--pred_file', default=None,
                   help='where to store/load the prediction bundle (default: <out_dir>/predictions.pkl)')
    p.add_argument('--no_viz', action='store_true',
                   help='run the model and save predictions only; skip rendering')
    p.add_argument('--viz_only', action='store_true',
                   help='skip the model; render from an existing --pred_file')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_file = args.pred_file or os.path.join(args.out_dir, 'predictions.pkl')

    # ------- visualization-only: no model, no pickle, just stored predictions -------
    if args.viz_only:
        print(f'Loading predictions from {pred_file}...')
        bundle = load_bundle(pred_file)
        render_bundle(bundle, args.out_dir, ego_frame=args.ego_frame, fps=args.fps,
                      zoom=args.zoom, single=args.single_t is not None, fmt=args.format, style=args.style)
        return

    print(f'Loading {args.data_pkl}...')
    env = dill.load(open(args.data_pkl, 'rb'), encoding='latin1')

    if args.list:
        for i, s in enumerate(env.scenes):
            print(f'  [{i}] name={s.name}  timesteps={s.timesteps}  nodes={len(s.nodes)}  '
                  f'robot={None if s.robot is None else s.robot.id}')
        return

    # select scene by name, else index, else first
    scene = env.scenes[0]
    if args.scene is not None:
        by_name = [s for s in env.scenes if str(s.name) == str(args.scene)]
        if by_name:
            scene = by_name[0]
        elif args.scene.isdigit() and int(args.scene) < len(env.scenes):
            scene = env.scenes[int(args.scene)]
        else:
            raise SystemExit(f'scene {args.scene!r} not found; use --list')
    print(f'Scene {scene.name}: {scene.timesteps} timesteps, {len(scene.nodes)} nodes')

    xlim, ylim = scene_extent(scene)

    device = resolve_device(args.gpu)
    print(f'Loading model from {args.model_dir} (ts={args.model_ts}) on {device}...')
    eval_stg, hyp = load_model(args.model_dir, env, ts=args.model_ts, device=device)

    # ego pose from the scene's robot node, stored so the visualizer can toggle ego view
    ego_for = lambda t: robot_pose_at(scene, t)

    timesteps = [args.single_t] if args.single_t is not None \
        else list(range(2, scene.timesteps - 1, args.frame_stride))
    frames = run_predictions(eval_stg, scene, timesteps, args.ph, args.num_samples, ego_fn=ego_for)

    meta = {'source': 'nuscenes', 'scene': scene.name, 'dt': scene.dt, 'ph': args.ph,
            'num_samples': args.num_samples, 'x_min': 0.0, 'y_min': 0.0,
            'xlim': xlim, 'ylim': ylim, 'zoom': args.zoom, 'gif_prefix': f'scene{scene.name}'}
    save_bundle(pred_file, meta, frames)
    print(f'Saved {len(frames)} prediction frames -> {pred_file}')

    if not args.no_viz:
        render_bundle({'meta': meta, 'frames': frames}, args.out_dir, ego_frame=args.ego_frame,
                      fps=args.fps, zoom=args.zoom, single=args.single_t is not None,
                      fmt=args.format, style=args.style)


if __name__ == '__main__':
    main()
