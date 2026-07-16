"""
Run a pre-trained Trajectron++ model on FutureDet / Unity-sim ground-truth agent
tracks and visualize the predicted trajectory *distribution*.

Usage (from experiments/unity_lidar_sim/):
    python unity_predict.py --num_samples 200 --frame_stride 2
    python unity_predict.py --single_t 40         # one debug frame
See `python unity_predict.py -h`.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import dill
import torch

sys.path.append('../../trajectron')
sys.path.append('../nuScenes')
from environment import Environment, Scene, Node, GeometricMap, derivative_of
from helper import load_model
from utils import prediction_output_to_trajectories
from traj_viz import save_bundle, load_bundle, render_bundle


def resolve_device(gpu):
    """Turn --gpu into a torch device string, with safe fallback to cpu.
    gpu < 0 -> cpu; gpu >= 0 -> cuda:<gpu> (falls back to cpu/cuda:0 if unavailable)."""
    if gpu is None or gpu < 0:
        return 'cpu'
    if not torch.cuda.is_available():
        print(f'WARNING: --gpu {gpu} requested but CUDA is unavailable; using cpu')
        return 'cpu'
    n = torch.cuda.device_count()
    if gpu >= n:
        print(f'WARNING: --gpu {gpu} out of range (only {n} CUDA device(s)); using cuda:0')
        gpu = 0
    return f'cuda:{gpu}'

# --------------------------------------------------------------------------- #
# Constants copied from process_data.py so the node encoding matches training  #
# --------------------------------------------------------------------------- #
FREQUENCY = 2                      # Hz the int_ee_me weights were trained at
dt = 1.0 / FREQUENCY               # 0.5 s

data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

standardization = {
    'PEDESTRIAN': {
        'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
        'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}},
        'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}
    },
    'VEHICLE': {
        'position': {'x': {'mean': 0, 'std': 80}, 'y': {'mean': 0, 'std': 80}},
        'velocity': {'x': {'mean': 0, 'std': 15}, 'y': {'mean': 0, 'std': 15}, 'norm': {'mean': 0, 'std': 15}},
        'acceleration': {'x': {'mean': 0, 'std': 4}, 'y': {'mean': 0, 'std': 4}, 'norm': {'mean': 0, 'std': 4}},
        'heading': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1},
                    '°': {'mean': 0, 'std': np.pi}, 'd°': {'mean': 0, 'std': 1}}
    }
}


# --------------------------------------------------------------------------- #
# 1. Read unity ground-truth tracks                                        #
# --------------------------------------------------------------------------- #
def load_gt_tracks(gt_json_path, stride):
    """Read gt_agents.json -> per-agent contiguous tracks resampled to 2 Hz.

    gt_agents.json is a time-ordered list of per-frame dicts with 'vehicles' and
    'agents' (pedestrians) lists; each entry has id/type/cx/cy/yaw (map frame).
    We index frames by their position in the list, keep every `stride`-th frame
    (the source is ~10 Hz -> stride 5 gives the 2 Hz the model expects), dedup
    repeated (frame,id) rows, and for each agent keep its longest contiguous run.

    Returns: list of dicts {id, type, first_timestep, x, y, heading} and the total
    number of resampled timesteps.
    """
    frames = json.load(open(gt_json_path))
    kept = frames[::stride]
    n_kept = len(kept)
    timestep_t_ns = np.array([f['t_ns'] for f in kept], dtype=np.int64)   # sim clock per timestep
    # gather per-agent rows keyed by the resampled timestep index
    agents = {}   # id -> {'type': str, 'rows': {t: (x, y, yaw)}}
    for new_t, f in enumerate(kept):
        for key, ntype in (('vehicles', 'VEHICLE'), ('agents', 'PEDESTRIAN')):
            for a in f.get(key) or []:
                aid = a['id']
                rec = agents.setdefault(aid, {'type': ntype, 'rows': {}})
                if new_t in rec['rows']:
                    continue  # dedup repeated per-frame stamps
                rec['rows'][new_t] = (a['cx'], a['cy'], a.get('yaw', 0.0))

    tracks = []
    for aid, rec in agents.items():
        ts = sorted(rec['rows'].keys())
        if len(ts) < 2:
            continue
        # longest contiguous run of timesteps
        best_start, best_len = ts[0], 1
        run_start, run_len = ts[0], 1
        for i in range(1, len(ts)):
            if ts[i] == ts[i - 1] + 1:
                run_len += 1
            else:
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
                run_start, run_len = ts[i], 1
        if run_len > best_len:
            best_start, best_len = run_start, run_len
        if best_len < 2:
            continue
        run_ts = list(range(best_start, best_start + best_len))
        xy = np.array([rec['rows'][t][:2] for t in run_ts], dtype=float)
        yaw = np.array([rec['rows'][t][2] for t in run_ts], dtype=float)
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi   # normalize to [-pi, pi]
        tracks.append({'id': str(aid), 'type': rec['type'], 'first_timestep': best_start,
                       'x': xy[:, 0], 'y': xy[:, 1], 'heading': yaw})
    return tracks, n_kept, timestep_t_ns


def load_ego_poses(poses_csv):
    """Read unity poses.csv (frame_idx,t_ns,gx,gy,gz,yaw) = per-frame ego/sensor
    pose in the map frame. Returns (t_ns, xy, yaw) sorted by t_ns for nearest lookup."""
    df = pd.read_csv(poses_csv)
    order = np.argsort(df['t_ns'].values)
    return (df['t_ns'].values[order].astype(np.int64),
            df[['gx', 'gy']].values[order].astype(float),
            df['yaw'].values[order].astype(float))


def ego_pose_at(t_ns, ego_t, ego_xy, ego_yaw):
    """Nearest ego pose (by sim clock t_ns) to the given timestep."""
    i = int(np.argmin(np.abs(ego_t - t_ns)))
    return ego_xy[i], ego_yaw[i]


# --------------------------------------------------------------------------- #
# 2. Build a Trajectron++ Scene/Environment (mirrors process_data.process_scene)#
# --------------------------------------------------------------------------- #
def build_environment(tracks, n_timesteps, margin=50.0):
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 10.0,
        (env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0,
    }
    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE

    # global offset so scene coords start near 0 (matches process_data convention)
    all_x = np.concatenate([t['x'] for t in tracks])
    all_y = np.concatenate([t['y'] for t in tracks])
    x_min = np.round(all_x.min() - margin)
    y_min = np.round(all_y.min() - margin)
    x_max = np.round(all_x.max() + margin)
    y_max = np.round(all_y.max() + margin)

    scene = Scene(timesteps=n_timesteps, dt=dt, name='sweep_s1')
    scene.map = make_blank_map(x_max - x_min, y_max - y_min)

    for tr in tracks:
        x = tr['x'] - x_min
        y = tr['y'] - y_min
        vx = derivative_of(x, dt)
        vy = derivative_of(y, dt)
        ax = derivative_of(vx, dt)
        ay = derivative_of(vy, dt)

        if tr['type'] == 'VEHICLE':
            heading = tr['heading']
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            data_dict = {
                ('position', 'x'): x, ('position', 'y'): y,
                ('velocity', 'x'): vx, ('velocity', 'y'): vy,
                ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                ('acceleration', 'x'): ax, ('acceleration', 'y'): ay,
                ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                ('heading', 'x'): heading_v[:, 0], ('heading', 'y'): heading_v[:, 1],
                ('heading', '°'): heading, ('heading', 'd°'): derivative_of(heading, dt, radian=True),
            }
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            node_type = env.NodeType.VEHICLE
        else:
            data_dict = {
                ('position', 'x'): x, ('position', 'y'): y,
                ('velocity', 'x'): vx, ('velocity', 'y'): vy,
                ('acceleration', 'x'): ax, ('acceleration', 'y'): ay,
            }
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)
            node_type = env.NodeType.PEDESTRIAN

        node = Node(node_type=node_type, node_id=tr['id'], data=node_data,
                    first_timestep=tr['first_timestep'])
        scene.nodes.append(node)

    env.scenes = [scene]
    return env, scene, (x_min, y_min, x_max, y_max)


def make_blank_map(x_size, y_size):
    """A uniform (empty) 3-channel GeometricMap covering the scene.

    int_ee_me was trained with a VEHICLE map encoder, so the forward pass requires a
    map tensor. We have no simulator map, so this is a zero raster: vehicle prediction
    then relies purely on kinematics + agent interactions, with no road context.
    homography = 3*I matches the nuScenes process_data.py convention (3 px / meter).
    """
    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    data = np.zeros((3, int(np.round(3 * x_size)), int(np.round(3 * y_size))), dtype=np.uint8)
    gmap = GeometricMap(data=data, homography=homography, description='blank')
    return {'PEDESTRIAN': gmap, 'VEHICLE': gmap, 'VISUALIZATION': gmap}


# --------------------------------------------------------------------------- #
# 3. Prediction engine (model -> serializable per-timestep arrays)             #
# --------------------------------------------------------------------------- #
def predict_frame(eval_stg, scene, t, ph, num_samples, min_history_timesteps=1):
    """Run the two predict() calls at one timestep and extract a serializable record:
    per node, history / GT future / sampled trajectories / most-likely path (all in
    scene-local coords). Returns None if no node is predictable at t."""
    with torch.no_grad():
        preds = eval_stg.predict(scene, np.array([t]), ph, num_samples=num_samples,
                                 min_history_timesteps=min_history_timesteps,
                                 z_mode=False, gmm_mode=False, full_dist=True)
        preds_mm = eval_stg.predict(scene, np.array([t]), ph, num_samples=1,
                                    min_history_timesteps=min_history_timesteps,
                                    z_mode=True, gmm_mode=True)
    if not preds:
        return None
    pred_d, hist_d, fut_d = prediction_output_to_trajectories(preds, scene.dt, 10, ph, map=None)
    predmm_d, _, _ = prediction_output_to_trajectories(preds_mm, scene.dt, 10, ph, map=None)
    tk = list(pred_d.keys())[0]
    nodes = []
    for node in sorted(pred_d[tk].keys(), key=lambda n: n.id):
        nodes.append({
            'id': node.id, 'type': node.type.name,
            'history': np.asarray(hist_d[tk][node], dtype=np.float32),
            'future': np.asarray(fut_d[tk][node], dtype=np.float32),
            'samples': np.asarray(pred_d[tk][node][0], dtype=np.float32),   # (S, ph, 2)
            'ml': np.asarray(predmm_d[tk][node][0, 0], dtype=np.float32),    # (ph, 2)
        })
    return {'t': int(t), 'nodes': nodes}


def run_predictions(eval_stg, scene, timesteps, ph, num_samples,
                    min_history_timesteps=1, ego_fn=None):
    """Predict over a list of timesteps -> list of frame records. `ego_fn(t)` (optional)
    returns (pos, yaw) of the ego at t (stored so the visualizer can toggle ego view)."""
    frames = []
    for i, t in enumerate(timesteps):
        rec = predict_frame(eval_stg, scene, t, ph, num_samples, min_history_timesteps)
        if rec is not None:
            ego = None if ego_fn is None else ego_fn(t)
            rec['ego'] = None if ego is None else [np.asarray(ego[0], dtype=float).tolist(),
                                                   float(ego[1])]
            frames.append(rec)
        print(f'  predict [{i + 1}/{len(timesteps)}] t={t}', end='\r', flush=True)
    print()
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt_json', default='unity_data/sweep_s1_hdl_frame_fixed/gt_agents.json')
    p.add_argument('--model_dir', default='../nuScenes/models/int_ee_me')
    p.add_argument('--model_ts', type=int, default=12)
    p.add_argument('--stride', type=int, default=5, help='resample source frames (10Hz) -> 2Hz')
    p.add_argument('--ph', type=int, default=6, help='prediction horizon (steps @2Hz; 6 = 3s)')
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--frame_stride', type=int, default=2, help='render every k-th timestep in the video')
    p.add_argument('--single_t', type=int, default=None, help='render just one timestep (debug)')
    p.add_argument('--out_dir', default='unity_out')
    p.add_argument('--save_env', action='store_true', help='also dump the built Environment pkl')
    p.add_argument('--ego_frame', action='store_true',
                   help='render zoomed-in, in the ego agents coordinate frame (ego at origin, heading up)')
    p.add_argument('--poses_csv', default='unity_data/sweep_s1_hdl_frame_fixed/poses.csv',
                   help='ego/sensor poses (map frame) for --ego_frame')
    p.add_argument('--zoom', type=float, default=80.0, help='ego-frame half-window in metres')
    p.add_argument('--fps', type=float, default=5.0, help='playback frame rate of the output video')
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

    # ------- visualization-only: no model, just load stored predictions -------
    if args.viz_only:
        print(f'Loading predictions from {pred_file}...')
        bundle = load_bundle(pred_file)
        render_bundle(bundle, args.out_dir, ego_frame=args.ego_frame, fps=args.fps,
                      zoom=args.zoom, single=args.single_t is not None, fmt=args.format, style=args.style)
        return

    # ------- prediction: build scene, run model, save bundle -------
    print('Loading GT tracks...')
    tracks, n_timesteps, timestep_t_ns = load_gt_tracks(args.gt_json, args.stride)
    print(f'  {len(tracks)} tracks, {n_timesteps} timesteps @ {FREQUENCY}Hz')

    # ego poses are stored regardless so the visualizer can toggle the ego view later
    ego_t = ego_xy = ego_yaw = None
    if os.path.exists(args.poses_csv):
        ego_t, ego_xy, ego_yaw = load_ego_poses(args.poses_csv)
        print(f'  ego poses: {len(ego_t)} from {args.poses_csv}')

    def ego_for(t):
        if ego_t is None:
            return None
        return ego_pose_at(timestep_t_ns[t], ego_t, ego_xy, ego_yaw)

    print('Building Environment...')
    env, scene, (x_min, y_min, x_max, y_max) = build_environment(tracks, n_timesteps)
    if args.save_env:
        with open(os.path.join(args.out_dir, 'env.pkl'), 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

    # world-coord axis limits from actual agent positions (with a small pad)
    all_x = np.concatenate([t['x'] for t in tracks])
    all_y = np.concatenate([t['y'] for t in tracks])
    xlim = (all_x.min() - 15, all_x.max() + 15)
    ylim = (all_y.min() - 15, all_y.max() + 15)

    device = resolve_device(args.gpu)
    print(f'Loading model from {args.model_dir} (ts={args.model_ts}) on {device}...')
    eval_stg, hyp = load_model(args.model_dir, env, ts=args.model_ts, device=device)

    timesteps = [args.single_t] if args.single_t is not None \
        else list(range(2, n_timesteps - 1, args.frame_stride))
    frames = run_predictions(eval_stg, scene, timesteps, args.ph, args.num_samples, ego_fn=ego_for)

    meta = {'source': 'unity', 'scene': scene.name, 'dt': dt, 'ph': args.ph,
            'num_samples': args.num_samples, 'x_min': x_min, 'y_min': y_min,
            'xlim': xlim, 'ylim': ylim, 'zoom': args.zoom, 'gif_prefix': 'distribution'}
    save_bundle(pred_file, meta, frames)
    print(f'Saved {len(frames)} prediction frames -> {pred_file}')

    # ------- optional rendering from the just-computed bundle -------
    if not args.no_viz:
        render_bundle({'meta': meta, 'frames': frames}, args.out_dir, ego_frame=args.ego_frame,
                      fps=args.fps, zoom=args.zoom, single=args.single_t is not None,
                      fmt=args.format, style=args.style)


if __name__ == '__main__':
    main()
