"""
Run a pre-trained Trajectron++ model on FutureDet / Unity-sim ground-truth agent
tracks and visualize the predicted trajectory *distribution*.

Scene, model, and render settings live in a YAML config (default config.yaml,
see that file); the command line only takes the knobs that actually change
run-to-run.

Usage (from experiments/unity_lidar_sim/):
    python unity_predict.py
    python unity_predict.py --config configs/sweep_s1_rail.yaml --gpu 0
    python unity_predict.py --style gaussian --format mp4
See `python unity_predict.py -h`. To re-render a stored bundle without
re-running the model, use visualize.py instead.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import yaml
import dill
import torch
from PIL import Image

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
# YAML config (everything except --gpu/--style/--format; see config.yaml)      #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    'data_dir': 'unity_data/sweep_s1_hdl_frame_fixed',
    'model_dir': '../nuScenes/models/int_ee_me',
    'model_ts': 12,
    'stride': 5,
    'ph': 6,
    'num_samples': 200,
    'frame_stride': 1,
    'single_t': None,
    'ego_frame': False,
    'zoom': 80.0,
    'fps': 2.0,
    'out_dir': None,
    'pred_file': None,
    'save_env': False,
    'no_viz': False,
    'workers': None,
    'map_png': 'unity_maps/Demo_drivable.png',
    'map_json': 'unity_maps/Demo_drivable.json',
    'map_for_model': True,
    # Track filtering: we only predict agents relevant to the ego. Drop any agent
    # that never comes within `ego_radius` m of the ego over its lifetime, and any
    # agent that never moves (total path < `min_motion` m). ego_radius=None disables
    # the proximity gate (e.g. when no poses.csv is available it is skipped anyway).
    'ego_radius': 150.0,
    'min_motion': 1.0,
}


def load_config(path):
    """DEFAULT_CONFIG overridden by the YAML file at `path` (missing file -> defaults only)."""
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        unknown = set(user_cfg) - set(cfg)
        if unknown:
            raise SystemExit(f'unknown key(s) in {path}: {sorted(unknown)}')
        cfg.update(user_cfg)
    return cfg


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


def filter_tracks(tracks, timestep_t_ns, ego_t, ego_xy, ego_yaw,
                  ego_radius=150.0, min_motion=1.0):
    """Keep only agents that matter for ego-centric prediction:

      - stationary: drop any agent whose total path length is < `min_motion` m
        (parked props / never-moving detections add a static blob and nothing else).
      - far from ego: drop any agent that never comes within `ego_radius` m of the
        ego over its lifetime -- we only predict agents near the ego. Skipped when
        ego poses are unavailable (ego_t is None) or `ego_radius` is None, in which
        case only the stationary filter applies.

    Distances use each track's own timesteps against the nearest ego pose (matching
    how the ego is sampled everywhere else). Returns the kept subset of `tracks`."""
    kept = []
    for tr in tracks:
        path_len = float(np.sum(np.hypot(np.diff(tr['x']), np.diff(tr['y']))))
        if path_len < min_motion:
            print(f'  drop {tr["type"]} {tr["id"]}: stationary (path {path_len:.2f} m)')
            continue
        if ego_t is not None and ego_radius is not None:
            run_ts = tr['first_timestep'] + np.arange(len(tr['x']))
            min_d = min(float(np.hypot(tr['x'][k] - ep[0], tr['y'][k] - ep[1]))
                        for k, t in enumerate(run_ts)
                        for ep in (ego_pose_at(timestep_t_ns[t], ego_t, ego_xy, ego_yaw)[0],))
            if min_d > ego_radius:
                print(f'  drop {tr["type"]} {tr["id"]}: far from ego (min {min_d:.1f} m > {ego_radius:.0f} m)')
                continue
        kept.append(tr)
    return kept


# --------------------------------------------------------------------------- #
# 2. Build a Trajectron++ Scene/Environment (mirrors process_data.process_scene)#
# --------------------------------------------------------------------------- #
def scene_bounds(tracks, margin=50.0):
    """World-coord (x_min, y_min, x_max, y_max) covering all tracks, padded by `margin`.
    This is also the local-coordinate origin shift `build_environment` applies (matches
    process_data.py's convention of shifting scene coords to start near 0)."""
    all_x = np.concatenate([t['x'] for t in tracks])
    all_y = np.concatenate([t['y'] for t in tracks])
    x_min = np.round(all_x.min() - margin)
    y_min = np.round(all_y.min() - margin)
    x_max = np.round(all_x.max() + margin)
    y_max = np.round(all_y.max() + margin)
    return x_min, y_min, x_max, y_max


def build_environment(tracks, n_timesteps, x_min, y_min, x_max, y_max,
                      scene_name='sweep_s1', map_gmap=None):
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 10.0,
        (env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0,
    }
    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE

    scene = Scene(timesteps=n_timesteps, dt=dt, name=scene_name)
    gmap = map_gmap if map_gmap is not None else make_blank_map(x_max - x_min, y_max - y_min)
    scene.map = {'PEDESTRIAN': gmap, 'VEHICLE': gmap, 'VISUALIZATION': gmap}

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
    return env, scene


def make_blank_map(x_size, y_size):
    """A uniform (empty) 3-channel GeometricMap covering the scene -- the fallback when no
    drivable-area raster is configured (see load_drivable_map).

    int_ee_me was trained with a VEHICLE map encoder, so the forward pass requires a
    map tensor. With no map, this is a zero raster: vehicle prediction then relies purely
    on kinematics + agent interactions, with no road context.
    homography = 3*I matches the nuScenes process_data.py convention (3 px / meter).
    """
    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    data = np.zeros((3, int(np.round(3 * x_size)), int(np.round(3 * y_size))), dtype=np.uint8)
    return GeometricMap(data=data, homography=homography, description='blank')


def load_drivable_map(png_path, json_path, x_min, y_min, target_res_m_per_px=1.0 / 3.0, pad_m=80.0):
    """Load the Unity-exported drivable/undrivable raster (see unity_maps/handoff.md) as a
    GeometricMap for the VEHICLE map encoder, in THIS scene's local coordinates (x_min/y_min
    is the same origin shift build_environment/scene_bounds applies to the tracks).

    This is a soft prior fed to the map encoder, not a hard mask on the output -- nothing
    stops the model from placing probability mass off the drivable ribbon (e.g. an emergency
    vehicle cutting across lanes), it's just less likely, the same way nuScenes drivable-area
    context works for the real dataset.

    Channel 0 = drivable (this raster). Channels 1/2 are road_divider/lane_divider in
    process_data.py's VEHICLE map (see its map_mask_vehicle) -- we don't have that data, so
    they stay zero; int_ee_me's pretrained 3-channel conv still applies unmodified.

    Resampled from the raster's native 0.1 m/px down to `target_res_m_per_px` (default 3 px/m,
    matching what int_ee_me's map encoder was trained on -- see handoff.md "Scale vs. patch
    size"), then padded with `pad_m` metres of non-drivable margin: agents can sit outside the
    mapped area (this raster covers one fixed Unity level, not every scene's full extent), and
    without padding a too-far-out point's cropped patch could wrap around to unrelated map
    content instead of reading as (correctly) unmapped.
    """
    meta = json.load(open(json_path))
    img = np.asarray(Image.open(png_path).convert('L'))              # (H, W), drivable=255

    src_res = meta['resolution_m_per_px']
    scale = src_res / target_res_m_per_px
    new_w = max(1, round(img.shape[1] * scale))
    new_h = max(1, round(img.shape[0] * scale))
    small = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BOX))
    small = (small > 127).astype(np.uint8) * 255                     # re-binarize post-resample

    pad_px = int(round(pad_m / target_res_m_per_px))
    padded = np.pad(small, pad_px, mode='constant', constant_values=0)

    layer = padded.T                                                  # (H,W) -> [col, row]
    data = np.zeros((3,) + layer.shape, dtype=np.uint8)
    data[0] = layer

    H_world = np.array(meta['homography_world_to_px'], dtype=float)   # world (rx,ry) -> (col,row)
    to_world = np.array([[1., 0., x_min], [0., 1., y_min], [0., 0., 1.]])  # scene-local -> world
    rescale = np.diag([scale, scale, 1.0])
    shift = np.array([[1., 0., pad_px], [0., 1., pad_px], [0., 0., 1.]])
    homography = shift @ rescale @ H_world @ to_world                  # scene-local -> padded px

    gmap = GeometricMap(data=data, homography=homography, description=f"drivable:{meta['scene']}")
    b = meta['world_bounds_ros']
    viz_bounds = (b['x_min'], b['x_max'], b['y_min'], b['y_max'])
    return gmap, viz_bounds


# --------------------------------------------------------------------------- #
# 3. Prediction engine (model -> serializable per-timestep arrays)             #
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

    # data_dir unifies gt_agents.json + poses.csv: both live beside each other for a
    # scene, so pointing at one directory can never pair mismatched files.
    gt_json = os.path.join(cfg['data_dir'], 'gt_agents.json')
    poses_csv = os.path.join(cfg['data_dir'], 'poses.csv')
    scene_name = os.path.basename(os.path.normpath(cfg['data_dir'])) or 'unity'
    out_dir = cfg['out_dir'] or os.path.join('unity_out', scene_name)
    os.makedirs(out_dir, exist_ok=True)
    pred_file = cfg['pred_file'] or os.path.join(out_dir, 'predictions.pkl')

    print('Loading GT tracks...')
    tracks, n_timesteps, timestep_t_ns = load_gt_tracks(gt_json, cfg['stride'])
    print(f'  {len(tracks)} tracks, {n_timesteps} timesteps @ {FREQUENCY}Hz')

    # ego poses are stored regardless so the visualizer can toggle the ego view later
    ego_t = ego_xy = ego_yaw = None
    if os.path.exists(poses_csv):
        ego_t, ego_xy, ego_yaw = load_ego_poses(poses_csv)
        print(f'  ego poses: {len(ego_t)} from {poses_csv}')
        # guard against pairing agents from one scene with an ego from another: their
        # sim clocks won't overlap, and nearest-t_ns matching would return garbage.
        agent_t = np.asarray(timestep_t_ns, dtype=np.int64)
        if ego_t.max() < agent_t.min() or ego_t.min() > agent_t.max():
            print(f'  WARNING: ego t_ns range [{ego_t.min()},{ego_t.max()}] does not '
                  f'overlap agent t_ns range [{agent_t.min()},{agent_t.max()}] -- '
                  f'poses.csv likely belongs to a different scene than gt_agents.json.')
    else:
        print(f'  no ego poses found at {poses_csv} (ego view will be unavailable)')

    def ego_for(t):
        if ego_t is None:
            return None
        return ego_pose_at(timestep_t_ns[t], ego_t, ego_xy, ego_yaw)

    def ego_path_for(t):
        """Ego's actual future path over the prediction horizon (world coords), sampled at
        the same cadence as the agent predictions -- positions at timesteps t+1..t+ph,
        truncated near the end of the log. Used as the reference path for the crossing risk."""
        if ego_t is None:
            return None
        pts = [ego_pose_at(timestep_t_ns[t + h], ego_t, ego_xy, ego_yaw)[0]
               for h in range(1, cfg['ph'] + 1) if t + h < n_timesteps]
        return [np.asarray(p, dtype=float).tolist() for p in pts] if pts else None

    n_before = len(tracks)
    tracks = filter_tracks(tracks, timestep_t_ns, ego_t, ego_xy, ego_yaw,
                           ego_radius=cfg['ego_radius'], min_motion=cfg['min_motion'])
    print(f'  {len(tracks)}/{n_before} tracks kept after filtering (near-ego + moving)')

    x_min, y_min, x_max, y_max = scene_bounds(tracks)

    map_gmap = map_bounds = None
    if cfg['map_png'] and cfg['map_json']:
        if os.path.exists(cfg['map_png']) and os.path.exists(cfg['map_json']):
            map_gmap, map_bounds = load_drivable_map(cfg['map_png'], cfg['map_json'], x_min, y_min)
            print(f'  drivable map: {cfg["map_png"]}')
        else:
            print(f'  WARNING: map_png/map_json not found ({cfg["map_png"]}, {cfg["map_json"]}) '
                  f'-- falling back to a blank map')

    # map_for_model decouples the two uses of the map: when False the model runs on a blank
    # map (kinematics + interactions only) while the drivable raster is still stored in meta
    # for the visualization backdrop -- lets you A/B the map's effect without losing the
    # road context in the rendered video. map_bounds stays set, so viz is unaffected.
    model_map = map_gmap if cfg['map_for_model'] else None
    if map_gmap is not None and not cfg['map_for_model']:
        print('  map DISABLED for model inference (blank map); still drawn in visualization')

    print('Building Environment...')
    env, scene = build_environment(tracks, n_timesteps, x_min, y_min, x_max, y_max,
                                   scene_name=scene_name, map_gmap=model_map)
    if cfg['save_env']:
        with open(os.path.join(out_dir, 'env.pkl'), 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

    # world-coord axis limits from actual agent positions (with a small pad)
    all_x = np.concatenate([t['x'] for t in tracks])
    all_y = np.concatenate([t['y'] for t in tracks])
    xlim = (all_x.min() - 15, all_x.max() + 15)
    ylim = (all_y.min() - 15, all_y.max() + 15)

    device = resolve_device(args.gpu)
    print(f'Loading model from {cfg["model_dir"]} (ts={cfg["model_ts"]}) on {device}...')
    eval_stg, hyp = load_model(cfg['model_dir'], env, ts=cfg['model_ts'], device=device)

    timesteps = [cfg['single_t']] if cfg['single_t'] is not None \
        else list(range(2, n_timesteps - 1, cfg['frame_stride']))
    # 'gaussian' renders only the analytic GMM, so skip the expensive sampling pass.
    need_samples = args.style in ('samples', 'both')
    frames = run_predictions(eval_stg, scene, timesteps, cfg['ph'], cfg['num_samples'],
                             ego_fn=ego_for, need_samples=need_samples, ego_path_fn=ego_path_for)

    meta = {'source': 'unity', 'scene': scene.name, 'dt': dt, 'ph': cfg['ph'],
            'num_samples': cfg['num_samples'], 'x_min': x_min, 'y_min': y_min,
            'xlim': xlim, 'ylim': ylim, 'zoom': cfg['zoom'], 'gif_prefix': 'distribution'}
    if map_bounds is not None:
        meta['map_png'] = os.path.abspath(cfg['map_png'])
        meta['map_bounds'] = map_bounds
    save_bundle(pred_file, meta, frames)
    print(f'Saved {len(frames)} prediction frames -> {pred_file}')

    # ------- optional rendering from the just-computed bundle -------
    if not cfg['no_viz']:
        render_bundle({'meta': meta, 'frames': frames}, out_dir, ego_frame=cfg['ego_frame'],
                      fps=cfg['fps'], zoom=cfg['zoom'], single=cfg['single_t'] is not None,
                      fmt=args.format, style=args.style, workers=cfg['workers'])


if __name__ == '__main__':
    main()
