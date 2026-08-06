"""
Scene ingest for the Unity-sim prediction driver.

`unity_online.py` (streaming replay via BatchedOnlineTrajectron.incremental_forward)
needs this whole setup before any model runs:

    YAML config -> GT tracks + ego poses -> track filtering -> scene bounds
                -> drivable-area GeometricMap -> Environment / Scene / Nodes

That whole pipeline lives here, behind `prepare_scene(cfg)`, which returns a
`UnityScene` holding the Environment, the Scene, the ego pose lookups, and the
bundle metadata the driver writes to disk. The driver should not reach past
`UnityScene` into the raw JSON/CSV.

The Node encoding (column layout, 2 Hz cadence, standardization) is copied from
nuScenes `process_data.py` so it matches what the int_ee_me weights were trained on.
"""
import os
import json

import numpy as np
import pandas as pd
import yaml
import dill
import torch
from PIL import Image

import repo_paths  # noqa: F401  (sys.path side effect: trajectron/)
from environment import Environment, Scene, Node, GeometricMap, derivative_of


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
# YAML config (see configs/config.yaml for the annotated reference)            #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    'data_dir': 'unity_data/sweep_s1_hdl_frame_fixed',
    'model_dir': '../nuScenes/models/int_ee_me',
    'model_ts': 12,
    'stride': 5,
    'ph': 6,
    'num_samples': 200,
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
    # Causal constant-velocity prefilter on VEHICLE tracks (see cv_filter_tracks): at each
    # timestep the last `cv_filter_window` positions are fit by least squares and the model
    # is handed the fit's position and velocity with a zero acceleration, instead of a raw
    # second difference of position. 0 or null = off (raw differences, the original
    # behaviour). This is the flicker lever: it removes the channel the horizon squares.
    # Keep the window short -- a long one fits a straight line through a turning vehicle and
    # costs accuracy; 3 is the measured sweet spot, and above ~9 it is clearly harmful.
    'cv_filter_window': 0,
    # --- online run + risk scoring -------------------------------------------- #
    # Timesteps streamed into the encoders before the first prediction, which lands at
    # warmup_timesteps + 1. Longer warm-up = each agent's LSTM has more context at the
    # first prediction; see online_engine.OnlineEngine._warm_up.
    'warmup_timesteps': 1,
    # Observations an agent needs before it is predicted for. 1 = from its second
    # observation onwards, the conventional rule.
    'min_history_timesteps': 1,
    # The ego path stored in the bundle for context and drawn by the visualizers. Not an
    # input to the risk metric, which only uses the ego's pose at t:
    #   'logged'    - the ego's actual future over the horizon (uses information not
    #                 causally available at t)
    #   'projected' - constant-velocity dead reckoning from the ego's current pose,
    #                 i.e. what an online planner would actually have.
    'ego_path_mode': 'logged',
    # Feed the ego's planned future to the model (Trajectron++ `incl_robot_node`): the ego
    # joins the scene as the robot node, so it both becomes a neighbour in the interaction
    # graph and conditions every agent's prediction on where the ego is going. Requires a
    # checkpoint trained with incl_robot_node: true; the driver refuses to run if the
    # config and the checkpoint disagree.
    'ego_conditioning': False,
    'risk_radius': 25.0,       # metres; radius of the ego's keep-out disc (see risk_eval)
    'risk_file': None,         # null = <out_dir>/risk_online.csv
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
# 1. Read unity ground-truth tracks                                            #
# --------------------------------------------------------------------------- #
def _longest_time_run(t, max_gap):
    """Slice of the longest stretch of `t` with no gap wider than `max_gap`."""
    breaks = np.flatnonzero(np.diff(t) > max_gap) + 1
    segments = np.split(np.arange(t.size), breaks)
    return max(segments, key=len)


def load_gt_tracks(gt_json_path, stride, target_dt=dt):
    """Read gt_agents.json -> per-agent tracks on an exact `target_dt` grid.

    gt_agents.json is a time-ordered list of per-frame dicts with 'vehicles' and
    'agents' (pedestrians) lists; each entry has id/type/cx/cy/yaw (map frame), and
    each frame carries the sim clock in `t_ns`.

    Resampling is done in TIME, not by index. The source runs at a nominal 10 Hz but
    the actual spacing jitters (measured on sweep_s1_rail: taking every 5th frame gives
    0.5177 +/- 0.0272 s, not 0.5). Everything downstream -- `derivative_of`, the model's
    velocity and acceleration inputs, the dynamics -- assumes a uniform `dt`, so feeding
    it irregularly spaced samples labelled as uniform manufactures acceleration that the
    agent never had: at 5 m/s, 27 ms of timing jitter reads as 0.13 m of position error,
    and a second difference turns that into ~1 m/s^2 of pure noise. It also put a 3.5%
    bias on every speed. So each agent's x/y/yaw are interpolated onto a grid that really
    is `target_dt` apart, which is the interval the rest of the pipeline claims it is.

    `stride` is the legacy index-striding rate, kept only for logs written without
    `t_ns`; when timestamps are present it is unused.

    Returns: list of dicts {id, type, first_timestep, x, y, heading}, the number of
    timesteps on the grid, and the grid's sim-clock times in ns.
    """
    frames = json.load(open(gt_json_path))
    if not frames:
        return [], 0, np.empty(0, dtype=np.int64)

    if not all('t_ns' in f for f in frames):
        print('  WARNING: gt_agents.json has no t_ns; falling back to index striding, '
              'which assumes the source rate is exactly uniform')
        return _load_gt_tracks_by_index(frames, stride)

    frames = sorted(frames, key=lambda f: f['t_ns'])
    t_frame = np.array([f['t_ns'] for f in frames], dtype=np.int64) / 1e9
    raw_dt = float(np.median(np.diff(t_frame))) if len(t_frame) > 1 else target_dt

    # the uniform grid the model will believe it is being fed
    n_steps = int(np.floor((t_frame[-1] - t_frame[0]) / target_dt)) + 1
    grid = t_frame[0] + np.arange(n_steps) * target_dt
    timestep_t_ns = np.round(grid * 1e9).astype(np.int64)

    agents = {}   # id -> {'type': str, 'samples': {t_s: (x, y, yaw)}}
    for f, t_s in zip(frames, t_frame):
        for key, ntype in (('vehicles', 'VEHICLE'), ('agents', 'PEDESTRIAN')):
            for a in f.get(key) or []:
                rec = agents.setdefault(a['id'], {'type': ntype, 'samples': {}})
                rec['samples'].setdefault(t_s, (a['cx'], a['cy'], a.get('yaw', 0.0)))

    tracks = []
    for aid, rec in agents.items():
        t_a = np.array(sorted(rec['samples'].keys()), dtype=float)
        if t_a.size < 2:
            continue
        # Only a gap wider than one OUTPUT step means the agent really dropped out --
        # bridging it would invent motion, so the longest continuous stretch is used.
        # The threshold is deliberately not tied to the source rate: this log drops the
        # odd frame globally (0.26 s hiccups on a 0.10 s nominal period) and splitting a
        # track there would discard most of it for no reason.
        seg = _longest_time_run(t_a, target_dt)
        t_seg = t_a[seg]
        if t_seg.size < 2:
            continue
        xyz = np.array([rec['samples'][t] for t in t_seg], dtype=float)
        inside = np.flatnonzero((grid >= t_seg[0]) & (grid <= t_seg[-1]))
        if inside.size < 2:
            continue
        g = grid[inside]
        # yaw is unwrapped before interpolation so a track crossing +-pi does not spin
        yaw = np.interp(g, t_seg, np.unwrap(xyz[:, 2]))
        tracks.append({'id': str(aid), 'type': rec['type'],
                       'first_timestep': int(inside[0]),
                       'x': np.interp(g, t_seg, xyz[:, 0]),
                       'y': np.interp(g, t_seg, xyz[:, 1]),
                       'heading': (yaw + np.pi) % (2.0 * np.pi) - np.pi})
    return tracks, n_steps, timestep_t_ns


def _load_gt_tracks_by_index(frames, stride):
    """The pre-timestamp loader: keep every `stride`-th frame and assume it is uniform.

    Only reached for logs with no `t_ns`. It cannot correct spacing it cannot see, so it
    inherits whatever jitter the source had -- see `load_gt_tracks` for what that costs.
    """
    kept = frames[::stride]
    n_kept = len(kept)
    timestep_t_ns = np.array([f.get('t_ns', 0) for f in kept], dtype=np.int64)
    agents = {}   # id -> {'type': str, 'rows': {t: (x, y, yaw)}}
    for new_t, f in enumerate(kept):
        for key, ntype in (('vehicles', 'VEHICLE'), ('agents', 'PEDESTRIAN')):
            for a in f.get(key) or []:
                rec = agents.setdefault(a['id'], {'type': ntype, 'rows': {}})
                rec['rows'].setdefault(new_t, (a['cx'], a['cy'], a.get('yaw', 0.0)))

    tracks = []
    for aid, rec in agents.items():
        ts = np.array(sorted(rec['rows'].keys()))
        if ts.size < 2:
            continue
        run = _longest_time_run(ts.astype(float), 1.0)     # contiguous timestep indices
        if run.size < 2:
            continue
        run_ts = ts[run]
        xyz = np.array([rec['rows'][t] for t in run_ts], dtype=float)
        tracks.append({'id': str(aid), 'type': rec['type'], 'first_timestep': int(run_ts[0]),
                       'x': xyz[:, 0], 'y': xyz[:, 1],
                       'heading': (xyz[:, 2] + np.pi) % (2.0 * np.pi) - np.pi})
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


def cv_filter_tracks(tracks, window, dt_s=dt):
    """Causal constant-velocity prefilter over each VEHICLE track, in place.

    At every timestep i, fit p(tau) = p0 + v*tau by least squares to the positions in
    [i-window+1, i] and keep the fit's value and slope AT i. Only past samples are used, so
    this is something an online state estimator could actually produce; it is not smoothing
    with hindsight.

    Why it exists: `src/investigations/synthetic_input.py` shows the prediction flicker is
    bought entirely with input position noise, because the acceleration the model reads is a
    raw second difference and 0.5*a*T^2 magnifies it. Handing the model the fitted velocity
    and a zero acceleration instead removes the amplified channel at the source. The track's
    `cv_state` entry is what `make_node` then uses in place of the finite differences.

    The trade is real and window-dependent: constant velocity is a *model* of the agent, and
    a long window fits a straight line through a braking, turning vehicle. Measured
    rolling-fit residual RMS on these tracks is 0.25 / 0.58 / 0.91 m at window 5 / 9 / 13
    against a ~0.13 m noise floor, so past ~5 the fit is discarding real motion. Short
    windows are the usable range; see the module docstring of investigations/cv_prefilter.py
    for the accuracy measurements behind that.
    """
    if not window or window < 2:
        return tracks
    for tr in tracks:
        if tr['type'] != 'VEHICLE':
            continue                       # pedestrians are not constant-velocity in any useful sense
        pos = np.stack((tr['x'], tr['y']), axis=-1)
        n = len(pos)
        fit_pos = np.array(pos, dtype=float)
        fit_vel = np.zeros_like(fit_pos)
        for i in range(n):
            lo = max(0, i - window + 1)
            seg = pos[lo:i + 1]
            if len(seg) < 2:
                continue
            tau = np.arange(len(seg)) * dt_s
            A = np.stack([np.ones(len(seg)), tau], axis=1)
            coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
            fit_pos[i] = coef[0] + coef[1] * tau[-1]
            fit_vel[i] = coef[1]
        # heading follows the fitted velocity while the agent is moving; below that the
        # direction of a near-zero vector is meaningless, so the logged heading stands
        speed = np.linalg.norm(fit_vel, axis=-1)
        heading = np.where(speed > 1.0, np.arctan2(fit_vel[:, 1], fit_vel[:, 0]),
                           tr['heading'])
        tr['x'], tr['y'] = fit_pos[:, 0], fit_pos[:, 1]
        tr['heading'] = np.unwrap(heading)
        tr['cv_state'] = fit_vel
    return tracks


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


def make_node(env, tr, x_min, y_min, is_robot=False):
    """One track dict -> a Trajectron++ Node in scene-local coords.

    Column layout and derivative conventions are copied from nuScenes process_data.py, so
    the node state matches what the pre-trained weights were trained on.
    """
    x = tr['x'] - x_min
    y = tr['y'] - y_min
    if 'cv_state' in tr:
        # prefiltered: velocity is the fit's slope and the acceleration channel -- the one
        # the horizon squares -- is exactly zero, rather than a second difference of noise
        vx, vy = tr['cv_state'][:, 0], tr['cv_state'][:, 1]
        ax = ay = np.zeros_like(vx)
    else:
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

    return Node(node_type=node_type, node_id=tr['id'], data=node_data,
                first_timestep=tr['first_timestep'], is_robot=is_robot)


def ego_track(timestep_t_ns, ego_t, ego_xy, ego_yaw, n_timesteps, node_id='EGO'):
    """The ego's own pose history as a track dict, in the same shape as an agent track.

    poses.csv is sampled on the raw (~10 Hz) clock, so each 2 Hz scene timestep takes the
    nearest pose -- the same lookup used everywhere else the ego is read.
    """
    poses = [ego_pose_at(timestep_t_ns[t], ego_t, ego_xy, ego_yaw) for t in range(n_timesteps)]
    xy = np.array([p[0] for p in poses], dtype=float)
    yaw = np.array([p[1] for p in poses], dtype=float)
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
    return {'id': node_id, 'type': 'VEHICLE', 'first_timestep': 0,
            'x': xy[:, 0], 'y': xy[:, 1], 'heading': yaw}


def build_environment(tracks, n_timesteps, x_min, y_min, x_max, y_max,
                      scene_name='sweep_s1', map_gmap=None, robot_track=None):
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
        scene.nodes.append(make_node(env, tr, x_min, y_min))

    # The ego joins the scene as the robot node when the model conditions on its plan. It is
    # a full scene node, not just a side channel: it becomes a neighbour in the interaction
    # graph like any other vehicle. Both drivers then leave it out of the *predicted* set --
    # `get_timesteps_data` passes `return_robot=not incl_robot_node`, and
    # `BatchedOnlineTrajectron` skips it the same way.
    if robot_track is not None:
        robot = make_node(env, robot_track, x_min, y_min, is_robot=True)
        scene.nodes.append(robot)
        scene.robot = robot

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
# 3. The prepared scene handed to a prediction driver                          #
# --------------------------------------------------------------------------- #
class UnityScene(object):
    """A Unity scene ingested and ready for inference.

    Attributes:
        env, scene      the Trajectron++ Environment / Scene (scene-local coords)
        name            scene name, taken from the data_dir basename
        n_timesteps     number of 2 Hz timesteps in the log
        x_min, y_min    world -> scene-local origin shift applied to every node
        out_dir         resolved output directory (created)
        map_bounds      world bounds of the drivable raster, or None
        robot           the ego as a scene Node, or None unless `ego_conditioning`
    """

    def __init__(self, cfg, name, out_dir, tracks, n_timesteps, timestep_t_ns,
                 ego_t, ego_xy, ego_yaw, bounds, env, scene, map_bounds):
        self.robot = scene.robot
        self.cfg = cfg
        self.name = name
        self.out_dir = out_dir
        self.tracks = tracks
        self.n_timesteps = n_timesteps
        self.timestep_t_ns = timestep_t_ns
        self._ego_t, self._ego_xy, self._ego_yaw = ego_t, ego_xy, ego_yaw
        self.x_min, self.y_min, self.x_max, self.y_max = bounds
        self.env = env
        self.scene = scene
        self.map_bounds = map_bounds

    @property
    def has_ego(self):
        return self._ego_t is not None

    def ego_pose(self, t):
        """(xy, yaw) of the ego at timestep `t` in world coords, or None without poses.csv."""
        if not self.has_ego:
            return None
        return ego_pose_at(self.timestep_t_ns[t], self._ego_t, self._ego_xy, self._ego_yaw)

    def ego_logged_path(self, t, ph):
        """The ego's *actual* future path over the horizon (world coords): positions at
        timesteps t+1..t+ph, truncated at the end of the log. Returns None if unavailable.

        This is ground truth read out of the log, so it is not information an online
        planner would have at time t -- see `ego_projected_path` for the causal version."""
        if not self.has_ego:
            return None
        pts = [self.ego_pose(t + h)[0] for h in range(1, ph + 1) if t + h < self.n_timesteps]
        return [np.asarray(p, dtype=float).tolist() for p in pts] if pts else None

    def ego_projected_path(self, t, ph, dt_s=dt):
        """Constant-velocity dead reckoning of the ego over the horizon (world coords),
        using only poses at or before `t` -- the causally-available stand-in for
        `ego_logged_path`. Velocity is the finite difference over the last timestep;
        at t = 0 there is no velocity yet, so None is returned."""
        if not self.has_ego or t < 1:
            return None
        p1 = np.asarray(self.ego_pose(t)[0], dtype=float)
        p0 = np.asarray(self.ego_pose(t - 1)[0], dtype=float)
        vel = (p1 - p0) / dt_s
        return [(p1 + vel * (h * dt_s)).tolist() for h in range(1, ph + 1)]

    def ego_plan_state(self, t, ph, state_spec):
        """The ego's planned state over [t, t+ph] as the model's robot input, (ph+1, state).

        Scene-local and unstandardized; row 0 is the ego's present state and rows 1..ph its
        plan over the horizon. Returns None unless the scene was built with a robot node
        (`ego_conditioning`).

        With `ego_path_mode: logged` the plan *is* the ego's logged future -- fine when the
        ego is the vehicle being planned for, since a planner knows its own intended path,
        but it is not a causal input for a replayed third-party log.

        Past the end of the log the last known state is held rather than zero-padded. The
        offline preprocessing pads with zeros there, which teleports the ego to the scene
        origin; holding keeps the plan plausible over the final `ph` timesteps.
        """
        if self.robot is None:
            return None
        last = min(t + ph, self.n_timesteps - 1)
        traj = self.robot.get(np.array([t, last]), state_spec)
        if len(traj) < ph + 1:
            traj = np.vstack([traj, np.repeat(traj[-1:], ph + 1 - len(traj), axis=0)])
        return traj

    def ego_path(self, t, ph, mode='logged'):
        """The ego path stored in the bundle for context/viz; `mode` in
        {'logged', 'projected'}. The risk metric does not read it -- see risk_eval.proximity."""
        if mode == 'logged':
            return self.ego_logged_path(t, ph)
        if mode == 'projected':
            return self.ego_projected_path(t, ph)
        raise ValueError(f"unknown ego_path_mode {mode!r} (expected 'logged' or 'projected')")

    def bundle_meta(self, **extra):
        """The `meta` dict of an on-disk prediction bundle (see src/bundle.py for the format).
        Drivers pass their own gif_prefix / extra fields through `extra`."""
        all_x = np.concatenate([t['x'] for t in self.tracks])
        all_y = np.concatenate([t['y'] for t in self.tracks])
        meta = {'source': 'unity', 'scene': self.name, 'dt': dt, 'ph': self.cfg['ph'],
                'num_samples': self.cfg['num_samples'], 'x_min': self.x_min, 'y_min': self.y_min,
                'xlim': (all_x.min() - 15, all_x.max() + 15),
                'ylim': (all_y.min() - 15, all_y.max() + 15),
                'zoom': self.cfg['zoom'], 'gif_prefix': 'distribution',
                'cv_filter_window': self.cfg.get('cv_filter_window') or 0}
        if self.map_bounds is not None:
            meta['map_png'] = os.path.abspath(self.cfg['map_png'])
            meta['map_bounds'] = self.map_bounds
        meta.update(extra)
        return meta


def prepare_scene(cfg):
    """Ingest the scene `cfg` points at: tracks + ego poses -> filtered -> Environment.

    Returns a `UnityScene`. Side effects: creates the output directory and, when
    `cfg['save_env']`, dumps the Environment to <out_dir>/env.pkl."""
    # data_dir unifies gt_agents.json + poses.csv: both live beside each other for a
    # scene, so pointing at one directory can never pair mismatched files.
    gt_json = os.path.join(cfg['data_dir'], 'gt_agents.json')
    poses_csv = os.path.join(cfg['data_dir'], 'poses.csv')
    scene_name = os.path.basename(os.path.normpath(cfg['data_dir'])) or 'unity'
    out_dir = cfg['out_dir'] or os.path.join('unity_out', scene_name)
    os.makedirs(out_dir, exist_ok=True)

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

    n_before = len(tracks)
    tracks = filter_tracks(tracks, timestep_t_ns, ego_t, ego_xy, ego_yaw,
                           ego_radius=cfg['ego_radius'], min_motion=cfg['min_motion'])
    print(f'  {len(tracks)}/{n_before} tracks kept after filtering (near-ego + moving)')
    if not tracks:
        raise SystemExit('no tracks left after filtering -- loosen ego_radius / min_motion')

    if cfg.get('cv_filter_window'):
        cv_filter_tracks(tracks, int(cfg['cv_filter_window']))
        print(f'  constant-velocity prefilter on VEHICLE tracks, '
              f'window {cfg["cv_filter_window"]} frames')

    bounds = scene_bounds(tracks)
    x_min, y_min, x_max, y_max = bounds

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

    # Scene bounds deliberately cover the agent tracks only, ego included or not, so a
    # conditioned and an unconditioned run of the same scene share a coordinate frame. The
    # ego may sit slightly outside them; nothing crops on the bounds except the blank-map
    # fallback, and the ego is never map-encoded (it is not predicted).
    robot_track = None
    if cfg['ego_conditioning']:
        if ego_t is None:
            raise SystemExit(f'ego_conditioning requires ego poses, but {poses_csv} is missing')
        robot_track = ego_track(timestep_t_ns, ego_t, ego_xy, ego_yaw, n_timesteps)
        print('  ego joins the scene as the robot node (conditioning on its planned future)')

    print('Building Environment...')
    env, scene = build_environment(tracks, n_timesteps, x_min, y_min, x_max, y_max,
                                   scene_name=scene_name, map_gmap=model_map,
                                   robot_track=robot_track)
    if cfg['save_env']:
        with open(os.path.join(out_dir, 'env.pkl'), 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

    return UnityScene(cfg, scene_name, out_dir, tracks, n_timesteps, timestep_t_ns,
                      ego_t, ego_xy, ego_yaw, bounds, env, scene, map_bounds)
