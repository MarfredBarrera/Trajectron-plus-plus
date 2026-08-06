"""Tell the model the agents are driving at constant velocity, and price what that buys.

Where this comes from
---------------------
`synthetic_input.py` establishes that the flicker is bought entirely with input noise: a
synthetic agent whose acceleration is identically zero produces a rock-steady prediction
(5 s reach moving 0.09 m frame to frame), and adding Gaussian position noise of sigma s buys
back the flicker along a measured gain curve (s = 0.1 m -> 18.5 m). Real tracks sit at
s ~ 0.12 m, which is where the 12-16 m of measured flicker comes from.

The obvious move follows: if we believe the agents are driving at constant velocity, don't
hand the model a raw second difference of position -- hand it the constant-velocity fit. This
module implements that as a causal prefilter and measures both sides of the trade.

The prefilter
-------------
At each timestep t, take the agent's last `--window` raw positions, fit p(tau) = p0 + v*tau by
least squares, and rebuild the node's state from the fitted positions through the same
`process_data.py` arithmetic. Because the fitted positions are exactly collinear in time the
second difference is zero to floating point, so the acceleration channel the model reads is
exactly 0 and `d°` is exactly 0 -- the same input that made the synthetic agent steady. It is
causal: only positions up to and including t are used, so this is something an online system
could actually run.

Why this is not free
--------------------
A constant-velocity fit is a *model* of the agent, and it is wrong exactly where agents are
interesting. Measured on the Unity intersection tracks, three of eight vehicles hold speed to
within the noise floor while the rest vary speed by ~2 m/s (they brake for their turns), and
nearly all of them turn 90-180 degrees over the scene. Fitting a straight constant-speed line
through a braking, turning agent throws away real signal, so the flicker must be weighed
against what it costs in accuracy. Hence every configuration here reports both:

    d_std, ratio, d_lag1   the flicker statistic of `flicker.py`, unchanged
    ADE, FDE               average and final displacement error against the logged future,
                           in metres, over the same (agent, timestep) pairs

and `raw` versus `cv` are scored on *identical* pairs, so the two columns can be read against
each other directly. The baseline is itself rebuilt from a window rather than read from the
full scene, so the only thing that differs between the two modes is the fit.

Reading the result
------------------
Flicker falling is expected and is not by itself the answer -- a prediction pinned to a
straight line at constant speed cannot flicker, and would also be useless. The question this
module exists to answer is whether ADE/FDE hold up while it falls, and at which `--window`
the two curves cross.

Only VEHICLE nodes are refitted. Pedestrians are carried through raw in both modes so they
are not a difference between them, and `--heading raw` keeps the logged heading annotation
(nuScenes measures heading separately rather than deriving it from position) if you want the
position channel isolated from the heading channel.

Usage (from experiments/unity_lidar_sim/):
    python src/investigations/cv_prefilter.py --gpu 0
    python src/investigations/cv_prefilter.py --gpu 0 --windows 5,9,13
    python src/investigations/cv_prefilter.py --gpu 0 --env ../processed/nuScenes_train_mini_full.pkl
"""
import os
import sys
import csv
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import repo_paths  # noqa: F401  (sys.path side effect)

from environment import Scene, Node, derivative_of
from investigations.flicker import flicker_stats, load_trajectron

HERE = os.path.dirname(os.path.abspath(__file__))
UNITY_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_ENV = os.path.abspath(os.path.join(UNITY_ROOT, '..', 'processed',
                                           'nuScenes_test_mini_full.pkl'))
DEFAULT_MODEL = os.path.abspath(os.path.join(UNITY_ROOT, '..', 'nuScenes', 'models', 'int_ee_me'))
DEFAULT_OUT = os.path.join(UNITY_ROOT, 'unity_out', 'cv_prefilter')

POSITION = {'position': ['x', 'y']}


# --------------------------------------------------------------------------- #
# the prefilter
# --------------------------------------------------------------------------- #
def cv_fit(pos, dt):
    """Least-squares constant-velocity fit of a window of positions, evaluated back on the
    window's own timestamps. Returns (fitted positions, velocity, heading).

    Fitting and then *evaluating* rather than just taking the endpoint keeps the whole
    history the encoder reads on one straight line, which is what makes the acceleration the
    model is fed exactly zero rather than merely small.
    """
    n = len(pos)
    tau = np.arange(n) * dt
    A = np.stack([np.ones(n), tau], axis=1)
    coef, *_ = np.linalg.lstsq(A, pos, rcond=None)      # coef = [[p0x, p0y], [vx, vy]]
    fitted = A @ coef
    vel = coef[1]
    return fitted, vel, float(np.arctan2(vel[1], vel[0]))


def vehicle_frame(x, y, heading, dt):
    """The VEHICLE feature frame, derived from positions exactly as process_data.py does."""
    vx, vy = derivative_of(x, dt), derivative_of(y, dt)
    ax, ay = derivative_of(vx, dt), derivative_of(vy, dt)
    v = np.stack((vx, vy), axis=-1)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
    data = {('position', 'x'): x, ('position', 'y'): y,
            ('velocity', 'x'): vx, ('velocity', 'y'): vy,
            ('velocity', 'norm'): np.linalg.norm(v, axis=-1),
            ('acceleration', 'x'): ax, ('acceleration', 'y'): ay,
            ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
            ('heading', 'x'): heading_v[:, 0], ('heading', 'y'): heading_v[:, 1],
            ('heading', '°'): heading,
            ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(list(data.keys())))


def pedestrian_frame(x, y, dt):
    vx, vy = derivative_of(x, dt), derivative_of(y, dt)
    data = {('position', 'x'): x, ('position', 'y'): y,
            ('velocity', 'x'): vx, ('velocity', 'y'): vy,
            ('acceleration', 'x'): derivative_of(vx, dt), ('acceleration', 'y'): derivative_of(vy, dt)}
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(list(data.keys())))


def window_node(node, t, window, dt, mode, heading_mode):
    """Rebuild `node` over the `window` timesteps ending at global time t, re-indexed so the
    window occupies local times [0, W-1]. Returns (node, anchor position) or None.

    Re-indexing rather than keeping global time is what keeps this cheap: the scene handed to
    the model is `window` timesteps long, so its scene graph costs the same at every t.
    """
    lo = max(node.first_timestep, t - window + 1)
    if t > node.last_timestep or t - lo + 1 < 2:
        return None
    pos = node.get(np.array([lo, t]), POSITION)
    if np.isnan(pos).any():
        return None
    n = len(pos)

    if str(node.type) == 'VEHICLE':
        head = node.get(np.array([lo, t]), {'heading': ['°']})[:, 0]
        if mode == 'cv':
            pos, _, fit_heading = cv_fit(pos, dt)
            if heading_mode == 'fit':
                head = np.full(n, fit_heading)
        frame = vehicle_frame(pos[:, 0].copy(), pos[:, 1].copy(), head, dt)
    else:
        frame = pedestrian_frame(pos[:, 0].copy(), pos[:, 1].copy(), dt)

    new = Node(node_type=node.type, node_id=node.id, data=frame)
    new.first_timestep = window - n              # so the window always ends at local W-1
    return new, pos[-1]


def window_scene(scene, t, window, dt, mode, heading_mode):
    """A `window`-timestep scene holding every node present at global time t, refit per mode.

    The map is the original scene's object -- the same raster, in the same coordinates, since
    nothing here moves any agent out of the scene frame.
    """
    local = Scene(timesteps=window, map=scene.map, dt=dt, name=f'{scene.name}@{t}')
    anchors = {}
    for node in scene.nodes:
        built = window_node(node, t, window, dt, mode, heading_mode)
        if built is None:
            continue
        new, anchor = built
        local.nodes.append(new)
        if str(node.type) == 'VEHICLE':
            anchors[node.id] = (node, anchor)
    return local, anchors


# --------------------------------------------------------------------------- #
# running one configuration
# --------------------------------------------------------------------------- #
def run_mode(stg, env, scenes, hyp, horizons, num_samples, window, mode, heading_mode,
             dt, label):
    """Predict at every timestep of every scene through the prefilter, and collect both the
    reach series (for flicker) and the displacement errors (for accuracy), keyed by
    (scene, agent, t) so the modes can be paired afterwards."""
    import torch
    from tqdm import tqdm

    ph = max(horizons)
    out = {}
    for scene in tqdm(scenes, desc=label):
        for t in range(scene.timesteps):
            local, anchors = window_scene(scene, t, window, dt, mode, heading_mode)
            if not anchors:
                continue
            local.calculate_scene_graph(env.attention_radius,
                                        hyp['edge_addition_filter'],
                                        hyp['edge_removal_filter'])
            lt = window - 1
            with torch.no_grad():
                ml = stg.predict(local, np.array([lt]), ph, num_samples=1, z_mode=True,
                                 gmm_mode=True, full_dist=False, min_future_timesteps=0)
                samp = stg.predict(local, np.array([lt]), ph, num_samples=num_samples,
                                   full_dist=False, min_future_timesteps=0)
            for node, pred in ml.get(lt, {}).items():
                if str(node.type) != 'VEHICLE' or node.id not in anchors:
                    continue
                orig, anchor = anchors[node.id]
                gt = orig.get(np.array([t + 1, t + ph]), POSITION)
                out[(scene.name, node.id, t)] = {
                    'anchor': anchor,
                    'ml': np.asarray(pred[0, 0]),                       # (ph, 2)
                    'mean': np.asarray(samp[lt][node][:, 0].mean(axis=0)),
                    'gt': np.asarray(gt)}                               # (ph, 2), may hold NaN
    return out


def score(records, horizons, dt, label, min_len=8):
    """Flicker (per agent, over its reach series) and error (per prediction), one row per
    horizon and estimator."""
    rows = []
    by_agent = {}
    for (scene, agent, t), rec in records.items():
        by_agent.setdefault((scene, agent), []).append((t, rec))

    for h in horizons:
        for est in ('ml', 'mean'):
            flick, ade, fde = [], [], []
            for key, items in by_agent.items():
                items.sort(key=lambda it: it[0])
                ts = np.array([t for t, _ in items])
                reach = np.array([np.linalg.norm(r[est][h - 1] - r['anchor']) for _, r in items])
                # only differences between adjacent frames are flicker
                splits = np.flatnonzero(np.diff(ts) != 1) + 1
                for seg in np.split(np.arange(len(ts)), splits):
                    if len(seg) >= min_len:
                        flick.append(flicker_stats(reach[seg]))
                for _, r in items:
                    err = np.linalg.norm(r[est][:h] - r['gt'][:h], axis=1)
                    if np.isnan(err).any():
                        continue
                    ade.append(err.mean())
                    fde.append(err[-1])
            if not flick:
                continue
            med = lambda k: float(np.nanmedian([f[k] for f in flick]))     # noqa: E731
            rows.append({'case': label, 'horizon_s': h * dt, 'est': est,
                         'agents': len(flick), 'preds': len(ade),
                         'reach': med('mean'), 'd_std': med('d_std'), 'ratio': med('ratio'),
                         'd_lag1': med('d_lag1'),
                         'ADE': float(np.mean(ade)) if ade else np.nan,
                         'FDE': float(np.mean(fde)) if fde else np.nan})
    return rows


def print_table(rows):
    hdr = (f"{'case':<20}{'horizon':>8}{'est':>6}{'agents':>7}{'preds':>7}{'reach':>8}"
           f"{'d_std':>8}{'ratio':>8}{'d_lag1':>8}{'ADE':>8}{'FDE':>8}")
    print(hdr)
    print('-' * len(hdr))
    last = None
    for r in rows:
        if last is not None and r['case'] != last:
            print()
        last = r['case']
        print(f"{r['case']:<20}{r['horizon_s']:>7.1f}s{r['est']:>6}{r['agents']:>7}"
              f"{r['preds']:>7}{r['reach']:>8.1f}{r['d_std']:>8.2f}{r['ratio']:>8.3f}"
              f"{r['d_lag1']:>8.2f}{r['ADE']:>8.2f}{r['FDE']:>8.2f}")
    print('\nreach/d_std/ADE/FDE in m. Flicker is a median over agent tracks; ADE/FDE are')
    print('means over predictions. raw and cv are scored on identical (agent, timestep) pairs.')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default=DEFAULT_ENV)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--checkpoint', type=int, default=12)
    p.add_argument('--gpu', default='0', help="GPU index, or 'cpu'")
    p.add_argument('--horizons', default='6,10', help='horizons in steps (dt = 0.5 s)')
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--windows', default='9',
                   help='prefilter window lengths in frames, comma separated')
    p.add_argument('--heading', dest='heading_mode', default='fit', choices=['fit', 'raw'],
                   help="'fit' takes heading from the fitted velocity; 'raw' keeps the log's")
    p.add_argument('--max_scenes', type=int, default=None)
    p.add_argument('--min_len', type=int, default=8)
    p.add_argument('--out_dir', default=DEFAULT_OUT)
    args = p.parse_args()

    import dill
    horizons = sorted(int(h) for h in args.horizons.split(','))
    windows = [int(w) for w in args.windows.split(',')]
    device = 'cpu' if args.gpu == 'cpu' else f'cuda:{args.gpu}'

    with open(args.env, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    scenes = env.scenes[:args.max_scenes] if args.max_scenes else env.scenes
    dt = scenes[0].dt

    stg, hyp = load_trajectron(args.model, args.checkpoint, env, device)

    rows = []
    for window in windows:
        for mode in ('raw', 'cv'):
            label = f'{mode} w={window}' if mode == 'cv' else f'raw w={window}'
            recs = run_mode(stg, env, scenes, hyp, horizons, args.num_samples, window,
                            mode, args.heading_mode, dt, label)
            rows += score(recs, horizons, dt, label, args.min_len)

    if not rows:
        print('no usable predictions')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'cv_prefilter.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print_table(rows)
    print(f'\nper-case rows -> {csv_path}')


if __name__ == '__main__':
    main()
