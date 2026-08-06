"""Feed the checkpoint a perfectly clean input and see whether it still flickers.

The question this settles
------------------------
`flicker.py` shows the predicted 5 s reach swinging by tens of metres frame to frame on both
Unity and nuScenes, real data either way. Two readings survive that:

  (a) the model faithfully amplifies a jittery input -- the acceleration it is fed is a raw
      second difference of position, so it alternates sign, and 0.5*a*T^2 turns that into
      metres of reach; or
  (b) the encoder is unstable across timesteps in its own right, and would flicker even on an
      input with nothing to amplify.

Those differ in what a fix would be, so they are worth separating. This module removes the
input noise entirely: a synthetic agent driving in an exactly straight line at exactly
constant speed, whose acceleration is *identically zero* by construction. Under (a) the
flicker must vanish; under (b) it survives.

The synthetic tracks are built through the same recipe `experiments/nuScenes/process_data.py`
uses for real agents -- positions first, then `derivative_of` for velocity and acceleration,
heading unit vector from the velocity, `d°` from the heading series -- and dropped into a copy
of a real nuScenes Environment, so normalization, attention radii and node types are the
checkpoint's own. Nothing about how the model is called differs from the real-data runs.

What is varied
--------------
    --noise s1,s2,...   Gaussian position noise [m] added BEFORE the derivatives are taken,
                        so it propagates into velocity and acceleration exactly as
                        measurement noise does on a real track. 0.0 is the clean case. This
                        is the interesting knob: it turns the yes/no question into a gain
                        curve, "how much predicted flicker per metre of input jitter".
                        Heading is left exact so the test isolates the position channel.
    --turn_rate         rad/s; 0 is straight, non-zero makes an exact constant-curvature arc
                        (still zero *tangential* acceleration). Straight-line driving is
                        unusually easy, so this checks the clean result is not just that.
    --neighbors n       n more agents, each also exactly constant-velocity, in adjacent
                        lanes. With n=0 the interaction graph is empty and the edge encoder
                        is bypassed; with n>0 it runs, on inputs that are equally clean.
    --map blank|real    `blank` is uniformly drivable, which removes the map encoder as a
                        variable; `real` borrows a nuScenes scene's raster and lays the
                        synthetic track along the straightest real vehicle's own path, so
                        the map encoder sees a plausible road. A CNN over a patch that
                        translates smoothly should contribute nothing to frame-to-frame
                        jitter, and `real` is what checks that.

The statistic is `flicker.flicker_stats`, unchanged, so numbers here are directly comparable
to the Unity and nuScenes tables that module prints. The ground-truth row is the sharpest
reference available: for a constant-velocity agent the true reach is *exactly constant*, so
its `d_std` is 0 by construction and every metre of predicted `d_std` is flicker.

Usage (from experiments/unity_lidar_sim/):
    python src/investigations/synthetic_input.py --gpu 0
    python src/investigations/synthetic_input.py --gpu 0 --map real --neighbors 3
    python src/investigations/synthetic_input.py --gpu 0 --turn_rate 0.05 --plot
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import repo_paths  # noqa: F401  (sys.path side effect)

from environment import Scene, Node, GeometricMap, derivative_of

from investigations.flicker import (load_trajectron, predict_tracks, tracks_to_rows,
                                    summarize, plot_examples)

HERE = os.path.dirname(os.path.abspath(__file__))
UNITY_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_ENV = os.path.abspath(os.path.join(UNITY_ROOT, '..', 'processed',
                                           'nuScenes_test_mini_full.pkl'))
DEFAULT_MODEL = os.path.abspath(os.path.join(UNITY_ROOT, '..', 'nuScenes', 'models', 'int_ee_me'))
DEFAULT_OUT = os.path.join(UNITY_ROOT, 'unity_out', 'synthetic_input')

PX_PER_M = 3.0          # the nuScenes map homography is a pure scale; keep it identical


# --------------------------------------------------------------------------- #
# building a synthetic agent the way process_data.py builds a real one
# --------------------------------------------------------------------------- #
def vehicle_node_data(x, y, heading, dt):
    """The VEHICLE feature frame, derived from positions exactly as process_data.py does.

    Everything the model reads except position is a finite difference of position (or of
    heading), which is the whole point: give it positions with no jitter and the velocity is
    constant and the acceleration is identically zero, with no smoothing anywhere.
    """
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


def cv_track(p0, speed, heading0, turn_rate, n, dt, rng, noise=0.0):
    """Exact constant-speed track: a straight line, or a constant-curvature arc.

    Positions are integrated in closed form rather than stepped, so the speed is constant to
    machine precision and the only acceleration present is the centripetal one a turn
    genuinely requires. Noise is added after generation and before any derivative is taken.
    """
    t = np.arange(n) * dt
    heading = heading0 + turn_rate * t
    if abs(turn_rate) < 1e-9:
        pos = p0 + speed * t[:, None] * np.array([np.cos(heading0), np.sin(heading0)])
    else:
        r = speed / turn_rate
        pos = p0 + r * np.stack([np.sin(heading) - np.sin(heading0),
                                 np.cos(heading0) - np.cos(heading)], axis=-1)
    if noise > 0:
        pos = pos + rng.normal(0.0, noise, size=pos.shape)
    return pos[:, 0].copy(), pos[:, 1].copy(), heading


def blank_map(width_m, height_m):
    """A uniformly drivable raster, so the map encoder is a constant and cannot contribute."""
    shape = (3, int(width_m * PX_PER_M), int(height_m * PX_PER_M))
    data = np.full(shape, 255, dtype=np.uint8)
    homography = np.array([[PX_PER_M, 0., 0.], [0., PX_PER_M, 0.], [0., 0., PX_PER_M]])
    return GeometricMap(data=data, homography=homography, description='synthetic: all drivable')


def straightest_vehicle(scene):
    """The real vehicle whose path is closest to a straight line, with its mean velocity.

    Used to place a synthetic track on a real road: starting where it started and running
    along its own average heading keeps the agent on tarmac for the length of the scene,
    which is what makes the map encoder see something plausible.
    """
    best, best_score = None, -np.inf
    for node in scene.nodes:
        if str(node.type) != 'VEHICLE' or node.timesteps < 10:
            continue
        p = node.get(np.array([node.first_timestep, node.last_timestep]),
                     {'position': ['x', 'y']})
        p = p[~np.isnan(p).any(axis=1)]
        if len(p) < 10:
            continue
        path_len = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        disp = np.linalg.norm(p[-1] - p[0])
        if path_len < 20.0:
            continue
        score = disp / path_len                       # 1.0 is perfectly straight
        if score > best_score:
            best, best_score = p, score
    if best is None:
        return None
    heading = np.arctan2(*(best[-1] - best[0])[::-1])
    return best[0], heading, best_score


def build_scene(proto_env, dt, n_steps, speed, heading0, turn_rate, neighbors, noise,
                map_mode, proto_scene, rng):
    """A Scene holding one synthetic ego-less agent (plus optional neighbours) and a map."""
    if map_mode == 'real' and proto_scene is not None:
        placed = straightest_vehicle(proto_scene)
        if placed is None:
            raise SystemExit('no usable straight vehicle in the scene to place a track on')
        p0, heading0, _ = placed
        scene_map = proto_scene.map
    else:
        span = speed * n_steps * dt + 120.0
        p0 = np.array([60.0, span / 2.0])
        scene_map = {'VEHICLE': blank_map(span, span), 'PEDESTRIAN': blank_map(span, span)}

    scene = Scene(timesteps=n_steps, dt=dt, name='synthetic')
    lateral = np.array([-np.sin(heading0), np.cos(heading0)])
    for i in range(1 + neighbors):
        # neighbours sit in adjacent lanes, offset laterally and staggered along the road so
        # the interaction graph is populated without anyone being on top of anyone else
        offset = p0 + lateral * (3.5 * ((i + 1) // 2) * (1 if i % 2 else -1)) \
                    + np.array([np.cos(heading0), np.sin(heading0)]) * (8.0 * i)
        x, y, heading = cv_track(offset, speed, heading0, turn_rate, n_steps, dt, rng, noise)
        node = Node(node_type=proto_env.NodeType.VEHICLE, node_id=f'cv{i}',
                    data=vehicle_node_data(x, y, heading, dt))
        node.first_timestep = 0
        scene.nodes.append(node)
    scene.map = scene_map
    return scene


# --------------------------------------------------------------------------- #
def print_table(summary):
    """`flicker.summarize`'s rows, with the case label where the source column would be."""
    hdr = (f"{'case':<24}{'horizon':>8}{'est':>6}{'trk':>5}{'reach':>9}{'d_std':>8}"
           f"{'ratio':>8}{'d_lag1':>8}{'a_sig':>8}{'speed':>8}")
    print(hdr)
    print('-' * len(hdr))
    last = None
    for r in summary:
        if last is not None and r['source'] != last:
            print()
        last = r['source']
        print(f"{r['source']:<24}{r['horizon_s']:>7.1f}s{r['est']:>6}{r['tracks']:>5}"
              f"{r['mean_reach']:>9.1f}{r['d_std']:>8.2f}{r['ratio']:>8.3f}"
              f"{r['d_lag1']:>8.2f}{r['a_sigma']:>8.3f}{r['speed']:>8.2f}")
    print('\nreach/d_std/a_sig in m, m and m/s^2; speed m/s; medians over agents.')
    print('a_sig is the input acceleration the model was actually fed.')
    print('gt d_std is ~0 by construction: a constant-velocity agent has a constant reach,')
    print('so its d_lag1 is undefined (nan) and every metre of predicted d_std is flicker.')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default=DEFAULT_ENV,
                   help='real environment to borrow normalization / attention radii from')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--checkpoint', type=int, default=12)
    p.add_argument('--gpu', default='0', help="GPU index, or 'cpu'")
    p.add_argument('--horizons', default='6,10', help='horizons in steps (dt = 0.5 s)')
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--steps', type=int, default=40, help='timesteps in the synthetic scene')
    p.add_argument('--speed', type=float, default=8.0, help='m/s')
    p.add_argument('--heading', type=float, default=0.0, help='rad, blank map only')
    p.add_argument('--turn_rate', type=float, default=0.0, help='rad/s; 0 = straight')
    p.add_argument('--neighbors', type=int, default=0)
    p.add_argument('--noise', default='0.0,0.02,0.05,0.1,0.2,0.5',
                   help='position noise sigmas [m] to sweep')
    p.add_argument('--map', dest='map_mode', default='blank', choices=['blank', 'real'])
    p.add_argument('--scene', default=None, help='scene to borrow the map from (--map real)')
    p.add_argument('--min_len', type=int, default=8, help='shortest usable track, in frames')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--out_dir', default=DEFAULT_OUT)
    args = p.parse_args()

    import dill
    import warnings

    horizons = sorted(int(h) for h in args.horizons.split(','))
    sigmas = [float(s) for s in args.noise.split(',')]
    device = 'cpu' if args.gpu == 'cpu' else f'cuda:{args.gpu}'

    with open(args.env, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    proto_scene = env.scenes[0]
    if args.scene is not None:
        match = [s for s in env.scenes if str(s.name) == str(args.scene)]
        proto_scene = match[0] if match else proto_scene
    dt = proto_scene.dt

    # loaded once against the real environment: normalization and attention radii come from
    # the checkpoint's own training env, and swapping in a synthetic scene afterwards does
    # not change either (the model reads scenes only through the `scene` argument).
    stg, hyperparams = load_trajectron(args.model, args.checkpoint, env, device)

    rows, all_tracks = [], {}
    for sigma in sigmas:
        rng = np.random.default_rng(args.seed)
        scene = build_scene(env, dt, args.steps, args.speed, args.heading, args.turn_rate,
                            args.neighbors, sigma, args.map_mode, proto_scene, rng)
        env.scenes = [scene]
        label = (f'{args.map_mode} noise={sigma:g}'
                 + (f' turn={args.turn_rate:g}' if args.turn_rate else '')
                 + (f' nb={args.neighbors}' if args.neighbors else ''))
        tracks = predict_tracks(stg, env, [scene], horizons, args.num_samples, hyperparams,
                                source=label, node_type='VEHICLE', desc=label)
        rows += tracks_to_rows(tracks, dt, horizons, args.min_len, min_speed=0.0)
        all_tracks[label] = tracks

    if not rows:
        print('no usable predictions')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'synthetic_input.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with warnings.catch_warnings():
        # the clean ground-truth reach is exactly constant, so its lag-1 is an all-NaN
        # median -- that nan is the expected answer here, not a problem to report
        warnings.simplefilter('ignore', RuntimeWarning)
        summary = summarize(rows, horizons, dt)
    print_table(summary)
    print(f'\nper-case rows -> {csv_path}')
    if args.plot:
        path = plot_examples(all_tracks, horizons, args.out_dir, n_per_source=1,
                             name='synthetic_reach.png')
        if path:
            print(f'reach traces -> {path}')


if __name__ == '__main__':
    main()
