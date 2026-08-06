"""Does the prediction flicker on nuScenes too?

The question
------------
Re-predicting every 0.5 s, the 5 s reach of a Unity agent swings by tens of metres between
consecutive frames even though the agent is driving smoothly. The measured mechanism is
arithmetic rather than stochastic: the decoder emits an acceleration, the dynamics integrate
it as reach ~ v0*T + 0.5*a*T^2, and at T = 5 s that second term multiplies the acceleration
by 12.5. The acceleration *input* is a raw second difference of position, so it alternates
sign frame to frame, and the model -- memoryless across calls, trained only against future
positions -- follows it.

If that reading is right the flicker is a property of the checkpoint and its horizon, not of
our data pipeline, and **the same checkpoint must flicker the same way on nuScenes**. That is
what this module measures. It runs `models/int_ee_me` over a processed nuScenes environment,
computes one flicker statistic there, computes the identical statistic on a stored Unity
bundle, and prints them side by side.

The statistic
-------------
For an agent at timestep t, with p_t its position and yhat_t(h) the predicted position h
steps ahead:

    reach_t = || yhat_t(h) - p_t ||           how far the model says it will get in h*dt

`reach` is used rather than displacement error because it needs no ground-truth future (so it
is defined on every frame, including the last h) and because it is the quantity that visibly
pulsates: the length of the drawn fan. Per agent track we report

    mean          mean reach over the track                       [m]
    d_std         std of the frame-to-frame change in reach       [m]
    ratio         d_std / mean -- flicker as a fraction of reach  [-]
    d_lag1        lag-1 autocorrelation of that change            [-]

`d_lag1` is the discriminating one. A prediction that tracks a genuinely changing intent
drifts, so its increments are uncorrelated (d_lag1 ~ 0); a prediction that oscillates about a
stable value overshoots and comes back, so its increments alternate sign (d_lag1 ~ -0.5). The
same statistic on the ground-truth reach is reported as the baseline the model should match:
the agent's own reach does change frame to frame, and only the *excess* is flicker.

Two estimators of yhat are computed and should agree, which is itself the check that this is
not Monte Carlo noise:
  ml    the deterministic most-likely path (z_mode + gmm_mode, one sample, no randomness)
  mean  the mean over `--num_samples` sampled trajectories

and everything is reported at two horizons, which separates the two terms: reach itself is
dominated by v0*T and grows linearly with the horizon, while the acceleration term grows as
T^2. If the flicker were only the horizon being long, `d_std` would scale like `mean` and
`ratio` would hold flat between 3 s and 5 s; `ratio` rising with the horizon is the
acceleration term making itself felt. The checkpoint was trained at a 3 s horizon, so 5 s is
also extrapolation, and reporting both keeps that from being mistaken for the whole story.

Note on inputs, so the comparison is not confounded: the acceleration statistics of the two
datasets are computed here from positions in exactly the same way (second difference over dt,
projected on the heading) and printed in the same table. If they match, the datasets are
equally noisy and any difference in flicker is the model's.

Unlike the other modules in this package this one does run the model -- there is no stored
bundle for nuScenes -- so it needs a GPU and the checkpoint. The Unity half reads a bundle
only, and `--skip_nuscenes` uses just that.

Usage (from experiments/unity_lidar_sim/):
    python src/investigations/flicker.py --gpu 0
    python src/investigations/flicker.py --env ../processed/nuScenes_train_mini_full.pkl
    python src/investigations/flicker.py --skip_nuscenes --plot
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import repo_paths  # noqa: F401  (sys.path side effect)

from bundle import load_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
UNITY_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_BUNDLE = os.path.join(UNITY_ROOT, 'unity_out', 'intersection_probe',
                              'predictions_online.pkl')
DEFAULT_ENV = os.path.abspath(os.path.join(UNITY_ROOT, '..', 'processed',
                                           'nuScenes_test_mini_full.pkl'))
DEFAULT_MODEL = os.path.abspath(os.path.join(UNITY_ROOT, '..', 'nuScenes', 'models', 'int_ee_me'))
DEFAULT_OUT = os.path.join(UNITY_ROOT, 'unity_out', 'flicker')


# --------------------------------------------------------------------------- #
# the statistic
# --------------------------------------------------------------------------- #
def lag1(x):
    """Lag-1 autocorrelation. NaN when it is not defined (<3 points, or constant)."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 1e-12:
        return np.nan
    return float(x[:-1] @ x[1:] / denom)


def flicker_stats(reach):
    """The per-track summary described in the module docstring."""
    r = np.asarray(reach, dtype=float)
    d = np.diff(r)
    mean = float(r.mean())
    return {'n': int(r.size),
            'mean': mean,
            'std': float(r.std()),
            'p2p': float(r.max() - r.min()),
            'd_std': float(d.std()) if d.size else np.nan,
            'd_absmed': float(np.median(np.abs(d))) if d.size else np.nan,
            'ratio': float(d.std() / mean) if d.size and mean > 1e-6 else np.nan,
            'd_lag1': lag1(d)}


def accel_stats(pos, dt):
    """Longitudinal acceleration of a track, as a raw second difference of position.

    Deliberately not read from any stored velocity/acceleration column: the point is to
    measure the two datasets with one ruler. Returns sigma and the lag-1 autocorrelation of
    the acceleration itself, which is what the flicker mechanism blames.
    """
    p = np.asarray(pos, dtype=float)
    if p.shape[0] < 4:
        return {'a_sigma': np.nan, 'a_lag1': np.nan, 'speed': np.nan}
    v = np.diff(p, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    heading = v[:-1] / np.maximum(speed[:-1, None], 1e-6)     # unit heading at each a
    a_lon = np.einsum('ij,ij->i', a, heading)
    return {'a_sigma': float(a_lon.std()),
            'a_lag1': lag1(a_lon),
            'speed': float(speed.mean())}


# --------------------------------------------------------------------------- #
# track assembly -- one dict per (source, agent), whatever the source
# --------------------------------------------------------------------------- #
ESTIMATORS = ('ml', 'mean', 'gt')


def _longest_run(mask):
    """Indices of the longest run of consecutive True in `mask`."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    splits = np.flatnonzero(np.diff(idx) != 1) + 1
    return max(np.split(idx, splits), key=len)


def tracks_to_rows(tracks, dt, horizons, min_len, min_speed):
    """Turn raw per-frame records into per-track rows, one row per (agent, horizon, est).

    All three estimators are scored on exactly the same frames, so the `gt` row is the
    paired baseline for the `ml`/`mean` rows and not a different set of agents. Since the
    ground-truth endpoint is only defined h steps before a track ends, that common window
    is the prefix where every estimator is finite, and a track too short after that
    windowing is dropped at that horizon rather than reported on a longer one.
    """
    rows = []
    for _, rec in sorted(tracks.items()):
        t = np.asarray(rec['t'])
        order = np.argsort(t)
        t = t[order]
        pos = np.asarray(rec['pos'])[order]
        # only frames that are actually consecutive can be differenced; take the longest run
        splits = np.flatnonzero(np.diff(t) != 1) + 1
        seg = max(np.split(np.arange(t.size), splits), key=len)
        if seg.size < min_len:
            continue
        a = accel_stats(pos[seg], dt)
        if not np.isfinite(a['speed']) or a['speed'] < min_speed:
            continue
        for h in horizons:
            ends = {est: np.asarray(rec[f'{est}_{h}'])[order][seg]
                    for est in ESTIMATORS if f'{est}_{h}' in rec}
            if not ends:
                continue
            common = np.ones(seg.size, dtype=bool)
            for e in ends.values():
                common &= np.isfinite(e).all(axis=1)
            keep = _longest_run(common)
            if keep.size < min_len:
                continue
            for est, e in ends.items():
                reach = np.linalg.norm(e[keep] - pos[seg][keep], axis=1)
                row = {'source': rec['source'], 'scene': rec['scene'], 'agent': rec['agent'],
                       'horizon_s': h * dt, 'est': est}
                row.update(flicker_stats(reach))
                row.update(a)
                rows.append(row)
                rec.setdefault('reach', {})[(h, est)] = (t[seg][keep], reach)
    return rows


def unity_tracks(bundle_path, horizons, min_len, min_speed, node_type='VEHICLE'):
    b = load_bundle(bundle_path)
    dt, ph = b['meta']['dt'], b['meta']['ph']
    if max(horizons) > ph:
        raise ValueError(f'bundle stores ph={ph}; cannot evaluate horizon {max(horizons)}')
    scene = b['meta'].get('scene', 'unity')
    tracks = {}
    for frame in b['frames']:
        for node in frame['nodes']:
            if node_type and node['type'] != node_type:
                continue
            rec = tracks.setdefault(node['id'],
                                    {'source': 'unity', 'scene': scene, 'agent': node['id'],
                                     't': [], 'pos': []})
            rec['t'].append(frame['t'])
            rec['pos'].append(node['history'][-1])
            for h in horizons:
                rec.setdefault(f'ml_{h}', []).append(node['ml'][h - 1])
                rec.setdefault(f'mean_{h}', []).append(node['samples'][:, h - 1].mean(axis=0))
                fut = node['future']
                rec.setdefault(f'gt_{h}', []).append(
                    fut[h - 1] if len(fut) >= h else np.full(2, np.nan))
    return tracks_to_rows(tracks, dt, horizons, min_len, min_speed), dt, tracks


def load_trajectron(model_dir, checkpoint, env, device):
    """The checkpoint, bound to `env`. Attention-radius overrides are applied to `env` in
    place first, because `set_environment` reads them when it builds the edge models."""
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

    registrar = ModelRegistrar(model_dir, device)
    registrar.load_models(checkpoint)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        hyperparams = json.load(f)
    for override in hyperparams.get('override_attention_radius', []):
        nt1, nt2, radius = override.split(' ')
        env.attention_radius[(nt1, nt2)] = float(radius)

    stg = Trajectron(registrar, hyperparams, None, device)
    stg.set_environment(env)
    stg.set_annealing_params()
    return stg, hyperparams


def predict_tracks(stg, env, scenes, horizons, num_samples, hyperparams,
                   source='nuScenes', node_type='VEHICLE', desc='scenes'):
    """Sweep the model over every timestep of every scene and collect raw per-frame records
    in the shape `tracks_to_rows` expects. Shared by the nuScenes and the synthetic drivers
    so both are measured through one prediction path.

    Two passes per timestep: the deterministic most-likely path and the Monte Carlo fan.
    """
    import torch
    from tqdm import tqdm

    ph = max(horizons)
    for scene in scenes:
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    tracks = {}
    for scene in tqdm(scenes, desc=desc):
        for t in range(scene.timesteps):
            ts = np.array([t])
            with torch.no_grad():
                ml = stg.predict(scene, ts, ph, num_samples=1, z_mode=True, gmm_mode=True,
                                 full_dist=False, min_future_timesteps=0)
                samp = stg.predict(scene, ts, ph, num_samples=num_samples, z_mode=False,
                                   gmm_mode=False, full_dist=False, min_future_timesteps=0)
            for node, pred in ml.get(t, {}).items():
                if node_type and str(node.type) != node_type:
                    continue
                key = (scene.name, str(node))
                rec = tracks.setdefault(key, {'source': source, 'scene': scene.name,
                                              'agent': str(node), 't': [], 'pos': []})
                rec['t'].append(t)
                rec['pos'].append(node.get(np.array([t, t]), {'position': ['x', 'y']})[0])
                for h in horizons:
                    rec.setdefault(f'ml_{h}', []).append(pred[0, 0, h - 1])
                    rec.setdefault(f'mean_{h}', []).append(
                        samp[t][node][:, 0, h - 1].mean(axis=0))
                    fut = node.get(np.array([t + h, t + h]), {'position': ['x', 'y']})[0]
                    inside = t + h <= node.last_timestep
                    rec.setdefault(f'gt_{h}', []).append(
                        fut if inside else np.full(2, np.nan))
    return tracks


def nuscenes_tracks(env_path, model_dir, checkpoint, horizons, min_len, min_speed,
                    num_samples, device, node_type='VEHICLE', max_scenes=None):
    import dill

    with open(env_path, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    stg, hyperparams = load_trajectron(model_dir, checkpoint, env, device)
    scenes = env.scenes[:max_scenes] if max_scenes else env.scenes
    tracks = predict_tracks(stg, env, scenes, horizons, num_samples, hyperparams,
                            source='nuScenes', node_type=node_type, desc='nuScenes scenes')
    dt = env.scenes[0].dt
    return tracks_to_rows(tracks, dt, horizons, min_len, min_speed), dt, tracks


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summarize(rows, horizons, dt):
    """Median across agent tracks, which is what the side-by-side table shows."""
    out = []
    for source in sorted({r['source'] for r in rows}):
        for h in horizons:
            for est in ESTIMATORS:
                sel = [r for r in rows
                       if r['source'] == source and r['est'] == est
                       and abs(r['horizon_s'] - h * dt) < 1e-9]
                if not sel:
                    continue
                med = lambda k: float(np.nanmedian([r[k] for r in sel]))  # noqa: E731
                out.append({'source': source, 'horizon_s': h * dt, 'est': est,
                            'tracks': len(sel), 'mean_reach': med('mean'),
                            'd_std': med('d_std'), 'ratio': med('ratio'),
                            'd_lag1': med('d_lag1'), 'a_sigma': med('a_sigma'),
                            'a_lag1': med('a_lag1'), 'speed': med('speed')})
    return out


def print_table(summary):
    hdr = (f"{'source':<9}{'horizon':>8}{'est':>6}{'tracks':>8}{'reach':>9}{'d_std':>8}"
           f"{'ratio':>8}{'d_lag1':>8}{'a_sig':>8}{'a_lag1':>8}{'speed':>8}")
    print(hdr)
    print('-' * len(hdr))
    for r in summary:
        print(f"{r['source']:<9}{r['horizon_s']:>7.1f}s{r['est']:>6}{r['tracks']:>8}"
              f"{r['mean_reach']:>9.1f}{r['d_std']:>8.2f}{r['ratio']:>8.3f}"
              f"{r['d_lag1']:>8.2f}{r['a_sigma']:>8.2f}{r['a_lag1']:>8.2f}{r['speed']:>8.2f}")
    print('\nreach/d_std/a_sig in m, m and m/s^2; speed m/s; medians over agent tracks.')
    print("gt = the agents' own reach, the baseline the model's should match.")


def plot_examples(all_tracks, horizons, out_dir, n_per_source=3, name='flicker_reach.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h = max(horizons)
    panels = []
    for source, tracks in all_tracks.items():
        cand = [(k, v) for k, v in tracks.items()
                if 'reach' in v and (h, 'ml') in v['reach'] and (h, 'gt') in v['reach']]
        cand.sort(key=lambda kv: -len(kv[1]['reach'][(h, 'ml')][0]))
        panels += [(source, v) for _, v in cand[:n_per_source]]
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 1.9 * len(panels)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (source, rec) in zip(axes, panels):
        t_ml, r_ml = rec['reach'][(h, 'ml')]
        t_gt, r_gt = rec['reach'][(h, 'gt')]
        ax.plot(t_gt, r_gt, color='0.55', lw=1.2, label='actual')
        ax.plot(t_ml, r_ml, color='tab:red', lw=1.4, label='predicted')
        ax.set_ylabel('reach [m]')
        agent = rec['agent'].split('/')[-1][:8]     # node ids are long; the CSV has them whole
        ax.set_title(f'{source}  {agent}', fontsize=9, loc='left')
    axes[0].legend(fontsize=8, frameon=False)
    axes[-1].set_xlabel('frame')
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', default=DEFAULT_BUNDLE, help='stored Unity prediction bundle')
    p.add_argument('--env', default=DEFAULT_ENV, help='processed nuScenes environment pickle')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--checkpoint', type=int, default=12)
    p.add_argument('--gpu', default='0', help="GPU index, or 'cpu'")
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--horizons', default='6,10',
                   help='horizons in steps (dt=0.5 s), comma separated')
    p.add_argument('--min_len', type=int, default=8, help='shortest usable track, in frames')
    p.add_argument('--min_speed', type=float, default=1.0,
                   help='m/s; drops parked vehicles, which cannot flicker')
    p.add_argument('--node_type', default='VEHICLE')
    p.add_argument('--max_scenes', type=int, default=None)
    p.add_argument('--skip_nuscenes', action='store_true')
    p.add_argument('--skip_unity', action='store_true')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--out_dir', default=DEFAULT_OUT)
    args = p.parse_args()

    horizons = sorted(int(h) for h in args.horizons.split(','))
    device = 'cpu' if args.gpu == 'cpu' else f'cuda:{args.gpu}'

    rows, all_tracks, dt = [], {}, 0.5
    if not args.skip_unity:
        u_rows, dt, u_tracks = unity_tracks(args.bundle, horizons, args.min_len,
                                            args.min_speed, args.node_type)
        rows += u_rows
        all_tracks['unity'] = u_tracks
    if not args.skip_nuscenes:
        n_rows, dt, n_tracks = nuscenes_tracks(args.env, args.model, args.checkpoint, horizons,
                                               args.min_len, args.min_speed, args.num_samples,
                                               device, args.node_type, args.max_scenes)
        rows += n_rows
        all_tracks['nuScenes'] = n_tracks

    if not rows:
        print('no tracks survived the filters')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'flicker_tracks.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows, horizons, dt)
    print_table(summary)
    print(f'\nper-track rows -> {csv_path}')
    if args.plot:
        path = plot_examples(all_tracks, horizons, args.out_dir)
        if path:
            print(f'example reach traces -> {path}')


if __name__ == '__main__':
    main()
