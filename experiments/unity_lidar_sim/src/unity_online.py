"""Online (streaming) Trajectron++ prediction + risk evaluation on Unity-sim tracks.

This is the only driver: BatchedOnlineTrajectron keeps one recurrent state per agent and
advances it by a single new observation per timestep, with all agents of a type in the batch
dimension. Agents are discovered and aged out as they enter and leave the scene, and each
timestep is scored for risk the moment it is predicted.

Risk is computed here and nowhere else. Each timestep's result is written into the frame
record, so it reaches disk alongside the predictions it describes, and every figure -- the
ones below and the ones src/risk_eval/cli.py makes later -- is built by reading that file
back rather than by rescoring.

Per timestep the loop is: observe -> predict -> score. Nothing downstream of the model reads
ahead of `t`, apart from the two clearly-marked evaluation artefacts (each agent's logged
future, and -- with the default `ego_path_mode: logged` -- the ego's logged future used as
the risk reference path). Set `ego_path_mode: projected` in the config for a fully causal
run that dead-reckons the ego instead.

Results accumulate in memory and are written once the simulation finishes, so disk I/O never
sits inside the timed prediction loop.

Scene, model, and evaluation settings live in a YAML config (see configs/config.yaml);
the command line only takes the knobs that change run-to-run.

Usage (from experiments/unity_lidar_sim/):
    python src/unity_online.py --config configs/sweep_s1_rail_online.yaml --gpu 0
    python src/unity_online.py --config configs/sweep_s1_rail_online.yaml --risk_viz
See `python src/unity_online.py -h`. To re-render a stored bundle without re-running the
model, use src/visualization/visualize.py; to re-score one, use src/risk_eval/cli.py.
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from scene_conditioning.unity_scene import load_config, prepare_scene, resolve_device, dt
from online_engine import OnlineEngine
from risk_eval.proximity import evaluate_frame, record_frame, load_recorded, summarize, write_csv
from risk_eval.render import render as render_risk
from risk_eval.timeseries import plot_entry_counts, write_counts_csv
from bundle import save_bundle, load_bundle
from visualization.traj_viz import render_bundle
from visualization.colors import palette_for


# Columns of the per-timestep run log: one row per predicted timestep.
LOG_FIELDS = ['t', 't_sec', 'n_agents', 'n_edges', 'runtime_s', 'mean_risk', 'max_risk']


def write_log_csv(rows, path):
    """Write the per-timestep run log (see LOG_FIELDS)."""
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote per-timestep run log -> {path}')


def run(scene, engine, cfg, radius, ego_path_mode):
    """The online loop: for each timestep, observe -> predict -> score.

    Everything is accumulated in memory and returned; the caller writes it out once the
    simulation is over, so no disk I/O happens inside the loop being timed.

    Risk is scored here and only here. Each frame's result is written back into the frame
    record (`record_frame`), so it travels to disk with the predictions and the visualizers
    never rescore anything.

    Returns (frames, rows, log_rows): the bundle frames with their risk attached, the
    per-(timestep, agent) risk rows, and the run log.
    """
    frames, rows, log_rows = [], [], []
    off = np.array([scene.x_min, scene.y_min])

    for t in range(engine.first_timestep, scene.n_timesteps):

        # prediction step
        rec = engine.step(t)
        if rec is None:
            continue

        ego = scene.ego_pose(t)
        rec['ego'] = None if ego is None else [np.asarray(ego[0], dtype=float).tolist(),
                                               float(ego[1])]
        rec['ego_path'] = scene.ego_path(t, cfg['ph'], mode=ego_path_mode)

        frame_rows, entry = evaluate_frame(rec, off, radius=radius)
        if entry is not None:
            rows.extend(frame_rows)
            record_frame(rec, frame_rows, entry)      # risk goes into the bundle
        frames.append(rec)

        n_agents, n_edges = engine.num_tracked()
        risks = [r['risk'] for r in frame_rows]
        log_rows.append({'t': t, 't_sec': round(t * dt, 3), 'n_agents': n_agents,
                         'n_edges': n_edges, 'runtime_s': round(rec['runtime_s'], 4),
                         'mean_risk': round(float(np.mean(risks)), 6) if risks else '',
                         'max_risk': round(float(np.max(risks)), 6) if risks else ''})
        print(f'  t={t:>4}  {n_agents:>2} agents  {n_edges:>3} edges  '
              f'{rec["runtime_s"]:.3f}s ({1.0 / max(rec["runtime_s"], 1e-9):.1f} Hz)  '
              f'peak risk {max(risks, default=0.0):.2f}', end='\r', flush=True)
    print()
    return frames, rows, log_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/config.yaml',
                   help='YAML config with scene/model/render/risk settings (see configs/config.yaml)')
    p.add_argument('--gpu', type=int, default=-1, help='CUDA device index to run on (e.g. --gpu 2); -1 = CPU')
    p.add_argument('--style', default='samples', choices=['samples', 'gaussian', 'both'],
                   help='distribution render style: sample fan, Gaussian blobs, or both')
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'], help='output video format(s)')
    p.add_argument('--risk_viz', action='store_true',
                   help='also render the proximity-risk overlay video (ego disc + entering samples)')
    args = p.parse_args()
    cfg = load_config(args.config)

    scene = prepare_scene(cfg)
    out_dir = scene.out_dir
    pred_file = cfg['pred_file'] or os.path.join(out_dir, 'predictions_online.pkl')
    risk_file = cfg['risk_file'] or os.path.join(out_dir, 'risk_online.csv')
    log_file = os.path.join(out_dir, 'online_log.csv')

    ego_path_mode = cfg['ego_path_mode']
    if not scene.has_ego:
        print('  WARNING: no ego poses -- risk cannot be evaluated (nothing to centre the disc on)')
    else:
        print(f"  risk: {cfg['risk_radius']:g} m disc centred on the ego's pose at each "
              f"timestep (causal)")
        print(f"  ego path stored for context/viz: {ego_path_mode}"
              + ("  (the ego's LOGGED future -- not causally available at t, and not an "
                 "input to the risk metric)" if ego_path_mode == 'logged' else
                 '  (constant-velocity projection)'))

    device = resolve_device(args.gpu)
    print(f'Loading model from {cfg["model_dir"]} (ts={cfg["model_ts"]}) on {device}...')
    # The style picks which decoder passes run (see online_engine.DECODE_PASSES). 'gaussian'
    # renders only the analytic GMM, but the proximity risk is defined on samples, so the
    # sampling pass is only skippable when risk is off the table anyway.
    engine = OnlineEngine(scene, cfg['model_dir'], cfg['model_ts'], device,
                          ph=cfg['ph'], num_samples=cfg['num_samples'],
                          warmup_timesteps=cfg['warmup_timesteps'],
                          min_history_timesteps=cfg['min_history_timesteps'],
                          style=args.style, force_samples=scene.has_ego)

    print(f'Streaming t={engine.first_timestep}..{scene.n_timesteps - 1} '
          f'(ph={cfg["ph"]}, {cfg["num_samples"]} samples/agent, '
          f'risk radius={cfg["risk_radius"]:g} m)...')
    frames, rows, log_rows = run(scene, engine, cfg,
                                 radius=cfg['risk_radius'],
                                 ego_path_mode=ego_path_mode)

    # ------- the simulation is over; now persist everything -------
    meta = scene.bundle_meta(online=True, gif_prefix='distribution_online',
                             warmup_timesteps=cfg['warmup_timesteps'],
                             min_history_timesteps=cfg['min_history_timesteps'],
                             ego_path_mode=ego_path_mode,
                             risk_radius=cfg['risk_radius'],
                             # the dynamics config the controls in `ctrl_*` were emitted
                             # under, limits included, so consumers need not load the model
                             dynamic=engine.hyperparams['dynamic'])
    save_bundle(pred_file, meta, frames)
    print(f'Saved {len(frames)} prediction frames -> {pred_file}')
    write_csv(rows, risk_file)
    write_log_csv(log_rows, log_file)

    runtimes = np.array([r['runtime_s'] for r in frames]) if frames else np.zeros(0)
    if runtimes.size:
        print(f'  inference: {runtimes.mean():.3f}s/timestep mean, {runtimes.max():.3f}s max '
              f'({1.0 / runtimes.mean():.1f} Hz mean)')
    summarize(rows, radius=cfg['risk_radius'])

    # ------- visualization, built from what was just written to disk -------
    # Deliberately re-read rather than reusing the in-memory objects: the figures are then
    # produced by exactly the path src/risk_eval/cli.py takes on a stored run, so a plot made
    # here and one made later from the same file cannot disagree.
    bundle = load_bundle(pred_file)
    if not cfg['no_viz']:
        render_bundle(bundle, out_dir, ego_frame=cfg['ego_frame'], fps=cfg['fps'],
                      zoom=cfg['zoom'], fmt=args.format, style=args.style,
                      workers=cfg['workers'])
    if args.risk_viz:
        disk_rows, per_frame = load_recorded(bundle)
        if per_frame:
            # One identity map for the video and the time series, so an agent is the same
            # colour in both.
            agent_colors = palette_for(bundle['frames'])
            render_risk(bundle, per_frame, out_dir, fps=cfg['fps'], fmt=args.format,
                        ego_frame=cfg['ego_frame'], zoom=cfg['zoom'],
                        radius=cfg['risk_radius'], agent_colors=agent_colors)
            plot_entry_counts(disk_rows, os.path.join(out_dir, 'risk_timeseries.png'),
                              cfg['risk_radius'], dt=dt, agent_colors=agent_colors)
            write_counts_csv(disk_rows, os.path.join(out_dir, 'risk_timeseries.csv'))


if __name__ == '__main__':
    main()
