"""CLI: build the risk visualizations for a stored run.

Risk is scored **online**, inside unity_online.py, at the moment each timestep is predicted,
and recorded into the bundle along with the predictions. This tool reads that recorded
result back and draws it -- it does not re-derive risk, so a figure made here always shows
what the run actually decided at the time.

    python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl
    python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl --viz

`--rescore` is the deliberate exception: it recomputes risk from the stored samples at a
different radius, for sweeping the threshold without paying for inference again. It is
also the automatic fallback for a bundle written before risk was recorded.

Run from experiments/unity_lidar_sim/ (paths in the args are relative to that).
"""
import os
import sys

# Runnable directly, which puts *this file's* directory on sys.path rather than src/; add the
# source root so the sibling packages resolve either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from bundle import load_bundle
from risk_eval.proximity import (DEFAULT_RADIUS, evaluate, has_recorded_risk, load_recorded,
                                 summarize, write_csv)
from risk_eval.render import render
from risk_eval.timeseries import plot_entry_counts, write_counts_csv


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred_file', required=True, help='prediction bundle (.pkl) from a scored run')
    p.add_argument('--out_dir', default=None, help='where to write CSV / frames (default: next to pred_file)')
    p.add_argument('--viz', action='store_true', help='also render the proximity-risk overlay video')
    p.add_argument('--no_timeplot', action='store_true',
                   help='skip the entering-trajectories-over-time figure')
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'],
                   help='video output format(s) for the risk visualization')
    p.add_argument('--ego_frame', action='store_true',
                   help='render in the ego frame (ego at origin, heading up); outputs risk_ego.*')
    p.add_argument('--zoom', type=float, default=None,
                   help='ego-frame half-window in metres (default: bundle zoom)')
    p.add_argument('--fps', type=float, default=2.0, help='video playback frame rate')
    p.add_argument('--rescore', action='store_true',
                   help='recompute risk from the stored samples instead of reading the run\'s '
                        'own result -- use with --radius to sweep the threshold')
    p.add_argument('--radius', type=float, default=None,
                   help='ego disc radius in metres; implies --rescore when it differs from '
                        'the radius the run was scored with')
    args = p.parse_args()

    bundle = load_bundle(args.pred_file)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pred_file))
    os.makedirs(out_dir, exist_ok=True)
    print(f'Loaded {len(bundle["frames"])} frames from {args.pred_file}  '
          f'(scene {bundle["meta"].get("scene")})')

    scored_radius = float(bundle['meta'].get('risk_radius', DEFAULT_RADIUS))
    radius = scored_radius if args.radius is None else args.radius
    recorded = has_recorded_risk(bundle)
    rescore = args.rescore or radius != scored_radius or not recorded

    if not rescore:
        print(f'  reading the risk this run recorded online (R = {radius:g} m)')
        rows, per_frame = load_recorded(bundle)
    else:
        why = ('no recorded risk in this bundle -- it predates online scoring'
               if not recorded else f'requested, vs the {scored_radius:g} m the run used')
        print(f'  RESCORING at R = {radius:g} m ({why})')
        rows, per_frame = evaluate(bundle, radius=radius)

    summarize(rows, radius=radius)
    if rows:
        write_csv(rows, os.path.join(out_dir, 'risk_proximity.csv'))
        if not args.no_timeplot:
            plot_entry_counts(rows, os.path.join(out_dir, 'risk_timeseries.png'), radius,
                              dt=bundle['meta'].get('dt', 0.5))
            write_counts_csv(rows, os.path.join(out_dir, 'risk_timeseries.csv'))
    if args.viz:
        render(bundle, per_frame, out_dir, fps=args.fps, fmt=args.format,
               ego_frame=args.ego_frame, zoom=args.zoom, radius=radius)


if __name__ == '__main__':
    main()
