"""CLI: build the risk visualizations for a stored run.

Risk is scored **online**, inside unity_online.py, at the moment each timestep is predicted,
and recorded into the bundle along with the predictions. This tool reads that recorded
result back and draws it -- it does not re-derive risk, so a figure made here always shows
what the run actually decided at the time.

    python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl
    python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl --viz
    python src/risk_eval/cli.py --pred_file unity_out/intersection_probe/predictions_online.pkl \
        --junction_colors            # time-series agents in the investigation's own hues

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

import numpy as np

from bundle import load_bundle
from risk_eval.proximity import (DEFAULT_RADIUS, evaluate, has_recorded_risk, load_recorded,
                                 summarize, write_csv)
from risk_eval.render import render
from risk_eval.timeseries import plot_entry_counts, write_counts_csv
from visualization.colors import DEFAULT_CENTER, junction_palette, palette_for


def agent_palette(bundle, args):
    """The identity map every figure this run draws is coloured under.

    Two studies draw the same vehicles, and a figure is only readable next to the other if the
    agent that is amber in one is amber in the other -- so which scheme applies is the
    caller's choice, made once here.
    """
    if not args.junction_colors:
        return palette_for(bundle['frames'])
    meta = bundle['meta']
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    center = [float(v) for v in args.junction_center.split(',')]
    return junction_palette(bundle['frames'], off, center)


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
    p.add_argument('--junction_colors', action='store_true',
                   help='colour agents -- in both the time series and the overlay video -- '
                        'the way the intersection investigations do (hue by roster position '
                        'at the junction) instead of by order of appearance')
    p.add_argument('--junction_center', default=','.join(str(c) for c in DEFAULT_CENTER),
                   help='junction centre "x,y" in map coords, for --junction_colors')
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
    # One identity map for everything this run draws: the video and the time series always
    # name an agent with the same colour, whichever scheme is in force.
    agent_colors = agent_palette(bundle, args)
    if rows:
        write_csv(rows, os.path.join(out_dir, 'risk_proximity.csv'))
        if not args.no_timeplot:
            plot_entry_counts(rows, os.path.join(out_dir, 'risk_timeseries.png'), radius,
                              dt=bundle['meta'].get('dt', 0.5), agent_colors=agent_colors)
            write_counts_csv(rows, os.path.join(out_dir, 'risk_timeseries.csv'))
    if args.viz:
        render(bundle, per_frame, out_dir, fps=args.fps, fmt=args.format,
               ego_frame=args.ego_frame, zoom=args.zoom, radius=radius,
               agent_colors=agent_colors)


if __name__ == '__main__':
    main()
