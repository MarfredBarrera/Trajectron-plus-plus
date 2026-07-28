"""CLI: score a stored prediction bundle for crossing risk (and optionally render it).

    python src/risk_eval/cli.py --pred_file unity_out/sweep_s1_rail/predictions.pkl
    python src/risk_eval/cli.py --pred_file unity_out/sweep_s1_rail/predictions.pkl --viz

Run from experiments/unity_lidar_sim/ (paths in the args are relative to that). The online
driver scores each timestep as it predicts it and needs none of this.
"""
import os
import sys

# Runnable directly, which puts *this file's* directory on sys.path rather than src/; add the
# source root so the sibling packages resolve either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from bundle import load_bundle
from risk_eval.crossing import evaluate, summarize, write_csv
from risk_eval.render import render


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred_file', required=True, help='prediction bundle (.pkl) with samples + ego_path')
    p.add_argument('--out_dir', default=None, help='where to write CSV / frames (default: next to pred_file)')
    p.add_argument('--viz', action='store_true', help='also render the crossing-risk video')
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'],
                   help='video output format(s) for the risk visualization')
    p.add_argument('--ego_frame', action='store_true',
                   help='render in the ego frame (ego at origin, heading up); outputs risk_ego.*')
    p.add_argument('--zoom', type=float, default=None,
                   help='ego-frame half-window in metres (default: bundle zoom)')
    p.add_argument('--fps', type=float, default=2.0, help='video playback frame rate')
    p.add_argument('--time_window', type=int, default=1,
                   help='temporal slack (in horizon steps) for a crossing to count: agent and '
                        'ego must reach the crossing within this many steps of each other. '
                        '0 = strictly simultaneous; -1 = ignore timing (pure geometric crossing)')
    args = p.parse_args()

    bundle = load_bundle(args.pred_file)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pred_file))
    os.makedirs(out_dir, exist_ok=True)
    print(f'Loaded {len(bundle["frames"])} frames from {args.pred_file}  '
          f'(scene {bundle["meta"].get("scene")})')

    print(f'  time_window = {args.time_window} step(s)'
          + ('  (strictly simultaneous)' if args.time_window == 0
             else '  (timing ignored, pure geometric)' if args.time_window < 0
             else f'  (~±{args.time_window * bundle["meta"].get("dt", 0.5):.1f}s slack)'))
    rows, per_frame = evaluate(bundle, time_window=args.time_window)
    summarize(rows)
    if rows:
        write_csv(rows, os.path.join(out_dir, 'risk_crossings.csv'))
    if args.viz:
        render(bundle, per_frame, out_dir, fps=args.fps, fmt=args.format,
               ego_frame=args.ego_frame, zoom=args.zoom)


if __name__ == '__main__':
    main()
