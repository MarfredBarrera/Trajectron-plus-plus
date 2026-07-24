"""
Render a stored Trajectron++ prediction bundle to frames + a GIF.

This is the *visualization-only* entry point: it loads a bundle written by
unity_predict.py / nuscenes_predict.py and renders it. It imports only traj_viz (numpy /
matplotlib / PIL) -- no torch, no model, no GPU -- so you can re-render, switch to the
ego frame, change fps/zoom, etc. without ever touching the model.

Usage (from experiments/unity_lidar_sim/):
    python visualize.py --pred_file unity_out/sweep_s1_rail/predictions.pkl
    python visualize.py --pred_file unity_out/sweep_s1_rail/predictions.pkl --ego_frame --zoom 60 --fps 2
Or, for a nuScenes bundle (from experiments/nuScenes/):
    python ../unity_lidar_sim/visualize.py --pred_file nuscenes_out/predictions.pkl --out_dir nuscenes_out
"""
import os
import argparse

from traj_viz import load_bundle, render_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred_file', required=True, help='prediction bundle (.pkl) to render')
    p.add_argument('--out_dir', default=None,
                   help='output directory (default: the bundle file\'s directory)')
    p.add_argument('--ego_frame', action='store_true', help='render in the ego frame (if ego stored)')
    p.add_argument('--zoom', type=float, default=None, help='ego-frame half-window in metres')
    p.add_argument('--fps', type=float, default=2.0, help='video playback frame rate')
    p.add_argument('--format', default='gif', choices=['gif', 'mp4', 'both'],
                   help='output video format(s)')
    p.add_argument('--style', default='samples', choices=['samples', 'gaussian', 'both'],
                   help='how to draw the distribution: sample fan, Gaussian blobs, or both')
    p.add_argument('--single', action='store_true', help='render only the first frame (debug, no video)')
    p.add_argument('--workers', type=int, default=None,
                   help='parallel render processes (default: auto; rendering is CPU/matplotlib, not GPU)')
    args = p.parse_args()

    bundle = load_bundle(args.pred_file)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pred_file))
    print(f'Loaded {len(bundle["frames"])} frames from {args.pred_file}  '
          f'(scene {bundle["meta"].get("scene")}, source {bundle["meta"].get("source")})')
    render_bundle(bundle, out_dir, ego_frame=args.ego_frame, fps=args.fps,
                  zoom=args.zoom, single=args.single, fmt=args.format, style=args.style,
                  workers=args.workers)


if __name__ == '__main__':
    main()
