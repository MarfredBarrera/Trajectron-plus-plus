"""
Rendering + on-disk bundle IO for the Trajectron++ trajectory-distribution viz.

This module has NO torch / model dependencies, so `visualize.py` can render stored
predictions without a GPU or loading the model. The prediction scripts
(`unity_predict.py`, `nuscenes_predict.py`) run the model, extract per-timestep
trajectory arrays, and `save_bundle(...)` them; the visualizer `load_bundle(...)`s and
renders.

Bundle format (pickle):
    {'meta':  {source, scene, dt, ph, num_samples, x_min, y_min, xlim, ylim,
               zoom, gif_prefix},
     'frames': [ {'t': int,
                  'ego': [[ex, ey], yaw] or None,      # world coords; enables ego view
                  'nodes': [ {'id', 'type',
                              'history' (H,2), 'future' (F,2),
                              'samples' (S,ph,2), 'ml' (ph,2)}, ... ]}, ... ]}

All node arrays are in SCENE-LOCAL coords; meta['x_min'/'y_min'] shift them to world.
Whether a frame is drawn in world or ego view is a *render-time* choice -- the ego pose
is always stored, so the visualizer can toggle it without re-running the model.
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from PIL import Image

# distinct colors cycled per agent
AGENT_COLORS = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0', '#E8A33D',
                '#7B68EE', '#2ECC71', '#E74C3C', '#1ABC9C', '#F39C12', '#9B59B6']


# --------------------------------------------------------------------------- #
# Bundle IO                                                                     #
# --------------------------------------------------------------------------- #
def save_bundle(path, meta, frames):
    with open(path, 'wb') as f:
        pickle.dump({'meta': meta, 'frames': frames}, f)


def load_bundle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Rendering                                                                     #
# --------------------------------------------------------------------------- #
def ego_transform(ego_pos, ego_yaw):
    """World (map) coords -> ego frame: ego at origin, heading pointing +y (up)."""
    a = np.pi / 2.0 - ego_yaw
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    def f(world_pts):
        return (np.atleast_2d(world_pts) - np.asarray(ego_pos)) @ R.T
    return f


def _color_for(node_id, order):
    if node_id not in order:
        order[node_id] = len(order)
    return AGENT_COLORS[order[node_id] % len(AGENT_COLORS)]


def _draw_sample_fan(ax, hist, s, c):
    """Each sample as a thin translucent trajectory from the current position through
    the horizon (line density = probability mass)."""
    cur = np.broadcast_to(hist[-1], (s.shape[0], 1, 2))
    sample_lines = np.concatenate([cur, s], axis=1)         # (num_samples, ph+1, 2)
    ax.add_collection(LineCollection(sample_lines, colors=c, linewidths=0.5,
                                     alpha=0.06, zorder=400))


def _draw_gaussian_blobs(ax, hist, s, c, n_stds=(1.0, 2.0)):
    """Per-horizon-step Gaussian blob: at each future step, the sample cloud's mean and
    covariance -> a shaded covariance ellipse (this is the GMM's moment-matched mean/cov
    in position space). `s` is already in plot coords: (num_samples, ph, 2)."""
    if s.shape[0] < 3:
        return
    means = s.mean(axis=0)                                  # (ph, 2)
    for k in range(s.shape[1]):
        pts = s[:, k, :]
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-9, None)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))   # largest-eigenvalue axis
        for ns in n_stds:
            ax.add_patch(Ellipse(means[k], 2 * ns * np.sqrt(vals[1]), 2 * ns * np.sqrt(vals[0]),
                                 angle=angle, facecolor=c, edgecolor='none',
                                 alpha=0.10, zorder=380))
    # mean trajectory through the blobs
    mp = np.vstack([hist[-1], means])
    ax.plot(mp[:, 0], mp[:, 1], '-', color=c, lw=0.9, alpha=0.6, zorder=545)


def draw_frame(ax, record, off, transform, order, style='samples'):
    """Draw one stored frame record's nodes. `style`: 'samples' (trajectory fan),
    'gaussian' (per-step covariance blobs), or 'both'."""
    def to_plot(raw):                       # scene-local -> world -> plot coords
        raw = np.asarray(raw, dtype=float)
        if raw.size == 0:
            return raw
        return transform(raw.reshape(-1, 2) + off).reshape(raw.shape)

    for nd in record['nodes']:
        c = _color_for(nd['id'], order)
        hist = to_plot(nd['history'])
        fut = to_plot(nd['future'])
        s = to_plot(nd['samples'])          # (num_samples, ph, 2)
        mm = to_plot(nd['ml'])              # (ph, 2)

        # history
        ax.plot(hist[:, 0], hist[:, 1], '-', color=c, lw=1.5, alpha=0.85, zorder=300)
        # predicted distribution
        if style in ('samples', 'both'):
            _draw_sample_fan(ax, hist, s, c)
        if style in ('gaussian', 'both'):
            _draw_gaussian_blobs(ax, hist, s, c)
        # most-likely path
        ax.plot(*np.vstack([hist[-1], mm]).T, '-', color=c, lw=1.6, zorder=550)
        # GT future
        if len(fut):
            ax.plot(*np.vstack([hist[-1], fut]).T, '--', color='w', lw=1.8, zorder=600,
                    path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
        # current position
        ax.scatter([hist[-1, 0]], [hist[-1, 1]], s=28, color=c, edgecolors='k',
                   linewidths=0.6, zorder=650)


def render_frame_to_file(record, meta, out_path, order, ego_frame=False, zoom=None, style='samples'):
    """Render one stored frame record to an image. Returns True."""
    ego = record.get('ego')
    use_ego = ego_frame and ego is not None
    if zoom is None:
        zoom = meta.get('zoom', 60.0)
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    transform = ego_transform(np.array(ego[0]), ego[1]) if use_ego else (lambda p: np.atleast_2d(p))
    t, dtv = record['t'], meta.get('dt', 0.5)

    fig, ax = plt.subplots(figsize=(12, 12) if use_ego else (11, 13))
    ax.set_facecolor("#FFFFFF")

    draw_frame(ax, record, off, transform, order, style=style)

    handles = [Line2D([], [], color='w', ls='--', lw=1.8, label='Ground truth',
                      path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]),
               Line2D([], [], color='#888', lw=1.6, label='Most likely')]
    if style in ('samples', 'both'):
        handles.append(Line2D([], [], color='#888', lw=1.0, alpha=0.7, label='Predicted samples'))
    if style in ('gaussian', 'both'):
        handles.append(Patch(facecolor='#888', edgecolor='none', alpha=0.5, label='Predicted mean ±1,2σ'))
    if use_ego:
        handles.append(Line2D([], [], marker='^', color='#FF0000', ls='', ms=12, mec='k', label='Ego'))

    if use_ego:
        ax.scatter([0], [0], marker='^', s=260, color="#FF0000", edgecolors='k',
                   linewidths=1.0, zorder=800)
        for r in range(10, int(zoom) + 1, 10):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color='#333', lw=0.6, zorder=100))
        ax.axhline(0, color='#333', lw=0.5, zorder=90)
        ax.axvline(0, color='#333', lw=0.5, zorder=90)
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
        ax.set_xlabel('ego x (right) [m]'); ax.set_ylabel('ego y (forward) [m]')
        title = f'Trajectron++ distribution (ego frame)  |  t = {t}  ({t * dtv:.1f}s)'
    else:
        ax.set_xlim(meta['xlim']); ax.set_ylim(meta['ylim'])
        ax.set_xlabel('map x [m]'); ax.set_ylabel('map y [m]')
        title = f'Trajectron++ predicted distribution  |  t = {t}  ({t * dtv:.1f}s)'

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.legend(handles=handles, loc='upper right', fontsize=11, frameon=True, facecolor='white')
    fig.savefig(out_path, dpi=90, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


def assemble_gif(paths, out_path, fps):
    imgs = [Image.open(p) for p in paths]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=1000.0 / fps, loop=0)
    return out_path


def assemble_mp4(paths, out_path, fps):
    """Encode PNG frames to an .mp4 with OpenCV (mp4v). Frames are normalized to the
    first frame's size (rounded to even dims, which H.264-style codecs require)."""
    import cv2
    first = cv2.imread(paths[0])
    h, w = first.shape[:2]
    w, h = w - (w % 2), h - (h % 2)   # even dimensions
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError('cv2.VideoWriter failed to open (mp4v codec unavailable?)')
    for p in paths:
        img = cv2.imread(p)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()
    return out_path


def _assemble(paths, out_dir, prefix, ego_frame, fps, fmt):
    """Write GIF and/or MP4 from rendered PNG paths, per `fmt` in {gif, mp4, both}."""
    base = os.path.join(out_dir, f"{prefix}{'_ego' if ego_frame else ''}")
    outputs = []
    if fmt in ('gif', 'both'):
        outputs.append(assemble_gif(paths, base + '.gif', fps))
    if fmt in ('mp4', 'both'):
        outputs.append(assemble_mp4(paths, base + '.mp4', fps))
    for o in outputs:
        print(f'Wrote {o}  ({len(paths)} frames @ {fps} fps)')
    return outputs


def render_bundle(bundle, out_dir, ego_frame=False, fps=5.0, zoom=None, single=False,
                  fmt='gif', style='samples'):
    """Render every frame in a bundle to PNGs (in a mode-specific subdir) + a GIF and/or
    MP4 (`fmt` in {gif, mp4, both}). `style` in {samples, gaussian, both} picks how the
    predicted distribution is drawn. If `single`, write one PNG into out_dir, no video."""
    meta, frames = bundle['meta'], bundle['frames']
    if not frames:
        print('No frames in bundle.')
        return None
    os.makedirs(out_dir, exist_ok=True)
    order = {}

    if single:
        rec = frames[0]
        mode = 'ego' if ego_frame else 'world'
        out = os.path.join(out_dir, f"frame_{rec['t']:04d}_{mode}_{style}.png")
        render_frame_to_file(rec, meta, out, order, ego_frame=ego_frame, zoom=zoom, style=style)
        print(f'Rendered -> {out}')
        return out

    frame_dir = os.path.join(out_dir, 'frames_ego' if ego_frame else 'frames_world')
    os.makedirs(frame_dir, exist_ok=True)
    paths = []
    for i, rec in enumerate(frames):
        out = os.path.join(frame_dir, f"frame_{rec['t']:04d}.png")
        render_frame_to_file(rec, meta, out, order, ego_frame=ego_frame, zoom=zoom, style=style)
        paths.append(out)
        print(f'  render [{i + 1}/{len(frames)}] t={rec["t"]}', end='\r', flush=True)
    print()

    prefix = meta.get('gif_prefix', 'distribution')
    return _assemble(paths, out_dir, prefix, ego_frame, fps, fmt)
