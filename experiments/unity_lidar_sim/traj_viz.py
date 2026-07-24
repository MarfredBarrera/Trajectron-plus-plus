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
                              'samples' (S,ph,2), 'ml' (ph,2),
                              'dist_mus' (ph,K,2), 'dist_covs' (ph,K,2,2), 'dist_pis' (ph,K)},
                             ... ]}, ... ]}

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
from matplotlib.transforms import Affine2D
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
    f.R = R  # exposed so covariance matrices can be rotated the same way as points
    return f


def _identity_transform(p):
    return np.atleast_2d(p)


_identity_transform.R = np.eye(2)


def _color_for(node_id, order):
    if node_id not in order:
        order[node_id] = len(order)
    return AGENT_COLORS[order[node_id] % len(AGENT_COLORS)]


def _draw_sample_fan(ax, hist, s, c):
    """Each sample as a thin translucent trajectory from the current position through
    the horizon (line density = probability mass)."""
    if s.size == 0:   # bundle produced with --style gaussian: no samples were stored
        return
    cur = np.broadcast_to(hist[-1], (s.shape[0], 1, 2))
    sample_lines = np.concatenate([cur, s], axis=1)         # (num_samples, ph+1, 2)
    ax.add_collection(LineCollection(sample_lines, colors=c, linewidths=0.5,
                                     alpha=0.06, zorder=400))


def _draw_gaussian_blobs(ax, hist, dist_mus, dist_covs, dist_pis, transform, c,
                         n_stds=(1.0, 2.0), pi_threshold=0.05):
    """Per-horizon-step Gaussian blobs drawn straight from the decoder/GMM's own analytic
    parameters (mean + covariance propagated through the dynamics model), not estimated
    from sample statistics. `dist_mus` is already in plot coords: (ph, K, 2); `dist_covs`
    is raw scene-local covariance (ph, K, 2, 2) -- rotated (not translated) into plot
    coords using `transform.R`; `dist_pis` (ph, K) are the mixture weights (components
    below `pi_threshold` are skipped as visual clutter, not part of the actual math)."""
    R = transform.R
    ph, K = dist_pis.shape
    mean_path = [hist[-1]]
    for t in range(ph):
        # pi-weighted mean over *all* components, independent of the drawing threshold
        mean_path.append(np.sum(dist_pis[t][:, None] * dist_mus[t], axis=0))
        for k in range(K):
            if dist_pis[t, k] < pi_threshold:
                continue
            cov = R @ dist_covs[t, k] @ R.T
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-9, None)
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))   # largest-eigenvalue axis
            for ns in n_stds:
                ax.add_patch(Ellipse(dist_mus[t, k], 2 * ns * np.sqrt(vals[1]), 2 * ns * np.sqrt(vals[0]),
                                     angle=angle, facecolor=c, edgecolor='none',
                                     alpha=0.10, zorder=380))
    mp = np.vstack(mean_path)
    ax.plot(mp[:, 0], mp[:, 1], '-', color=c, lw=0.9, alpha=0.6, zorder=545)


def _draw_map_background(ax, map_rgba, bounds, ego=None, crop_radius=None):
    """Shade the non-drivable area under the trajectories (see unity_maps/handoff.md);
    drivable pixels are fully transparent so the ribbon itself stays visually clean --
    it's a soft prior for the model, not a hard mask, and the viz treats it the same way.
    World view: axis-aligned imshow at the raster's native extent. Ego view: crop around
    the ego position first (the raster covers far more area than one zoom window) then
    rotate the crop the same way `ego_transform` rotates points, via an Affine2D transform
    on the image artist."""
    if map_rgba is None or bounds is None:
        return
    x0, x1, y0, y1 = bounds
    h, w = map_rgba.shape[:2]
    if ego is None:
        ax.imshow(map_rgba, extent=(x0, x1, y0, y1), origin='upper', zorder=0)
        return

    ego_pos, ego_yaw = ego
    r = crop_radius or 200.0
    px_x, px_y = w / (x1 - x0), h / (y1 - y0)
    col0 = int(np.clip((ego_pos[0] - r - x0) * px_x, 0, w))
    col1 = int(np.clip((ego_pos[0] + r - x0) * px_x, 0, w))
    row0 = int(np.clip((y1 - (ego_pos[1] + r)) * px_y, 0, h))
    row1 = int(np.clip((y1 - (ego_pos[1] - r)) * px_y, 0, h))
    if col1 <= col0 or row1 <= row0:
        return   # ego is entirely outside the mapped area
    crop = map_rgba[row0:row1, col0:col1]
    cx0, cx1 = x0 + col0 / px_x, x0 + col1 / px_x
    cy0, cy1 = y1 - row1 / px_y, y1 - row0 / px_y

    a = np.pi / 2.0 - ego_yaw     # same rotation ego_transform applies to points
    tr = Affine2D().translate(-ego_pos[0], -ego_pos[1]).rotate(a) + ax.transData
    ax.imshow(crop, extent=(cx0, cx1, cy0, cy1), origin='upper', zorder=0, transform=tr)


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
            dm = to_plot(nd['dist_mus'])    # (ph, K, 2); covariance is rotated, not offset
            _draw_gaussian_blobs(ax, hist, dm, nd['dist_covs'], nd['dist_pis'], transform, c)
        # most-likely path
        ax.plot(*np.vstack([hist[-1], mm]).T, '-', color=c, lw=1.6, zorder=550)
        # GT future
        if len(fut):
            ax.plot(*np.vstack([hist[-1], fut]).T, '--', color='w', lw=1.8, zorder=600,
                    path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
        # current position
        ax.scatter([hist[-1, 0]], [hist[-1, 1]], s=28, color=c, edgecolors='k',
                   linewidths=0.6, zorder=650)


def render_frame_to_file(record, meta, out_path, order, ego_frame=False, zoom=None, style='samples',
                         map_rgba=None, map_bounds=None):
    """Render one stored frame record to an image. Returns True."""
    ego = record.get('ego')
    use_ego = ego_frame and ego is not None
    if zoom is None:
        zoom = meta.get('zoom', 60.0)
    off = np.array([meta.get('x_min', 0.0), meta.get('y_min', 0.0)])
    transform = ego_transform(np.array(ego[0]), ego[1]) if use_ego else _identity_transform
    t, dtv = record['t'], meta.get('dt', 0.5)

    fig, ax = plt.subplots(figsize=(12, 12) if use_ego else (11, 13))
    ax.set_facecolor("#FFFFFF")

    map_ego = (np.array(ego[0]), ego[1]) if use_ego else None
    _draw_map_background(ax, map_rgba, map_bounds, ego=map_ego, crop_radius=zoom * 1.6 if use_ego else None)

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
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
        ax.set_xlabel('ego x (right) [m]'); ax.set_ylabel('ego y (forward) [m]')
        title = f'Trajectron++ distribution (ego frame)  |  t = {t}  ({t * dtv:.1f}s)'
    else:
        ax.set_xlim(meta['xlim']); ax.set_ylim(meta['ylim'])
        ax.set_xlabel('map x [m]'); ax.set_ylabel('map y [m]')
        title = f'Trajectron++ predicted distribution  |  t = {t}  ({t * dtv:.1f}s)'

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    leg = ax.legend(handles=handles, loc='upper right', fontsize=11, frameon=True, facecolor='white')
    leg.set_zorder(1000)   # keep the legend above every trajectory/blob artist
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


def _build_map_rgba(meta, target_long_px=1600):
    """Build the non-drivable shading overlay once, downsampled so imshow is cheap.

    The native drivable raster is ~15.6 MP but a rendered frame is only ~1000 px wide, so
    imshow resamples the whole thing every frame (~0.5 s each) for no visible gain. We
    downsample to `target_long_px` on the long side (extent/world coords are unchanged --
    only pixel resolution drops), which also shrinks what gets shipped to render workers.
    Alpha is area-averaged (BOX) so thin non-drivable seams stay visible."""
    map_png = meta.get('map_png')
    if not map_png or not os.path.exists(map_png):
        return None, None
    gray = np.asarray(Image.open(map_png).convert('L'))
    h, w = gray.shape
    f = max(1, int(round(max(h, w) / float(target_long_px))))
    if f > 1:
        alpha = np.asarray(Image.fromarray(np.where(gray < 127, 100, 0).astype(np.uint8))
                           .resize((w // f, h // f), Image.BOX))
    else:
        alpha = np.where(gray < 127, 100, 0).astype(np.uint8)
    map_rgba = np.zeros(alpha.shape + (4,), dtype=np.uint8)
    map_rgba[..., :3] = 90
    map_rgba[..., 3] = alpha
    return map_rgba, meta.get('map_bounds')


def _precompute_order(frames):
    """Global node_id -> color index by first appearance (frame, then node order), so every
    frame -- including ones rendered in parallel worker processes -- colors agents the same."""
    order = {}
    for rec in frames:
        for nd in rec['nodes']:
            _color_for(nd['id'], order)
    return order


# per-worker read-only state, set once by the pool initializer to avoid re-pickling the
# map raster / meta / color map for every frame.
_WORKER = {}


def _render_init(meta, map_rgba, map_bounds, order, frame_dir, ego_frame, zoom, style):
    _WORKER.update(meta=meta, map_rgba=map_rgba, map_bounds=map_bounds, order=dict(order),
                   frame_dir=frame_dir, ego_frame=ego_frame, zoom=zoom, style=style)


def _render_one(rec):
    w = _WORKER
    out = os.path.join(w['frame_dir'], f"frame_{rec['t']:04d}.png")
    render_frame_to_file(rec, w['meta'], out, w['order'], ego_frame=w['ego_frame'],
                         zoom=w['zoom'], style=w['style'], map_rgba=w['map_rgba'],
                         map_bounds=w['map_bounds'])
    return out


def render_bundle(bundle, out_dir, ego_frame=False, fps=2.0, zoom=None, single=False,
                  fmt='gif', style='samples', workers=None):
    """Render every frame in a bundle to PNGs (in a mode-specific subdir) + a GIF and/or
    MP4 (`fmt` in {gif, mp4, both}). `style` in {samples, gaussian, both} picks how the
    predicted distribution is drawn. If `single`, write one PNG into out_dir, no video."""
    meta, frames = bundle['meta'], bundle['frames']
    if not frames:
        print('No frames in bundle.')
        return None
    os.makedirs(out_dir, exist_ok=True)

    # global color map + drivable-map background, both built once and reused for every
    # frame (see unity_maps/handoff.md). Only non-drivable pixels get an alpha tint, so the
    # drivable ribbon itself stays clean.
    order = _precompute_order(frames)
    map_rgba, map_bounds = _build_map_rgba(meta)

    if single:
        rec = frames[0]
        mode = 'ego' if ego_frame else 'world'
        out = os.path.join(out_dir, f"frame_{rec['t']:04d}_{mode}_{style}.png")
        render_frame_to_file(rec, meta, out, order, ego_frame=ego_frame, zoom=zoom, style=style,
                             map_rgba=map_rgba, map_bounds=map_bounds)
        print(f'Rendered -> {out}')
        return out

    frame_dir = os.path.join(out_dir, 'frames_ego' if ego_frame else 'frames_world')
    os.makedirs(frame_dir, exist_ok=True)

    # rendering is matplotlib/CPU (no GPU) and each frame is independent, so fan the frames
    # out across processes. `workers` None -> auto (capped so we don't hog a shared box).
    if workers is None:
        workers = min(len(frames), (os.cpu_count() or 1), 32)
    workers = max(1, int(workers))
    paths = [os.path.join(frame_dir, f"frame_{rec['t']:04d}.png") for rec in frames]

    if workers == 1:
        for i, rec in enumerate(frames):
            render_frame_to_file(rec, meta, paths[i], order, ego_frame=ego_frame, zoom=zoom,
                                 style=style, map_rgba=map_rgba, map_bounds=map_bounds)
            print(f'  render [{i + 1}/{len(frames)}] t={rec["t"]}', end='\r', flush=True)
        print()
    else:
        import concurrent.futures as cf
        print(f'  rendering {len(frames)} frames across {workers} processes...')
        with cf.ProcessPoolExecutor(
                max_workers=workers, initializer=_render_init,
                initargs=(meta, map_rgba, map_bounds, order, frame_dir, ego_frame, zoom, style)) as ex:
            done = 0
            for _ in ex.map(_render_one, frames):
                done += 1
                print(f'  render [{done}/{len(frames)}]', end='\r', flush=True)
        print()

    prefix = meta.get('gif_prefix', 'distribution')
    return _assemble(paths, out_dir, prefix, ego_frame, fps, fmt)
