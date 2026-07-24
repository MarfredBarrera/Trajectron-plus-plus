"""
Measure how much the drivable/occupancy map changes Trajectron++ predictions on a
Unity scene, by counting predicted trajectories that cross into NON-drivable
("occupied") area.

Inputs are two prediction bundles produced by unity_predict.py for the SAME scene:
one built with the drivable map, one with a blank map (map_png/json null). Because
both share identical GT inputs and differ only by the map tensor fed to the VEHICLE
encoder, we can pair predictions by (timestep, node id) and isolate the map's effect.

Occupancy is read from the Unity drivable raster (drivable = white 255). Each predicted
(x,y) point in scene-local coords is shifted back to world (rx,ry) via meta.x_min/y_min,
mapped to a pixel via the JSON world->px homography, and classified:
    drivable  -> in-map and white (on the taxiway ribbon)
    occupied  -> in-map and black (apron / off-lane, where a vehicle should not be)
    offmap    -> outside the exported raster (unmapped; also not on a known lane)
"off-drivable" = occupied + offmap = anywhere not on a known drivable lane.

Usage (from experiments/unity_lidar_sim/, trajectron-gpu env not required -- CPU only):
    python occupancy_eval.py \
        --map-pred  unity_out/sweep_s1_rail_mapcmp/pred_map.pkl \
        --nomap-pred unity_out/sweep_s1_rail_mapcmp/pred_nomap.pkl \
        --png unity_maps/Demo_drivable.png --json unity_maps/Demo_drivable.json \
        --out-dir unity_out/sweep_s1_rail_mapcmp
"""
import os
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the Demo raster is ~15.6 MP


def load_bundle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_map(png_path, json_path):
    meta = json.load(open(json_path))
    img = np.asarray(Image.open(png_path).convert('L'))          # (H, W); drivable = 255
    drivable = img > 127
    H = np.array(meta['homography_world_to_px'], dtype=float)     # world (rx,ry,1) -> (col,row)
    return drivable, H, meta


def world_to_px(pts_world, H):
    """pts_world: (...,2) -> integer (col,row) via the 3x3 world->px homography."""
    flat = pts_world.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1))
    hom = np.concatenate([flat, ones], axis=1) @ H.T             # (N,3)
    colrow = hom[:, :2] / hom[:, 2:3]
    return np.round(colrow).astype(np.int64).reshape(pts_world.shape)


def classify(pts_world, drivable, H):
    """Classify each world point. Returns (in_map, is_drivable) boolean arrays of shape pts[...,0].
    is_drivable is only meaningful where in_map is True."""
    Himg, Wimg = drivable.shape
    colrow = world_to_px(pts_world, H)
    col = colrow[..., 0]
    row = colrow[..., 1]
    in_map = (col >= 0) & (col < Wimg) & (row >= 0) & (row < Himg)
    ccol = np.clip(col, 0, Wimg - 1)
    crow = np.clip(row, 0, Himg - 1)
    is_drivable = drivable[crow, ccol] & in_map
    return in_map, is_drivable


def to_world(arr, xmin, ymin):
    a = np.asarray(arr, dtype=np.float64).copy()
    a[..., 0] += xmin
    a[..., 1] += ymin
    return a


def analyse_bundle(bundle, drivable, H, node_type='VEHICLE'):
    """Walk a bundle and tally point/trajectory occupancy for the given node type.

    Returns a dict of aggregate counters plus:
      per_key: {(t, node_id): dict(traj_cross_occ, traj_cross_off, n_traj, ml_cross_occ, ml_cross_off)}
      step_occ / step_off: (ph,) counts of occupied/off-drivable sample points per horizon step
      step_tot: (ph,) total sample points per horizon step
    """
    meta = bundle['meta']
    xmin, ymin = meta['x_min'], meta['y_min']
    ph = meta['ph']

    agg = dict(
        pt_total=0, pt_drivable=0, pt_occupied=0, pt_offmap=0,       # sample points
        traj_total=0, traj_cross_occ=0, traj_cross_off=0,           # sample trajectories
        ml_total=0, ml_cross_occ=0, ml_cross_off=0,                 # most-likely paths
        occ_pts_per_traj_sum=0,                                     # penetration (mean off pts / traj)
        gt_total=0, gt_drivable=0, gt_occupied=0, gt_offmap=0,      # GT history+future (validation)
    )
    per_key = {}
    step_occ = np.zeros(ph, dtype=np.int64)
    step_off = np.zeros(ph, dtype=np.int64)
    step_tot = np.zeros(ph, dtype=np.int64)

    for f in bundle['frames']:
        t = f['t']
        for nd in f['nodes']:
            if nd['type'] != node_type:
                # GT validation still uses only the requested type below
                pass
            if nd['type'] != node_type:
                continue

            # ---- sample trajectories: (S, ph, 2) ----
            s = np.asarray(nd['samples'])
            if s.size:
                sw = to_world(s, xmin, ymin)                        # (S, ph, 2)
                in_map, is_drv = classify(sw, drivable, H)          # (S, ph)
                occ = in_map & ~is_drv                              # in-map black
                off = ~is_drv                                       # occ OR offmap
                agg['pt_total'] += occ.size
                agg['pt_drivable'] += int(is_drv.sum())
                agg['pt_occupied'] += int(occ.sum())
                agg['pt_offmap'] += int((~in_map).sum())
                traj_occ = occ.any(axis=1)                          # (S,) crosses in-map occupied
                traj_off = off.any(axis=1)                          # (S,) crosses any off-drivable
                agg['traj_total'] += s.shape[0]
                agg['traj_cross_occ'] += int(traj_occ.sum())
                agg['traj_cross_off'] += int(traj_off.sum())
                agg['occ_pts_per_traj_sum'] += int(occ.sum())       # summed then / traj_total later
                step_occ += occ.sum(axis=0)
                step_off += off.sum(axis=0)
                step_tot += s.shape[0]
                pk = dict(n_traj=int(s.shape[0]),
                          traj_cross_occ=int(traj_occ.sum()),
                          traj_cross_off=int(traj_off.sum()))
            else:
                pk = dict(n_traj=0, traj_cross_occ=0, traj_cross_off=0)

            # is the agent currently (last observed position) ON the mapped drivable region?
            # the map can only plausibly steer prediction where the agent actually sits on it.
            hist = np.asarray(nd['history'])
            if hist.size:
                cur = to_world(hist[-1:], xmin, ymin)
                cin, cdrv = classify(cur, drivable, H)
                pk['cur_in_map'] = bool(cin.reshape(-1)[0])
                pk['cur_drivable'] = bool(cdrv.reshape(-1)[0])
            else:
                pk['cur_in_map'] = False
                pk['cur_drivable'] = False

            # ---- most-likely path: (ph, 2) ----
            ml = np.asarray(nd['ml'])
            if ml.size:
                mw = to_world(ml, xmin, ymin)
                in_map, is_drv = classify(mw, drivable, H)
                agg['ml_total'] += 1
                mocc = bool((in_map & ~is_drv).any())
                moff = bool((~is_drv).any())
                agg['ml_cross_occ'] += int(mocc)
                agg['ml_cross_off'] += int(moff)
                pk['ml_cross_occ'] = int(mocc)
                pk['ml_cross_off'] = int(moff)

            # ---- GT (history + future) for coordinate/coverage validation ----
            for k in ('history', 'future'):
                a = np.asarray(nd[k])
                if a.size:
                    aw = to_world(a, xmin, ymin)
                    in_map, is_drv = classify(aw, drivable, H)
                    agg['gt_total'] += is_drv.size
                    agg['gt_drivable'] += int(is_drv.sum())
                    agg['gt_occupied'] += int((in_map & ~is_drv).sum())
                    agg['gt_offmap'] += int((~in_map).sum())

            per_key[(t, nd['id'])] = pk

    return agg, per_key, (step_occ, step_off, step_tot)


def two_prop_ztest(k1, n1, k2, n2):
    """Two-sided two-proportion z-test. Returns (z, p). NB: sample trajectories within a
    node-timestep are correlated, so treat n as an upper bound on independent info."""
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float('nan'), float('nan')
    z = (p1 - p2) / se
    from math import erf, sqrt
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, pval


def wilcoxon(diffs):
    """Wilcoxon signed-rank on paired differences (drop zeros). Returns (stat, p_normal_approx, n)."""
    d = np.asarray([x for x in diffs if x != 0], dtype=float)
    n = len(d)
    if n < 1:
        return float('nan'), float('nan'), 0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    W_plus = ranks[d > 0].sum()
    W_minus = ranks[d < 0].sum()
    W = min(W_plus, W_minus)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W, float('nan'), n
    z = (W - mu) / sigma
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return W, p, n


def pct(k, n):
    return 100.0 * k / n if n else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map-pred', required=True)
    ap.add_argument('--nomap-pred', required=True)
    ap.add_argument('--png', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--node-type', default='VEHICLE')
    args = ap.parse_args()

    drivable, H, mapmeta = load_map(args.png, args.json)
    bmap = load_bundle(args.map_pred)
    bno = load_bundle(args.nomap_pred)

    results = {}
    for tag, b in (('map', bmap), ('nomap', bno)):
        agg, per_key, steps = analyse_bundle(b, drivable, H, node_type=args.node_type)
        results[tag] = dict(agg=agg, per_key=per_key, steps=steps)

    am, an = results['map']['agg'], results['nomap']['agg']

    # ---------- report ----------
    lines = []
    def out(s=''):
        lines.append(s)
        print(s)

    out('=' * 78)
    out(f'OCCUPANCY EFFECT OF THE MAP  --  node type: {args.node_type}')
    out(f'scene raster: {os.path.basename(args.png)}  ({drivable.shape[1]}x{drivable.shape[0]} px, '
        f'drivable frac={drivable.mean():.3f})')
    out('=' * 78)

    # GT coverage validation
    out('\n[coordinate/coverage check on GROUND-TRUTH points]  (map & nomap share GT)')
    for tag, a in (('map', am),):
        in_map_gt = a['gt_total'] - a['gt_offmap']
        out(f'  GT points on drivable: {pct(a["gt_drivable"], a["gt_total"]):.1f}%   '
            f'occupied(in-map black): {pct(a["gt_occupied"], a["gt_total"]):.1f}%   '
            f'off-map: {pct(a["gt_offmap"], a["gt_total"]):.1f}%   (n={a["gt_total"]})')
        out(f'  of the GT points that fall INSIDE the map, {pct(a["gt_drivable"], in_map_gt):.2f}% '
            f'are on drivable (a real offset would push some onto black).')

    # objective registration probe: shift ALL GT points by (dx,dy) and see which offset
    # maximizes the on-drivable fraction. argmax at (0,0) => no misregistration.
    gt_world = []
    for f in bmap['frames']:
        for nd in f['nodes']:
            if nd['type'] != args.node_type:
                continue
            for k in ('history', 'future'):
                arr = np.asarray(nd[k])
                if arr.size:
                    gt_world.append(to_world(arr, bmap['meta']['x_min'], bmap['meta']['y_min']).reshape(-1, 2))
    gt_world = np.concatenate(gt_world)
    offs = np.arange(-4.0, 4.01, 1.0)
    best = (0.0, 0.0, -1.0)
    base_frac = None
    for dx in offs:
        for dy in offs:
            shifted = gt_world + np.array([dx, dy])
            in_map, is_drv = classify(shifted, drivable, H)
            frac = is_drv.sum() / max(in_map.sum(), 1)     # drivable among in-map
            if dx == 0 and dy == 0:
                base_frac = frac
            if frac > best[2]:
                best = (dx, dy, frac)
    out(f'  registration probe: best on-drivable offset = (dx={best[0]:+.0f}, dy={best[1]:+.0f}) m '
        f'-> {best[2]*100:.2f}%  vs  (0,0) -> {base_frac*100:.2f}%')
    out('  -> argmax at (0,0) means the raster and trajectories are aligned; the figure offset '
        'was a backdrop-extent artifact (now fixed).')

    # Headline: trajectory crossing rate
    out('\n[PREDICTED SAMPLE TRAJECTORIES that cross into occupied area]  (S=%d per node-timestep)'
        % bmap['meta']['num_samples'])
    out('  metric                         nomap        map        abs.drop   rel.drop')
    for label, key in (('cross OCCUPIED (in-map black)', 'traj_cross_occ'),
                       ('cross OFF-DRIVABLE (occ+offmap)', 'traj_cross_off')):
        pm = pct(am[key], am['traj_total'])
        pn = pct(an[key], an['traj_total'])
        rel = (pn - pm) / pn * 100 if pn else float('nan')
        out(f'  {label:32s} {pn:6.2f}%    {pm:6.2f}%     {pn-pm:+6.2f}pp   {rel:5.1f}%')
    out(f'  (n trajectories each: nomap={an["traj_total"]}, map={am["traj_total"]})')

    # Point-level
    out('\n[PREDICTED SAMPLE POINTS]  (all S x ph points)')
    out('  metric                         nomap        map        abs.drop')
    for label, key in (('occupied (in-map black)', 'pt_occupied'),):
        pm = pct(am[key], am['pt_total']); pn = pct(an[key], an['pt_total'])
        out(f'  {label:32s} {pn:6.2f}%    {pm:6.2f}%     {pn-pm:+6.2f}pp')
    off_m = am['pt_occupied'] + am['pt_offmap']; off_n = an['pt_occupied'] + an['pt_offmap']
    pm = pct(off_m, am['pt_total']); pn = pct(off_n, an['pt_total'])
    out(f'  {"off-drivable (occ+offmap)":32s} {pn:6.2f}%    {pm:6.2f}%     {pn-pm:+6.2f}pp')
    out(f'  {"off-map only":32s} {pct(an["pt_offmap"],an["pt_total"]):6.2f}%    '
        f'{pct(am["pt_offmap"],am["pt_total"]):6.2f}%')
    out(f'  mean occupied points / trajectory: nomap={an["occ_pts_per_traj_sum"]/max(an["traj_total"],1):.3f}  '
        f'map={am["occ_pts_per_traj_sum"]/max(am["traj_total"],1):.3f}  (out of {bmap["meta"]["ph"]})')

    # Most-likely
    out('\n[MOST-LIKELY path]')
    for label, key in (('cross OCCUPIED', 'ml_cross_occ'), ('cross OFF-DRIVABLE', 'ml_cross_off')):
        pm = pct(am[key], am['ml_total']); pn = pct(an[key], an['ml_total'])
        out(f'  {label:32s} {pn:6.2f}%    {pm:6.2f}%     {pn-pm:+6.2f}pp   '
            f'({an[key]}/{an["ml_total"]} -> {am[key]}/{am["ml_total"]})')

    # ---------- significance ----------
    out('\n[SIGNIFICANCE]')
    # (a) unpaired two-proportion z-test on trajectory crossings (correlation caveat)
    z, p = two_prop_ztest(an['traj_cross_occ'], an['traj_total'],
                          am['traj_cross_occ'], am['traj_total'])
    out(f'  two-proportion z-test on OCCUPIED trajectory crossings: z={z:.2f}, p={p:.2e}')
    out('    (upper bound: the S samples per node-timestep are correlated, so effective n is smaller)')

    # (b) paired by (timestep, node id): per-key occupied-crossing fraction, map vs nomap
    pk_m = results['map']['per_key']; pk_n = results['nomap']['per_key']
    common = sorted(set(pk_m) & set(pk_n))
    diffs_occ, diffs_off = [], []
    n_worse, n_better, n_same = 0, 0, 0
    for k in common:
        m, n = pk_m[k], pk_n[k]
        if m['n_traj'] == 0 or n['n_traj'] == 0:
            continue
        fm = m['traj_cross_occ'] / m['n_traj']
        fn = n['traj_cross_occ'] / n['n_traj']
        d = fn - fm            # positive => map reduced crossings
        diffs_occ.append(d)
        diffs_off.append(n['traj_cross_off'] / n['n_traj'] - m['traj_cross_off'] / m['n_traj'])
        if d > 1e-9: n_better += 1
        elif d < -1e-9: n_worse += 1
        else: n_same += 1
    diffs_occ = np.array(diffs_occ)
    W, pw, nw = wilcoxon(diffs_occ)
    out(f'  paired by (timestep,node): {len(diffs_occ)} node-timesteps compared')
    out(f'    mean crossing-fraction change (nomap-map): {diffs_occ.mean()*100:+.2f}pp '
        f'(>0 => map helps); median={np.median(diffs_occ)*100:+.2f}pp')
    out(f'    map LOWER crossings: {n_better}   map HIGHER: {n_worse}   tied: {n_same}')
    out(f'    Wilcoxon signed-rank (occupied): W={W:.0f}, p={pw:.2e}, n_nonzero={nw}')

    # (c) stratified: only node-timesteps where the agent currently sits ON the mapped
    # drivable region -- the map cannot steer prediction where the agent isn't on it.
    n_onmap = sum(1 for k in common if pk_m[k].get('cur_drivable'))
    diffs_on = []
    for k in common:
        m, n = pk_m[k], pk_n[k]
        if m['n_traj'] == 0 or n['n_traj'] == 0 or not m.get('cur_drivable'):
            continue
        diffs_on.append(n['traj_cross_occ'] / n['n_traj'] - m['traj_cross_occ'] / m['n_traj'])
    diffs_on = np.array(diffs_on)
    out(f'\n  [stratified: agent currently ON mapped drivable region]  '
        f'{len(diffs_on)}/{len(common)} node-timesteps')
    if len(diffs_on):
        Won, pon, non = wilcoxon(diffs_on)
        # aggregate crossing rate on this subset
        sk = [k for k in common if pk_m[k].get('cur_drivable') and pk_m[k]['n_traj']]
        occ_m = sum(pk_m[k]['traj_cross_occ'] for k in sk); tot_m = sum(pk_m[k]['n_traj'] for k in sk)
        occ_n = sum(pk_n[k]['traj_cross_occ'] for k in sk); tot_n = sum(pk_n[k]['n_traj'] for k in sk)
        out(f'    occupied trajectory crossings: nomap={pct(occ_n,tot_n):.2f}%  '
            f'map={pct(occ_m,tot_m):.2f}%  (drop {pct(occ_n,tot_n)-pct(occ_m,tot_m):+.2f}pp)')
        out(f'    mean crossing-fraction change (nomap-map): {diffs_on.mean()*100:+.2f}pp')
        out(f'    Wilcoxon signed-rank: W={Won:.0f}, p={pon:.2e}, n_nonzero={non}')

    # pedestrian control (map should NOT change pedestrian predictions)
    ped_m, _, _ = analyse_bundle(bmap, drivable, H, node_type='PEDESTRIAN')
    ped_n, _, _ = analyse_bundle(bno, drivable, H, node_type='PEDESTRIAN')
    if ped_m['traj_total']:
        out('\n[PEDESTRIAN control]  (map feeds the VEHICLE encoder only; expect ~no change)')
        out(f'  occupied trajectory crossings: nomap={pct(ped_n["traj_cross_occ"],ped_n["traj_total"]):.2f}%  '
            f'map={pct(ped_m["traj_cross_occ"],ped_m["traj_total"]):.2f}%')

    out('\n' + '=' * 78)

    # ---------- save summary ----------
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'occupancy_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    summary = dict(
        node_type=args.node_type,
        map=am, nomap=an,
        traj_cross_occ_pct=dict(nomap=pct(an['traj_cross_occ'], an['traj_total']),
                                map=pct(am['traj_cross_occ'], am['traj_total'])),
        wilcoxon=dict(W=float(W), p=float(pw), n=int(nw),
                      mean_diff_pp=float(diffs_occ.mean() * 100),
                      map_lower=n_better, map_higher=n_worse, tied=n_same),
        ztest=dict(z=float(z), p=float(p)),
    )
    with open(os.path.join(args.out_dir, 'occupancy_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # ---------- figure ----------
    make_figure(bmap, bno, drivable, H, results, args.out_dir, args.node_type)
    print(f'\nWrote report + summary + figure to {args.out_dir}/')


def make_figure(bmap, bno, drivable, H, results, out_dir, node_type):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # crop the raster to the scene (from GT extent) for the scatter backdrop
    meta = bmap['meta']
    xmin, ymin = meta['x_min'], meta['y_min']
    gt = []
    for f in bmap['frames']:
        for nd in f['nodes']:
            if nd['type'] != node_type:
                continue
            for k in ('history', 'future'):
                a = np.asarray(nd[k])
                if a.size:
                    gt.append(to_world(a, xmin, ymin).reshape(-1, 2))
    gt = np.concatenate(gt)
    pad = 40.0
    wx0, wx1 = gt[:, 0].min() - pad, gt[:, 0].max() + pad
    wy0, wy1 = gt[:, 1].min() - pad, gt[:, 1].max() + pad
    corners = np.array([[wx0, wy0], [wx1, wy1]])
    cr = world_to_px(corners, H)
    c0, c1 = sorted(cr[:, 0]); r0, r1 = sorted(cr[:, 1])
    Himg, Wimg = drivable.shape
    c0, c1 = max(0, c0), min(Wimg, c1); r0, r1 = max(0, r0), min(Himg, r1)
    crop = drivable[r0:r1, c0:c1]
    # extent MUST match the (clipped) crop, not the requested window -- this scene runs off
    # the map edge, so the crop is smaller than [wx0,wx1]x[wy0,wy1]; using the requested
    # window would stretch the raster and make the roads look offset from the points.
    Hinv = np.linalg.inv(H)
    def _pxw(col, row):
        v = Hinv @ np.array([col, row, 1.0]); return v[0] / v[2], v[1] / v[2]
    wx0, _ = _pxw(c0 - 0.5, r0 - 0.5); wx1, _ = _pxw(c1 - 0.5, r0 - 0.5)
    _, wy1 = _pxw(c0 - 0.5, r0 - 0.5); _, wy0 = _pxw(c0 - 0.5, r1 - 0.5)

    def scatter_pts(ax, bundle, title):
        ax.imshow(crop, cmap='gray', origin='upper', extent=[wx0, wx1, wy1, wy0],
                  vmin=0, vmax=1, alpha=0.85)
        occ_pts, drv_pts = [], []
        for f in bundle['frames']:
            for nd in f['nodes']:
                if nd['type'] != node_type:
                    continue
                s = np.asarray(nd['samples'])
                if not s.size:
                    continue
                sw = to_world(s, xmin, ymin).reshape(-1, 2)
                in_map, is_drv = classify(sw, drivable, H)
                occ_pts.append(sw[~is_drv])
                drv_pts.append(sw[is_drv])
        occ_pts = np.concatenate(occ_pts); drv_pts = np.concatenate(drv_pts)
        # subsample for a legible plot
        rng = np.random.default_rng(0)
        def sub(a, n=25000):
            return a[rng.choice(len(a), min(n, len(a)), replace=False)] if len(a) else a
        drv_pts, occ_pts = sub(drv_pts), sub(occ_pts)
        ax.scatter(drv_pts[:, 0], drv_pts[:, 1], s=0.4, c='#2ECC71', alpha=0.10, linewidths=0)
        ax.scatter(occ_pts[:, 0], occ_pts[:, 1], s=0.6, c='#E74C3C', alpha=0.20, linewidths=0)
        ax.set_xlim(wx0, wx1); ax.set_ylim(wy1, wy0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('world x (m)'); ax.set_ylabel('world y (m)')
        ax.set_aspect('equal')

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    scatter_pts(axes[0], bno, 'NO MAP: predicted sample points\n(red = off-drivable)')
    scatter_pts(axes[1], bmap, 'WITH MAP: predicted sample points\n(red = off-drivable)')

    # horizon-step occupied rate
    so_m, sf_m, st_m = results['map']['steps']
    so_n, sf_n, st_n = results['nomap']['steps']
    ph = len(so_m); steps = np.arange(1, ph + 1)
    ax = axes[2]
    ax.plot(steps, 100 * so_n / np.maximum(st_n, 1), '-o', color='#C0392B', label='no map: occupied')
    ax.plot(steps, 100 * so_m / np.maximum(st_m, 1), '-o', color='#27AE60', label='with map: occupied')
    ax.plot(steps, 100 * sf_n / np.maximum(st_n, 1), '--', color='#C0392B', alpha=0.5, label='no map: off-drivable')
    ax.plot(steps, 100 * sf_m / np.maximum(st_m, 1), '--', color='#27AE60', alpha=0.5, label='with map: off-drivable')
    ax.set_xlabel('prediction horizon step (0.5 s each)')
    ax.set_ylabel('% of sample points')
    ax.set_title('Occupancy rate vs. horizon')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'Map effect on predicted-trajectory occupancy  ({node_type},  '
                 f'{meta["scene"]},  S={meta["num_samples"]}, ph={ph})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, 'occupancy_effect.png')
    fig.savefig(path, dpi=130)
    print(f'  figure -> {path}')


if __name__ == '__main__':
    main()
