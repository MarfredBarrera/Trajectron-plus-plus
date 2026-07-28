# Unity-sim LiDAR prediction pipeline

Trajectron++ over ground-truth agent tracks exported from a Unity simulator, in two flavours:
batch replay over a finished log, and streaming inference with per-timestep risk scoring.

All commands are run from this directory (`experiments/unity_lidar_sim/`), with the
`trajectron-gpu` conda environment; paths in the configs and in the arguments below are
relative to it.

```bash
# streaming: observe -> predict -> score, one timestep at a time
python src/unity_online.py --config configs/sweep_s1_rail_online.yaml --gpu 5 --style samples

# batch replay of a finished log
python src/unity_predict.py --config configs/sweep_s1_rail.yaml --gpu 0

# re-render a stored bundle (no model, no GPU)
python src/visualization/visualize.py --pred_file unity_out/<scene>/predictions.pkl

# re-score a stored bundle for ego-crossing risk
python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions.pkl --viz
```

## Layout

    src/unity_online.py         streaming driver: observe -> predict -> score per timestep
    src/unity_predict.py        batch driver: Trajectron.predict over a finished log
    src/online_engine.py        OnlineEngine, wrapping BatchedOnlineTrajectron
    src/bundle.py               the on-disk prediction bundle: format + save/load
    src/repo_paths.py           puts trajectron/ and experiments/nuScenes/ on sys.path

    src/scene_conditioning/     log ingest, Environment/map construction, ego conditioning
    src/risk_eval/              crossing-risk metric, its overlay video, and its CLI
    src/visualization/          bundle rendering + the model-free re-render CLI

    configs/                    one YAML per scene/run (see configs/config.yaml for every key)
    unity_data/<scene>/         gt_agents.json, poses.csv, frames.csv
    unity_maps/                 drivable-area raster + its world->pixel registration
    unity_out/<scene>/          prediction bundles, risk CSVs, rendered frames and video

`src/` is the source root: the scripts under it are meant to be run directly
(`python src/unity_online.py ...`), which puts `src/` on `sys.path`, so modules import each
other as `bundle`, `risk_eval.crossing`, `visualization.traj_viz`. Nothing depends on the
working directory to resolve an import.

## Data flow

Both drivers write the same bundle format (`src/bundle.py`), so prediction, scoring and
rendering are fully separable: a run can be re-scored or re-rendered without touching the
model. The streaming driver additionally scores each timestep as it produces it — the same
`risk_eval.crossing` code path the offline CLI uses, verified to give identical output.
