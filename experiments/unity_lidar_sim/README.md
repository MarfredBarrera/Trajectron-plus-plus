# Unity-sim LiDAR prediction pipeline

Trajectron++ over ground-truth agent tracks exported from a Unity simulator: streaming
inference with per-timestep risk scoring, recorded to disk and visualized from the record.

All commands are run from this directory (`experiments/unity_lidar_sim/`), with the
`trajectron-gpu` conda environment; paths in the configs and in the arguments below are
relative to it.

```bash
# streaming: observe -> predict -> score, one timestep at a time
python src/unity_online.py --config configs/sweep_s1_rail_online.yaml --gpu 5 --style samples

# re-render a stored bundle (no model, no GPU)
python src/visualization/visualize.py --pred_file unity_out/<scene>/predictions.pkl

# build the risk figures from a stored run's recorded risk (no model, no GPU)
python src/risk_eval/cli.py --pred_file unity_out/<scene>/predictions_online.pkl --viz
```

## Layout

    src/unity_online.py         the driver: observe -> predict -> score per timestep
    src/online_engine.py        OnlineEngine, wrapping BatchedOnlineTrajectron
    src/bundle.py               the on-disk prediction bundle: format + save/load
    src/repo_paths.py           puts trajectron/ and experiments/nuScenes/ on sys.path

    src/scene_conditioning/     log ingest, Environment/map construction, ego conditioning
    src/risk_eval/              proximity-risk metric, its overlay video, and its CLI
    src/visualization/          bundle rendering + the model-free re-render CLI

    configs/                    one YAML per scene/run (see configs/config.yaml for every key)
    unity_data/<scene>/         gt_agents.json, poses.csv, frames.csv
    unity_maps/                 drivable-area raster + its world->pixel registration
    unity_out/<scene>/          prediction bundles, risk CSVs, rendered frames and video

`src/` is the source root: the scripts under it are meant to be run directly
(`python src/unity_online.py ...`), which puts `src/` on `sys.path`, so modules import each
other as `bundle`, `risk_eval.proximity`, `visualization.traj_viz`. Nothing depends on the
working directory to resolve an import.

## Data flow

**Risk is scored online and recorded; visualizations are built from the record.** The
streaming driver scores each timestep at the moment it predicts it (`risk_eval.proximity`),
writes the per-sample result into the bundle (`risk_enter`) next to the predictions it
refers to, and exports the per-agent rows to CSV. Every figure — the overlay video, the
entering-trajectories-over-time plot — is then built by reading that file back, including
the ones the driver itself emits at the end of a run. So a plot made during the run and one
made from the stored bundle a week later come off the same code path and cannot disagree;
verified row-for-row against the CSV written inside the online loop.

The one deliberate exception is `--rescore`, which recomputes risk from the stored samples
at a different radius. That is for sweeping the threshold without re-running inference, and
it announces itself in the output so a rescored figure is never mistaken for the run's own
result.
