"""Turning a Unity-sim log into something Trajectron++ can consume.

`unity_scene` covers the whole path from raw logs to a ready-to-predict scene: ground-truth
track ingest, ego-pose alignment and filtering, scene bounds, Environment/Node construction,
the drivable-map raster, and -- when `ego_conditioning` is on -- the ego robot node and the
per-timestep ego plan the model is conditioned on (`UnityScene.ego_plan_state`).

Both drivers build their scene through `prepare_scene(cfg)`, so batch and online runs are
guaranteed to see identical data.
"""
