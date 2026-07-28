"""Rendering prediction bundles to frames and video.

`traj_viz` draws the predicted distribution (sample fan / Gaussian blobs) over the drivable
map, in world or ego frame; `visualize` is the model-free CLI that re-renders a stored
bundle. `colors` is kept dependency-free and importable on its own so the risk metric can
label agents consistently without pulling matplotlib into the online loop -- which is why
this file deliberately imports nothing.
"""
