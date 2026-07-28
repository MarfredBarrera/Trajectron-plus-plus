"""Puts the Trajectron++ source and the nuScenes experiment helpers on sys.path.

They are plain top-level modules (`environment`, `model`, `helper`, `utils`) in a repo that is
not installed as a package, so something has to point at them. Import this module for its side
effect before importing any of them:

    import repo_paths  # noqa: F401  (sys.path side effect)
    from environment import Environment

Resolved off `__file__`, not the working directory, so it holds no matter where a script is
launched from.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

for _p in (os.path.join(REPO_ROOT, 'trajectron'),
           os.path.join(REPO_ROOT, 'experiments', 'nuScenes')):
    if _p not in sys.path:
        sys.path.append(_p)
