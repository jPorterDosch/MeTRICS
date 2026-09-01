"""Zip-backed IO for the streamvggt datasets.

Thin re-export of dust3r.utils.zipio, which holds the canonical
implementation. Keeping one copy matters for correctness, not just tidiness:
the open-handle cache must be a single cache so a forked DataLoader worker
never inherits a parent's ZipFile through a second, independent module.

See dust3r/utils/zipio.py for the layout and the '<scene>.zip/' virtual-path
convention. The import direction (streamvggt -> dust3r) matches
streamvggt/loss/regr_3d_pose.py.
"""

from dust3r.utils.zipio import (  # noqa: F401
    SceneZipWriter,
    asset_root,
    exists,
    frames_root,
    listdir,
    np_load,
    read_bytes,
    split_zip_path,
)

__all__ = [
    "SceneZipWriter",
    "asset_root",
    "exists",
    "frames_root",
    "listdir",
    "np_load",
    "read_bytes",
    "split_zip_path",
]
