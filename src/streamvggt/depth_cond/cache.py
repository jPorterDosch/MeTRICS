"""Frozen-encoder feature cache.

The RGB patch-embed encoder (DINOv2 ViT-L/14) is frozen, so its output is
deterministic given RGB. Features are computed once and stored per frame,
keyed by a stable frame id; training then forwards/backwards only through the
conditioner + decoder (LoRA) + heads.

Features are computed and stored in fp32. This keeps one stable representation
on disk, but it may differ from an autocast live path.
"""

import hashlib
import os
import re
import uuid
from typing import Optional

import torch


class EncoderFeatureCache:
    def __init__(self, cache_dir: str, encoder_fingerprint: str) -> None:
        if not encoder_fingerprint:
            raise ValueError("encoder_fingerprint must be non-empty")
        self.dir = cache_dir
        self.namespace = f"encoder-features-v1:{encoder_fingerprint}"
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        # The readable prefix is lossy (distinct keys can sanitize identically,
        # e.g. 'a/b' and 'a_b'), so a digest of the namespace and RAW key is
        # always appended: filename uniqueness never depends on sanitization.
        digest = hashlib.sha1(f"{self.namespace}\0{key}".encode()).hexdigest()[:16]
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:80]
        return os.path.join(self.dir, f"{safe}_{digest}.pt")

    def has(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def load(self, key: str, device=None) -> Optional[torch.Tensor]:
        p = self._path(key)
        if not os.path.isfile(p):
            return None
        t = torch.load(p, map_location=device if device is not None else "cpu")
        return t

    def save(self, key: str, feats: torch.Tensor) -> None:
        # atomic write: partial files must never be readable as cache hits
        p = self._path(key)
        tmp = f"{p}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save(feats.detach().to(torch.float32).cpu(), tmp)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
