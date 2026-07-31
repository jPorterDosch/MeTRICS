"""CPU regressions for upheld Area E depth-conditioning findings."""

import os
import tempfile
import threading
from unittest import mock

import torch

from streamvggt.depth_cond.cache import EncoderFeatureCache


def test_concurrent_cache_writers_use_private_temp_files() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = EncoderFeatureCache(cache_dir)
        barrier = threading.Barrier(2)
        temp_paths: list[str] = []
        errors: list[BaseException] = []

        real_replace = os.replace

        def synchronized_replace(source: str, destination: str) -> None:
            temp_paths.append(source)
            barrier.wait()
            real_replace(source, destination)

        class FakeTensor:
            def detach(self) -> "FakeTensor":
                return self

            def to(self, _dtype: torch.dtype) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

        def fake_save(_tensor: FakeTensor, path: str) -> None:
            with open(path, "wb") as handle:
                handle.write(b"feature")

        def writer() -> None:
            try:
                cache.save("shared-key", FakeTensor())
            except BaseException as error:
                errors.append(error)

        with (
            mock.patch("streamvggt.depth_cond.cache.torch.save", fake_save),
            mock.patch("streamvggt.depth_cond.cache.os.replace", synchronized_replace),
        ):
            threads = [threading.Thread(target=writer) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not errors, errors
        assert len(set(temp_paths)) == 2
        assert os.path.isfile(cache._path("shared-key"))


if __name__ == "__main__":
    test_concurrent_cache_writers_use_private_temp_files()
