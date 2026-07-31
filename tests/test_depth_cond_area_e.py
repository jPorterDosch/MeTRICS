"""CPU regressions for upheld Area E depth-conditioning findings."""

import os
import tempfile
import threading
from unittest import mock

import torch

from streamvggt.depth_cond.cache import EncoderFeatureCache
from streamvggt.depth_cond.config import DepthCondCfg, InjectionType, LoRACfg, NormType


def assert_value_error(message: str, callback) -> None:
    try:
        callback()
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


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


def test_fixed_normalization_requires_finite_positive_constant() -> None:
    for constant in (0.0, -1.0, float("nan"), float("inf")):
        assert_value_error(
            "norm_constant_m",
            lambda constant=constant: DepthCondCfg(
                enabled=True,
                norm=NormType.FIXED,
                norm_constant_m=constant,
            ).validate(),
        )

    DepthCondCfg(enabled=True, norm=NormType.RAW, norm_constant_m=0.0).validate()


def test_enabled_lora_requires_finite_positive_alpha() -> None:
    for alpha in (0.0, -1.0, float("nan"), float("inf")):
        assert_value_error(
            "lora.alpha",
            lambda alpha=alpha: LoRACfg(enabled=True, alpha=alpha).validate(),
        )

    LoRACfg(enabled=False, alpha=0.0).validate()


def test_enabled_head_injection_requires_a_head() -> None:
    assert_value_error(
        "depth_cond.heads",
        lambda: DepthCondCfg(
            enabled=True,
            injection=InjectionType.HEAD,
            heads=[],
        ).validate(),
    )
    DepthCondCfg(enabled=True, injection=InjectionType.TOKEN, heads=[]).validate()
    DepthCondCfg(enabled=False, injection=InjectionType.HEAD, heads=[]).validate()


if __name__ == "__main__":
    test_concurrent_cache_writers_use_private_temp_files()
    test_fixed_normalization_requires_finite_positive_constant()
    test_enabled_lora_requires_finite_positive_alpha()
    test_enabled_head_injection_requires_a_head()
