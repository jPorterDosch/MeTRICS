"""CPU regressions for upheld Area E depth-conditioning findings."""

import os
import tempfile
import threading
from unittest import mock

import torch

import finetune_depth as fd
from streamvggt.depth_cond import cache as cache_module
from streamvggt.depth_cond.cache import EncoderFeatureCache
from streamvggt.depth_cond.config import (
    DepthCondCfg,
    EncoderCacheCfg,
    InjectionType,
    LoRACfg,
    MetricCfg,
    NormType,
)
from streamvggt.depth_cond.model import MetricStreamVGGT


def assert_value_error(message: str, callback) -> None:
    try:
        callback()
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_concurrent_cache_writers_use_private_temp_files() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = EncoderFeatureCache(cache_dir, "checkpoint")
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


def test_cache_contract_discloses_fp32_autocast_difference() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = EncoderFeatureCache(cache_dir, "checkpoint")
        cache.save("frame", torch.tensor([1.5], dtype=torch.bfloat16))
        assert cache.load("frame").dtype == torch.float32

    assert "numerically identical" not in cache_module.__doc__
    assert "may differ from an autocast live path" in cache_module.__doc__


def test_cache_namespace_separates_encoder_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        checkpoint_a = EncoderFeatureCache(cache_dir, "checkpoint-a")
        checkpoint_b = EncoderFeatureCache(cache_dir, "checkpoint-b")
        checkpoint_a.save("frame", torch.tensor([1.0]))

        assert checkpoint_a.load("frame") is not None
        assert checkpoint_b.load("frame") is None
        assert checkpoint_a._path("frame") != checkpoint_b._path("frame")


def test_build_model_resume_constructs_usable_encoder_cache() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        mcfg = MetricCfg(encoder_cache=EncoderCacheCfg(enabled=True, dir=cache_dir))
        args = fd.FinetuneDepthCfg(resume="synthetic-resume.pth")

        def initialize_without_backbone(model, cfg) -> None:
            torch.nn.Module.__init__(model)
            model.cfg = cfg
            model._encoder_cache_dir = cfg.encoder_cache.dir
            model.cache = None

        stats = {
            "total_params": 0,
            "trainable_params": 0,
            "trainable_pct": 0.0,
            "base_attention_frozen": True,
        }
        with (
            mock.patch.object(
                MetricStreamVGGT, "__init__", initialize_without_backbone
            ),
            mock.patch.object(MetricStreamVGGT, "apply_lora_adapters", return_value=0),
            mock.patch.object(
                MetricStreamVGGT, "freeze_for_finetune", return_value=stats
            ),
            mock.patch("builtins.open", mock.mock_open(read_data=b"resume-state")),
        ):
            model, _ = fd.build_model(args, mcfg, torch.device("cpu"))

        assert isinstance(model.cache, EncoderFeatureCache)
        model.cache.save("frame", torch.tensor([2.0]))
        assert torch.equal(model.cache.load("frame"), torch.tensor([2.0]))


def test_supplied_cache_keys_must_match_batch_cardinality() -> None:
    model = MetricStreamVGGT.__new__(MetricStreamVGGT)
    torch.nn.Module.__init__(model)
    model.cache = mock.Mock()
    model._encoder_cache_dir = "enabled"
    images = torch.zeros(2, 1, 3, 1, 1)

    assert_value_error(
        "expected 2 cache keys, received 1",
        lambda: model._cached_patch_tokens([{"cache_key": "only-one"}], images),
    )


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


def test_enabled_lora_requires_targets() -> None:
    assert_value_error(
        "lora.targets",
        lambda: LoRACfg(enabled=True, targets=[]).validate(),
    )
    LoRACfg(enabled=False, targets=[]).validate()


if __name__ == "__main__":
    test_concurrent_cache_writers_use_private_temp_files()
    test_cache_contract_discloses_fp32_autocast_difference()
    test_cache_namespace_separates_encoder_checkpoints()
    test_build_model_resume_constructs_usable_encoder_cache()
    test_supplied_cache_keys_must_match_batch_cardinality()
    test_fixed_normalization_requires_finite_positive_constant()
    test_enabled_lora_requires_finite_positive_alpha()
    test_enabled_head_injection_requires_a_head()
    test_enabled_lora_requires_targets()
