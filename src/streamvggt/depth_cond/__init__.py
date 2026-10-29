from .config import (
    DepthCondCfg,
    EncoderCacheCfg,
    EncoderType,
    HeadType,
    InjectionType,
    LoRACfg,
    LoRATarget,
    MetricCfg,
    NormType,
    SparseSimMode,
    TemporalType,
    TrainCondCfg,
    SHORT_HASH_LEN,
    experiment_hash,
    experiment_id,
    experiment_manifest,
)
from .sparse import simulate_sparse_depth

# The model-side modules load on first use (PEP 562), not with the package:
# the benchmark's scoring imports only config and sparse, and its results
# record every file a run imports (eval.provenance.loaded_code), so an
# eager import here would tie a baseline's results to our model's code.
_LAZY = {
    "DepthConditioner": "conditioner",
    "dpt_fusion_sizes": "conditioner",
    "masked_downsample": "conditioner",
    "LoRALinear": "lora",
    "LoRAQKV": "lora",
    "apply_lora": "lora",
    "param_stats": "lora",
    "EncoderFeatureCache": "cache",
    "MetricStreamVGGT": "model",
    "seed_everything": "utils",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DepthCondCfg",
    "LoRACfg",
    "EncoderCacheCfg",
    "TrainCondCfg",
    "MetricCfg",
    "EncoderType",
    "InjectionType",
    "TemporalType",
    "NormType",
    "HeadType",
    "LoRATarget",
    "SparseSimMode",
    "experiment_manifest",
    "experiment_hash",
    "experiment_id",
    "SHORT_HASH_LEN",
    "DepthConditioner",
    "masked_downsample",
    "dpt_fusion_sizes",
    "apply_lora",
    "LoRAQKV",
    "LoRALinear",
    "param_stats",
    "EncoderFeatureCache",
    "MetricStreamVGGT",
    "simulate_sparse_depth",
    "seed_everything",
]
