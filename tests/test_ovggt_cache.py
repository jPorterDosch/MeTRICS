"""The `ovggt:` curve source of scale_drift.py (eval/ovggt_cache.py). The
LoRA merge it relies on is export's merge_lora, checked by
tests/export_onnx_parity.py --cpu-unit; the end-to-end parity (OVGGT with
eviction off == our full-cache stream) needs the real weights and is a
one-off check, not a unit test."""

import scale_drift as SD


def test_ovggt_curve_source() -> None:
    c = SD.parse_curve("Ours, OVGGT cache (dropout @20)=ovggt:/runs/abc@20")
    assert (c.source, c.drop, c.is_ovggt, c.is_base) == (
        "ovggt:/runs/abc",
        20,
        True,
        False,
    )
    assert not SD.parse_curve("Ours=/runs/abc").is_ovggt


def test_cache_arm_curves_are_rgb_only_and_dropout_only() -> None:
    import argparse

    import pytest

    c = SD.parse_curve("OVGGT=ovggt")
    assert c.is_cache_arm and not c.metric and not c.is_ovggt
    assert SD.parse_curve("Ours, OVGGT cache=ovggt:/runs/abc").metric
    args = argparse.Namespace(
        experiment="dropout",
        window=20,
        ref_frames=20,
        dropout_at=20,
        step_at=40,
        step_factor=[1.25],
        sine_amplitude=1.2,
        sine_periods=[8.0],
        jitter_sigma=0.1,
        splice_scale=2.0,
        cut_pairs_per_bin=1,
    )
    # RGB-only curves alone: no drop of their own, --dropout-at only draws lines
    SD.validate(args, [c, SD.parse_curve("Inf=infinite_vggt")], 192)
    args.experiment = "step"
    args.dropout_at = None
    with pytest.raises(ValueError):
        SD.validate(args, [c], 192)
