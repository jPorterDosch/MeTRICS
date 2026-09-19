import numpy as np
import pytest

from streamvggt.datasets.base.segments import (
    _max_gap_from_rate,
    segment_frame_ids,
    segment_frame_ids_by_rate,
    split_by_rate,
    split_contiguous,
)


def test_contiguous_keys_stay_one_segment() -> None:
    segments = split_contiguous(np.arange(10), max_gap=1)
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0], np.arange(10))


def test_gap_starts_a_new_segment_at_the_frame_after_the_break() -> None:
    # 0 1 2 | 9 10 -- the break belongs to the frame AFTER it
    segments = split_contiguous(np.array([0, 1, 2, 9, 10]), max_gap=1)
    assert [seg.tolist() for seg in segments] == [[0, 1, 2], [3, 4]]


def test_step_equal_to_max_gap_is_still_continuous() -> None:
    assert len(split_contiguous(np.array([0.0, 0.1, 0.2]), max_gap=0.1)) == 1


def test_descending_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="ascending"):
        split_contiguous(np.array([3, 1, 2]), max_gap=1)


def test_short_segments_are_dropped_not_padded() -> None:
    # runs of 4 and 2 frames; only the first can fill a 4-view clip
    keys = np.array([0, 1, 2, 3, 50, 51])
    sequences = segment_frame_ids([10, 11, 12, 13, 14, 15], keys, 1, min_frames=4)
    assert sequences == [[10, 11, 12, 13]]


def test_segment_ids_keep_the_callers_global_numbering() -> None:
    keys = np.array([0, 1, 9, 10])
    sequences = segment_frame_ids([100, 101, 102, 103], keys, 1, min_frames=2)
    assert sequences == [[100, 101], [102, 103]]


def test_max_gap_tracks_the_frame_rate() -> None:
    thirty_fps = np.arange(100) / 30.0
    ten_fps = np.arange(100) / 10.0
    assert _max_gap_from_rate(thirty_fps, 1.5, 0.5) == pytest.approx(1.5 / 30.0)
    assert _max_gap_from_rate(ten_fps, 1.5, 0.5) == pytest.approx(1.5 / 10.0)


def test_max_gap_is_capped_for_a_mostly_fragmented_scene() -> None:
    # steps of 10 s dominate, so the median step IS a gap; without the ceiling
    # the whole scene would come back as one run
    fragmented = np.array([0.0, 0.1, 10.0, 10.1, 20.0, 20.1, 30.0])
    assert _max_gap_from_rate(fragmented, 1.5, 0.5) == 0.5
    assert len(split_contiguous(fragmented, max_gap=0.5)) == 4


def test_jittered_timestamps_do_not_split() -> None:
    # ARKit vga_wide alternates 0.033 / 0.034 s steps
    jittered = np.cumsum(np.array([0.0] + [0.033, 0.034] * 20))
    assert len(split_by_rate(jittered, 1.5, 0.5)) == 1


def test_a_locally_dense_run_is_re_split_against_its_own_rate() -> None:
    # scene median step is 0.2 s, so a single pass keeps the 0.3 s step inside
    # the dense 0.1 s run; the refinement pass must break it out
    keys = np.concatenate(
        [
            np.arange(6) * 0.1,  # dense run, 0.1 s steps
            [0.8],  # 0.3 s after the dense run: a break at the LOCAL rate
            0.8 + np.arange(1, 8) * 0.2,  # sparse run, 0.2 s steps
        ]
    )
    runs = split_by_rate(keys, 1.5, 0.5)
    for run in runs:
        steps = np.diff(keys[run])
        if steps.size >= 2:
            assert steps.max() <= 1.5 * np.median(steps) + 1e-9
    assert len(runs) > 1


def test_by_rate_segments_carry_global_ids_and_drop_short_runs() -> None:
    keys = np.array([0.0, 0.1, 0.2, 0.3, 9.0, 9.1])
    sequences = segment_frame_ids_by_rate(
        [10, 11, 12, 13, 14, 15], keys, 1.5, 0.5, min_frames=4
    )
    assert sequences == [[10, 11, 12, 13]]
