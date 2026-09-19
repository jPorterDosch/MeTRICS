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


def test_duplicated_frames_survive_into_a_run() -> None:
    # A frame duplicated once does NOT raise: one zero step among many leaves
    # the median alone, and a zero step is under any threshold, so the pair
    # stays inside its run and the clip carries the same frame twice. Nothing
    # downstream can tell -- which is why the preprocessing match is kept
    # one-to-one at the source (preprocess_arkitscenes_highres.py) rather than
    # relying on a check here.
    keys = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
    runs = split_by_rate(keys, 1.5, 0.5)
    assert [run.tolist() for run in runs] == [[0, 1, 2, 3, 4]]


def test_a_run_of_only_duplicates_fails_loudly() -> None:
    # when duplicates dominate, the median step IS zero and there is no rate
    # to derive: raise rather than return 0 and hand back one run per frame
    with pytest.raises(ValueError, match="positive"):
        split_by_rate(np.zeros(4), 1.5, 0.5)


def test_capture_slower_than_the_ceiling_is_dropped_not_merged() -> None:
    # 1 s steps against a 0.5 s ceiling: below 2 fps, so every step is a gap
    # and nothing survives a min_frames filter -- see _max_gap_from_rate
    slow = np.arange(10) * 1.0
    assert all(len(run) == 1 for run in split_by_rate(slow, 1.5, 0.5))
    assert segment_frame_ids_by_rate(list(range(10)), slow, 1.5, 0.5, 4) == []


def test_by_rate_segments_carry_global_ids_and_drop_short_runs() -> None:
    keys = np.array([0.0, 0.1, 0.2, 0.3, 9.0, 9.1])
    sequences = segment_frame_ids_by_rate(
        [10, 11, 12, 13, 14, 15], keys, 1.5, 0.5, min_frames=4
    )
    assert sequences == [[10, 11, 12, 13]]
