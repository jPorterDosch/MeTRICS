"""Contiguous-segment bookkeeping for clip sampling.

A processed "scene" is not always one continuous capture. ARKitScenes scenes
carry a median of 8 fragments each (median run 11 frames) because DUSt3R's
covisibility selection kept scattered windows, and even ScanNet skips a frame
on 0.05% of steps (largest observed jump: 141 frames). Sampling a clip from a
scene's flat frame list therefore produces clips that jump across seconds-long
breaks, which the model is trained to read as ordinary camera motion.

The loaders sample within `ids_all` (see BaseMultiViewDataset.get_seq_from_start_id),
so the fix is to hand them one entry per contiguous run rather than one per
scene: a clip then cannot cross a break by construction, with no change to the
sampler itself. ScanNetpp_Multi already works this way -- its `seq_img_list` /
`seqids` pair splits a scene into a DSLR and an iPhone run -- so the loaders
here adopt those names and that sentinel convention rather than a second idiom.

`max_gap` is a protocol parameter -- it changes which clips exist, hence every
metric computed from them -- so each dataset states its own value explicitly
rather than inheriting a default from here.
"""

import numpy as np


def split_contiguous(keys: np.ndarray, max_gap: float) -> list[np.ndarray]:
    """Split ascending frame keys into runs of consecutive frames.

    args:
        keys: per-frame position on the capture timeline, ascending, in the
            dataset's own units -- a frame index (ScanNet, HAMMER, TartanAir,
            ScanNet++) or a timestamp in seconds (ARKitScenes). Must be sorted;
            an unsorted array means the caller's frame ordering is wrong, which
            silently produces nonsense segments, so it raises instead.
        max_gap: largest step still counted as continuous, in the same units.
            A step strictly greater than this starts a new segment.

    returns:
        list of index arrays into `keys`, in order, together covering every
        frame exactly once. A fully continuous scene yields a single segment.
    """
    keys = np.asarray(keys)
    if keys.ndim != 1:
        raise ValueError(f"keys must be 1-D, got shape {keys.shape}")
    if not np.isfinite(keys).all():
        raise ValueError("keys must be finite; got NaN or inf")
    if keys.size == 0:
        return []
    if keys.size == 1:
        return [np.array([0])]

    steps = np.diff(keys)
    if (steps < 0).any():
        first = int(np.argmax(steps < 0))
        raise ValueError(
            f"keys must be ascending; keys[{first}]={keys[first]} > "
            f"keys[{first + 1}]={keys[first + 1]}"
        )
    if max_gap <= 0:
        raise ValueError(f"max_gap must be positive, got {max_gap}")

    # +1: np.diff index i is the step BETWEEN i and i+1, and a break there
    # means the new segment starts at i+1
    breaks = np.flatnonzero(steps > max_gap) + 1
    return np.split(np.arange(keys.size), breaks)


def _max_gap_from_rate(
    timestamps: np.ndarray, gap_factor: float, ceiling: float
) -> float:
    """Continuity threshold for a capture whose frame rate is not known up front.

    Timestamped datasets (ARKitScenes) carry scenes recorded at different
    rates, and the processed tree may be subsampled from the raw one, so a
    hard-coded step in seconds would be wrong for half the data. The nominal
    frame period is the MEDIAN step -- robust to the minority of steps that are
    gaps -- scaled by `gap_factor` to admit timestamp jitter.

    `ceiling` bounds the result: in a scene that is mostly fragments the median
    step is itself a gap, and an unbounded factor rule would then merge every
    fragment into one run.

    args:
        timestamps: ascending frame times in seconds.
        gap_factor: multiple of the nominal frame period still counted as
            continuous.
        ceiling: largest threshold to return, in seconds.

    returns:
        the threshold to pass as `max_gap`.
    """
    if gap_factor <= 1:
        raise ValueError(f"gap_factor must exceed 1, got {gap_factor}")
    if ceiling <= 0:
        raise ValueError(f"ceiling must be positive, got {ceiling}")
    steps = np.diff(np.asarray(timestamps, dtype=np.float64))
    if steps.size == 0:
        return ceiling
    nominal = float(np.median(steps))
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(
            f"median frame step must be positive and finite, got {nominal}; "
            "the timestamps are duplicated or out of order"
        )
    return min(gap_factor * nominal, ceiling)


def split_by_rate(
    keys: np.ndarray, gap_factor: float, ceiling: float, max_passes: int = 5
) -> list[np.ndarray]:
    """Split ascending keys into runs, deriving the threshold from the rate.

    One pass is not enough. The first threshold comes from the WHOLE scene, and
    a scene whose frames are irregular (ARKitScenes high-res: laser depth
    exists for a fraction of the capture) has runs that are locally denser than
    the scene as a whole. Such a run then keeps a step several times its own
    frame period -- measured at 2x on 186 of 770 high-res clips -- which is the
    very discontinuity the split is meant to remove.

    So each run is re-split against its OWN rate until the runs stop changing,
    which makes the postcondition local: within a returned run, no step exceeds
    `gap_factor` times that run's median step (or `ceiling`, whichever binds).

    args:
        keys: ascending timeline positions (see `split_contiguous`).
        gap_factor: multiple of a run's median step still counted continuous.
        ceiling: largest threshold to apply, in the same units as `keys`.
        max_passes: refinement limit. Splitting only ever subdivides, so this
            terminates on its own; the bound just keeps a pathological scene
            from iterating once per frame.

    returns:
        list of index arrays into `keys`, in order, covering every frame once.
    """
    runs = [np.arange(np.asarray(keys).size)]
    keys = np.asarray(keys)
    for _ in range(max_passes):
        refined = []
        for run in runs:
            if run.size < 3:  # a median step needs at least two steps
                refined.append(run)
                continue
            gap = _max_gap_from_rate(keys[run], gap_factor, ceiling)
            refined.extend(run[part] for part in split_contiguous(keys[run], gap))
        if len(refined) == len(runs):
            return refined
        runs = refined
    return runs


def segment_frame_ids(
    frame_ids: list[int], keys: np.ndarray, max_gap: float, min_frames: int
) -> list[list[int]]:
    """Group a scene's global frame ids into contiguous segments.

    Thin wrapper over `split_contiguous` for the loaders' `_load_data` loops,
    which hold global (offset-shifted) ids rather than within-scene positions.

    args:
        frame_ids: the scene's global frame ids, ordered as `keys` is.
        keys: timeline positions for the same frames (see `split_contiguous`).
        max_gap: continuity threshold, in `keys` units.
        min_frames: drop segments shorter than this -- a segment too short to
            fill one clip is unusable, and keeping it would put start ids in
            the sampling pool that cannot honor num_views.

    returns:
        list of global-id lists, one per usable segment. May be empty when no
        segment is long enough.
    """
    if len(frame_ids) != len(keys):
        raise ValueError(
            f"frame_ids and keys must align: {len(frame_ids)} vs {len(keys)}"
        )
    return [
        [frame_ids[p] for p in seg]
        for seg in split_contiguous(keys, max_gap)
        if len(seg) >= min_frames
    ]


def segment_frame_ids_by_rate(
    frame_ids: list[int],
    keys: np.ndarray,
    gap_factor: float,
    ceiling: float,
    min_frames: int,
) -> list[list[int]]:
    """`segment_frame_ids` for captures whose frame rate is not known up front.

    Same contract, except the continuity threshold is derived per run by
    `split_by_rate` instead of being passed in.
    """
    if len(frame_ids) != len(keys):
        raise ValueError(
            f"frame_ids and keys must align: {len(frame_ids)} vs {len(keys)}"
        )
    return [
        [frame_ids[p] for p in seg]
        for seg in split_by_rate(keys, gap_factor, ceiling)
        if len(seg) >= min_frames
    ]
