import logging
from typing import List, Tuple, Iterable, Union
try: 
    from typing import Literal, Optional
except:
    from typing_extensions import Literal
    from typing import Optional
import torch
import numpy as np
from numpy.typing import NDArray

import warnings

class DataFormatWarning(UserWarning):
    pass


class DepthMap:
    def __init__(self, data: np.ma.MaskedArray,
                 inverse: bool,
                 range: Optional[Tuple[float, float]] = None,
                 scale: Optional[float] = None,
                 shift: Optional[float] = None):
        """
        Represents a (potentially sparse) depth map of arbitrary scale and shift, parameterized as depth or
        inverse depth, indicated by the `inverse` argument.

        :param data: a masked array of shape (height, width) that represents the (inverse) depth map,
                     with invalid depths masked.
        :param inverse: If `True`, then interpret `data` as inverse depths.
        :param range: Optional tuple indicating the range to which the values may be clipped, e.g. for visualization
                      or discretization and storage purposes.
                      Values in `data` may exceed the range. The range is supposed to be constant for
                      inverse depth maps of the same video sequence.
        :param scale: Optional scale that can be used to obtain metric (inverse) depth as follows:
                      metric = (data - shift) / scale
        :param shift: Optional shift that can be used to obtain metric (inverse) depth as follows:
                      metric = (data - shift) / scale
        """
        if isinstance(data, np.ndarray) and not isinstance(data, np.ma.MaskedArray):
            DataFormatWarning("DepthMap requires the data field to be of type numpy.ma.MaskedArray, but received "
                           "data of another subtype of numpy.ndarray. An attempt will be made to convert the given "
                           "array.")
            data = np.ma.masked_array(data)
        self._data = data
        self._inverse = inverse
        self._range = range
        self._scale = scale
        self._shift = shift
        # _inverted_map caches the result of invert() to avoid redundant computations often occuring in alignment.
        self._inverted_map: Optional[DepthMap] = None

    @property
    def data(self):
        return self._data

    @property
    def inverse(self):
        return self._inverse

    @property
    def range(self):
        return self._range

    @property
    def scale(self):
        return self._scale

    @property
    def shift(self):
        return self._shift

    def invert(self):
        """
        Creates a new depth map with inverted parameterization, i.e. turns non-inverse depth maps to inverse depth maps
        and vice-versa. This conversion is only possible if `shift`=0, and will raise an exception otherwise.
        """
        if self._inverted_map is not None:
            return self._inverted_map
        if self._shift != 0:
            raise Exception(f"Depth maps with non-zero shift cannot be inverted (shift={self._shift}).")
        range = None if self._range is None else \
            (1 / self._range[1] if self._range[1] != 0 else self._range[0] / 1024,
             1 / self._range[0] if self._range[0] != 0 else self._range[1] / 1024)
        return DepthMap(1 / self._data,
                        inverse=not self._inverse,
                        range=range,
                        scale=None if self._scale is None else 1 / self._scale,
                        shift=0
                        )

    def is_metric(self):
        return (self._scale is not None) and (self._shift is not None)

    def metric_depth(self) -> np.ma.MaskedArray:
        if not self.is_metric():
            raise Exception("Not a metric depth map!")
        if self._inverse:
            return self._scale / (self._data - self._shift)
        else:
            return (self._data - self._shift) / self._scale


class Alignment:
    def __init__(self, inverse: bool, scale: float, shift: float,
                 metric_scale: Optional[float], metric_shift: Optional[float]):
        self.inverse = inverse
        self.scale = scale
        self.shift = shift
        self.metric_scale = metric_scale
        self.metric_shift = metric_shift

    def apply(self, depth_map: DepthMap, *, inplace: bool = False) -> DepthMap:
        if depth_map.inverse != self.inverse:
            if depth_map.shift != 0:
                raise Exception(f"Alignment parameters were computed in inverse={self.inverse} parameterization, but "
                                f"the given depth map with shift={depth_map.shift} is parameterized as "
                                f"inverse={depth_map.inverse}. Alignment cannot be performed.")
            depth_map = depth_map.invert()
        if inplace:
            data = depth_map.data
            data -= self.shift
            data /= self.scale
        else:
            data = (depth_map.data - self.shift) / self.scale
        range = ((depth_map.range[0] - self.shift) / self.scale, (depth_map.range[1] - self.shift) / self.scale) \
            if depth_map.range is not None else None
        return DepthMap(data, self.inverse, range, self.metric_scale, self.metric_shift)

    def apply_all(self, depth_maps: Iterable[Optional[DepthMap]]):
        for depth_map in depth_maps:
            if depth_map is None:
                yield None
                continue
            yield self.apply(depth_map)


def _union_mask(a: np.ma.MaskedArray, b: np.ma.MaskedArray):
    return np.logical_not(np.ma.mask_or(np.ma.getmaskarray(a), np.ma.getmaskarray(b), shrink=False))


def ensure_compatible_parameterization(prediction, ground_truth):
    if prediction.inverse != ground_truth.inverse:
        if ground_truth.shift != 0:
            raise Exception(f"Differing parameterization of prediction (inverse={prediction.inverse} and ground_truth "
                            f"(inverse={ground_truth.inverse}) with shift={ground_truth.shift}. Alignment cannot be "
                            f"performed.")
        ground_truth = ground_truth.invert()
    return ground_truth


def frame_aligning_scale_shift_lstsq(prediction: DepthMap, ground_truth: DepthMap):
    mask = _union_mask(prediction.data, ground_truth.data)
    X = prediction.data[mask][:, None]
    X = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=-1)
    coeffs, _, _, _ = np.linalg.lstsq(X, ground_truth.data[mask], rcond=None)
    if np.abs(coeffs[0]) <= 0.:
        return float("inf"), 0.
    scale = 1 / coeffs[0]
    shift = -coeffs[1] / coeffs[0]
    return scale, shift


def frame_aligning_scale_lstsq(prediction: DepthMap, ground_truth: DepthMap):
    mask = _union_mask(prediction.data, ground_truth.data)
    pred_values = prediction.data[mask] - prediction.shift
    gt_values = ground_truth.data[mask] - ground_truth.shift
    try:
        coeffs, _, _, _ = np.linalg.lstsq(pred_values[:, None], gt_values, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.array([1., 1., 1.])
        print('WARNING: Linalg SVD did not converge. Continue with scale of 1.')
    scale = 1 / coeffs[0]
    return scale


def frame_align_lstsq(prediction: DepthMap, ground_truth: DepthMap) -> Alignment:
    """
    Computes a scale and shift parameter such that `(prediction - shift)/scale` matches the ground truth optimally
    w.r.t. the mean squared error.
    If the shift of both prediction and ground truth is known (not None), the aligning shift is considered to be fixed,
    and will not be part of the alignment procedure (pure scale alignment).

    :param prediction: prediction map
    :param ground_truth: ground truth map

    :return: an instance of :class:`Alignment` containing computed parameters.
    """
    ground_truth = ensure_compatible_parameterization(prediction, ground_truth)
    if prediction.shift is not None and ground_truth.shift is not None:
        scale = frame_aligning_scale_lstsq(prediction, ground_truth)
        shift = prediction.shift - scale * ground_truth.shift
        return Alignment(ground_truth.inverse, scale, shift, ground_truth.scale, ground_truth.shift)
    scale, shift = frame_aligning_scale_shift_lstsq(prediction, ground_truth)
    return Alignment(ground_truth.inverse, scale, shift, ground_truth.scale, ground_truth.shift)

def align_prediction(prediction: Union[NDArray[np.floating], torch.Tensor], ground_truth: Union[NDArray[np.floating], torch.Tensor], 
                     valid_depth: Union[NDArray[np.floating], torch.Tensor], max_depth:float=80.)->Tuple[NDArray[np.floating], float, float]:
    """
    Aligns inverse-depth predictions to metric ground truth.

    Parameters
    ----------
    prediction : Tensor | NDArray[floating], shape (..., H, W)
        Inverse depth.
    ground_truth : Tensor | NDArray[floating], shape (..., H, W)
        Metric depth.
    valid_depth : Tensor | NDArray[bool], shape (..., H, W)
        Valid-pixel mask for `ground_truth`.
    max_depth : float
        Clip value applied after alignment.

    Returns
    -------
    aligned : same type as `prediction`
    inv_scale : float
    inv_shift : float
    """
    
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy()
    if isinstance(valid_depth, torch.Tensor):
        valid_depth = valid_depth.numpy()

    gt_mask = np.ma.array(ground_truth, mask=~valid_depth)
    prediction_mask = np.ma.array(prediction)
    # Raw predictions (have to be) are always inverse depth
    prediction_tmp = DepthMap(prediction_mask, inverse=True, range=None, scale=None, shift=None)
    gt_depth_tmp = DepthMap(gt_mask, inverse=False, range=None, scale=1, shift=0)
    # Calculate scale & shift for INVERSE depth
    alignment = frame_align_lstsq(prediction_tmp, gt_depth_tmp)
    scale, shift = alignment.scale, alignment.shift

    # Use scale & shift to align inverse depth
    align_pred = (prediction - shift) / scale

    # To make it metric we need to invert it again.
    # Avoid division by 0
    align_pred = np.where( align_pred == 0., 1e-4, align_pred) 
    # Clip to max depth 
    align_pred = np.clip(1. / align_pred, 0., max_depth)
    
    return align_pred, scale, shift