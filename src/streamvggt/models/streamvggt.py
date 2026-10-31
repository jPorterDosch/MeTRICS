import time

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class ScaleReadoutHead(nn.Module):
    """[..., 2C] last-layer scale token -> [...] log scale. The last layer is
    zero-init, so the readout starts at 0: a factorised output starts as the
    depth head's own (of the model WITH the token, which the extra token has
    already moved slightly off the pretrained one)."""

    def __init__(self, dim_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in), nn.Linear(dim_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.net(token.float()).squeeze(-1)


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        scale_token=False,
        scale_recurrent=False,
        scale_readout=False,
        scale_factorised=False,
    ):
        """scale_*: the optional scale token (depth_cond ScaleTokenCfg). Off:
        the pretrained architecture. scale_readout builds the log-scale MLP
        on the token; scale_factorised multiplies the depth by exp(readout)."""
        super().__init__()
        if scale_factorised and not scale_readout:
            raise ValueError("scale_factorised needs scale_readout")
        if scale_readout and not scale_token:
            raise ValueError("scale_readout needs scale_token")

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            scale_token=scale_token,
            scale_recurrent=scale_recurrent,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.scale_head = ScaleReadoutHead(2 * embed_dim) if scale_readout else None
        self.scale_factorised = scale_factorised

    def scale_log(self, aggregated_tokens_list) -> Optional[torch.Tensor]:
        """[B, S] log-scale readout of the last layer's scale token, or None
        without a readout."""
        if self.scale_head is None:
            return None
        return self.scale_head(aggregated_tokens_list[-1][:, :, self.aggregator.scale_token_idx])

    def scale_state(self, aggregated_tokens_list) -> Optional[torch.Tensor]:
        """[B, 2C] the newest frame's last-layer scale token: the next frame's
        `scale_prev` for a recurrent token, None otherwise."""
        if self.aggregator.scale_carry is None:
            return None
        return aggregated_tokens_list[-1][:, -1, self.aggregator.scale_token_idx]

    def apply_scale(self, depth: torch.Tensor, scale_log: Optional[torch.Tensor]) -> torch.Tensor:
        """depth [B, S, H, W, 1] times exp(readout) when factorised."""
        if not self.scale_factorised:
            return depth
        return depth * torch.exp(scale_log)[:, :, None, None, None]



    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        patch_tokens=None,
        depth_token_feats=None,
        depth_head_residuals=None,
    ):
        """Optional depth-conditioning inputs (all default None -> pretrained behavior):
            patch_tokens: precomputed aggregator.embed_patches output (encoder cache).
            depth_token_feats: [B,S,P_patch,C] added residually to the RGB patch
                tokens before the attention blocks (token injection, pre-KV-cache).
            depth_head_residuals: dict with an entry for EVERY head name
                ("depth", "point"): per-scale residual list, or None for a head
                that receives no conditioning. Missing keys are a plumbing bug
                and raise KeyError (head injection, post-cache).
        """
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        if depth_head_residuals is None:
            depth_head_residuals = {"depth": None, "point": None}
        if self.aggregator.scale_carry is not None:
            # every frame of the parallel forward is built before any output
            # exists: a recurrent token can only run through inference()
            raise ValueError("a recurrent scale token needs the one-frame-at-a-time forward")
        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images, patch_tokens=patch_tokens, injected_patch_feats=depth_token_feats
        )
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            scale_log = self.scale_log(aggregated_tokens_list)  # [B, S] or None
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    depth_residuals=depth_head_residuals["depth"],
                )
                predictions["depth"] = self.apply_scale(depth, scale_log)
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    depth_residuals=depth_head_residuals["point"],
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {}),
                    **({'scale_log': scale_log[:, s]}  # [B]
                    if scale_log is not None else {}),
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None,
                  depth_token_feats_list=None, depth_head_residuals_list=None,
                  frame_times_ms=None, window=None, patch_tokens_list=None,
                  depth_only=False):
        """Streaming inference with the causal KV cache.

        depth_token_feats_list / depth_head_residuals_list: optional per-frame
        depth-conditioning inputs (same semantics as in forward(), one entry per
        frame, entries may be None). Token feats enter BEFORE the KV cache.

        frame_times_ms: optional list; when given, one wall-clock millisecond
        entry per frame is APPENDED (aggregator + heads only -- the conditioning
        encoder runs in a separate upstream loop, see
        DepthConditionedStreamVGGT.inference, which adds its own per-frame cost
        into the same list). Attention is over a cache that grows with the frame
        index, so these are expected to ramp; frame 0 is the empty-cache case and
        also absorbs lazy CUDA init, so read it separately from the rest.

        window: keep only the last `window` frames' K/V after every step (the
        deployment's sliding cache, eval.streaming_window); None: unbounded.
        patch_tokens_list: optional per-frame precomputed embed_patches output
        ([B, P_patch, C], the encoder cache). Not under no_grad: the
        one-frame-at-a-time training forward (depth_cond train.window) runs
        through here with gradients.
        depth_only: run the depth head alone (and the scale readout): no
        camera, point or track head, and no such keys in the results. For the
        windowed training forward, whose losses and validation read depth only.
        """
        if depth_token_feats_list is not None and len(depth_token_feats_list) != len(frames):
            raise ValueError(
                f"depth_token_feats_list has {len(depth_token_feats_list)} entries "
                f"for {len(frames)} frames; must match (use None entries for "
                "unconditioned frames)"
            )
        if depth_head_residuals_list is not None and len(depth_head_residuals_list) != len(frames):
            raise ValueError(
                f"depth_head_residuals_list has {len(depth_head_residuals_list)} entries "
                f"for {len(frames)} frames; must match (use None entries for "
                "unconditioned frames)"
            )

        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth

        all_ress = []
        processed_frames = []
        scale_prev = None

        # CUDA events rather than a per-frame synchronize(): they are enqueued on
        # the stream like kernels, so the CPU keeps running ahead and the loop we
        # measure stays the loop that normally runs. A single sync after the clip
        # resolves them all. On CPU there is nothing to overlap, so perf_counter
        # is already exact.
        timing = frame_times_ms is not None
        timing_device = frames[0]["img"].device
        cuda_timing = timing and timing_device.type == "cuda"
        timing_stream = torch.cuda.current_stream(timing_device) if cuda_timing else None
        events, host_times = [], []

        for i, frame in enumerate(frames):
            if cuda_timing:
                events.append(
                    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                )
                events[-1][0].record(timing_stream)
            elif timing:
                t0 = time.perf_counter()
            # [B, 1, 3, H, W]: one frame of each of the B clips (unsqueeze(0)
            # read a training batch's [B, 3, H, W] as B frames of one clip)
            img = frame["img"]
            images = img[:, None] if img.dim() == 4 else img[None, None]
            token_feats = depth_token_feats_list[i] if depth_token_feats_list is not None else None
            head_residuals = depth_head_residuals_list[i] if depth_head_residuals_list is not None else None
            if head_residuals is None:
                head_residuals = {"depth": None, "point": None}
            aggregator_output = self.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i,
                patch_tokens=patch_tokens_list[i] if patch_tokens_list is not None else None,
                injected_patch_feats=token_feats,
                scale_prev=scale_prev,
            )

            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            scale_prev = self.scale_state(aggregated_tokens)
            if window is not None:
                past_key_values = truncate_kv(past_key_values, window)

            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None and not depth_only:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                scale_log = self.scale_log(aggregated_tokens)
                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx,
                        depth_residuals=head_residuals["depth"],
                    )
                    depth = self.apply_scale(depth, scale_log)[:, 0]
                    depth_conf = depth_conf[:, 0]

                if self.point_head is not None and not depth_only:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx,
                        depth_residuals=head_residuals["point"],
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None and not depth_only:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                **({'pts3d_in_other_view': pts3d,
                    'conf': pts3d_conf,
                    'camera_pose': camera_pose}
                if not depth_only else {}),
                'depth': depth,
                'depth_conf': depth_conf,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None and not depth_only else {}),
                **({'scale_log': scale_log[:, 0]}  # [B]
                if scale_log is not None else {}),
            })
            processed_frames.append(frame)
            if cuda_timing:
                events[-1][1].record(timing_stream)
            elif timing:
                host_times.append((time.perf_counter() - t0) * 1e3)
        
        if cuda_timing:
            # the one sync of the whole clip; the caller is about to read these
            # outputs, which would sync anyway
            torch.cuda.synchronize(timing_device)
            frame_times_ms.extend(s.elapsed_time(e) for s, e in events)
        elif timing:
            frame_times_ms.extend(host_times)

        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output


def truncate_kv(past_kv: list, window: int) -> list:
    """Keep the last `window` frame slices of every block's (K, V)
    ([B, heads, n_frames, P, head_dim]; dim 2 is the frame axis), the slice
    the export wrapper applies to its cache outputs."""
    return [
        None if kv is None else (kv[0][:, :, -window:], kv[1][:, :, -window:])
        for kv in past_kv
    ]
