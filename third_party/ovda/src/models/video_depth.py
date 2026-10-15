# Copyright (c) 2026 Heidelberg University.
#
# This source code is licensed under the Non-Commercial Software License 
# Heidelberg University Version 1.0 (NC-SA-UHDV1.0)
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

from .utils.preprocessing import VideoPreprocessor
from .dinov2 import DINOv2
from .modules.dpt_temporal import DPTHeadTemporalCrossAtt
from tqdm import tqdm
import time
import math
import cv2
import imageio.v3 as iio
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

class onlineVideoDepthAnything(nn.Module):
    def __init__(   
                self,
                encoder='vitl',
                features=256, 
                out_channels=[256, 512, 1024, 1024], 
                use_bn=False, 
                use_clstoken=False,
                cache_size=16,
                pe='ape',
                use_xformers=True,
                ):
        """
        oVDA Model
        -----------------
        Initialises the oVDA model. The model is written in a way that it is usable for edge devices.

        Parameters
        ----------
        encoder : {vitl, vits}
            Decides which encoder backbone will be used. Must be consistent with loaded checkpoint
        features : int
            number of features for the temporal head
        out_channels : list[int]
            List of the ouput channel size of the dpt head
        use_bn : bool
            Defines if batch norm is used
        use_clstoken : bool
            Defines if the class token of the DinoV2 backbone is used
        cache_size : int
            Sets the size of the Cache
        pe : {'ape', 'rope'}
            Selects mode of the positional encodings within the Motion Modules
        use_xformers : bool
            If XFormers should be used. For edge device testing this can not be used. 
        """
        super(onlineVideoDepthAnything, self).__init__()
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.cache_size = cache_size
        use_xformers = use_xformers
        self.pretrained = DINOv2(model_name=encoder)
        
        self.head = DPTHeadTemporalCrossAtt(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, cache_size=cache_size,
                                                pe=pe, use_xformers=use_xformers)

    @torch.no_grad()
    def forward(self, x, input_cache, mask_indices, input_position):
        """
        Model forward
        -----------------
        Implements the forward of a single input image, without handeling the cache. 

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (Batch, 1, Channels, Height, Width).
        input_cache : dict {torch.Tensor}
            Dictionary of torch.Tensors containing the cached frames
        mask_indices : torch.Tensor
            Tensor of indices which should be masked and not used for the Crossattention. 
            This is only usefull for initialisation, when the cache is not full yet
        input_position : torch.Tensor
            Tensor of int defining the position the current frame should be inputted. 
            When the cache is filled this is constant.

        Returns
        -------
        depth : torch.Tensor
            Predicted depth of size: [1, H, W]
        output_cache: dict {torch.Tensor}
            Output cache with the newly inputed latend.
    """
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth, output_cache = self.head(features, patch_h, patch_w, T,
                                        input_cache=input_cache,
                                        mask_indices=mask_indices,
                                        input_position=input_position)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth, output_cache # We cannot directly use .squeeze(1).unflatten(0, (1, 1)) on output depths because tensorrt does not work with that
    
    def setup_cache(self, h: int, w: int, device: str):
        '''
        Prepare cache
        -------------
        The Cache has dimension: [Number of Temporal Transformer 3DModules, Number of Temporal Attention, batch_size * h * w, Cache_size, Channels]
        Batch_size is in our case always 1 Frame; h and w are depending on the preprocessing resolution which is defined as: 
        Input_h / 14 and Input_w / 14 for Cache1 and Cache3. For the other two caches they are resized to half this resolution (c2) and double this 
        Resolution (c4). 

        Parameters
        ----------
        h : int
            Height of the input image. Used to determine the cache resolutions
        w : int 
            Width of the input image. Used to determine the cache resolutions
        device : str
            torch device to put the cache on. 

        Returns
        -------
        input_cache : dict {torch.Tensor}
            Dictionary of torch.Tensors containing the cached frames           
        '''
        c1 = int(h/14 * w/14)
        c2 = int(math.ceil(h/(14*2)) * math.ceil(w/(14*2)))
        c3 = int(h/14 * w/14)
        c4 = int(math.ceil((h * 2)/14) * math.ceil((w * 2)/14))

        input_cache1 = torch.zeros(1, 2, c1, self.cache_size, 192).to(device)
        input_cache2 = torch.zeros(1, 2, c2, self.cache_size, 384).to(device)
        input_cache3 = torch.zeros(1, 2, c3, self.cache_size, 64).to(device)
        input_cache4 = torch.zeros(1, 2, c4, self.cache_size, 64).to(device)

        input_cache = {'c1': input_cache1, 'c2': input_cache2, 'c3': input_cache3, 'c4': input_cache4}
        return input_cache
    
    @torch.no_grad()
    def infer_video_depth(self, frames, device, preprocess_device, input_size=518, fp32=False, print_process_res=False, output_raw=False, 
                          return_process_res=False):
        """
        oVDA Forward for Video
        -----------------
        Implements the oVDA depth prediction for a complete video or a batch of Videos. 
        It handles preprocessing of the video and handling of the cache during inference. 

        Parameters
        ----------
        frames : torch.Tensor | np.ndarray
            Input data of shape (Batch, Time, Channels, H, W) or (Time, Channels, H, W)
        device : str
            torch device string of type: 'cuda:0' defining the cuda device to run on or 'cpu' 
        preprocess_device : str
            torch device string defining the preprocess device. Can be a different one. 
        input_size : int, default=518
            Defining the rought resolution for processing. The exact Resolution will be automatically calculated.
        fp32 : bool, default=False
            Defining if the model is run in fp32 or fp16 (if False). Since fp32 is only marginally better, we recommend to use fp16.
        print_process_res : bool, default=False
            Prints out the resolution the preprocessing has resized the original input to. 
        output_raw : bool, default=False
            Returns the original prediction of oVDA. Will be in the resolution of the preprocessed input video. If set to False, 
            the depth prediction is resized to the original input video size
        return_process_res : bool, default=False
            Returns predictions and the process resolution. This is used for keeping track for downstream processing. 

        Returns
        -------
        out_depth : np.ndarray
            predicted depth of shape (B, T, H, W)
        """
        self.to(device)
        # Preprocessing of frames
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # VDA recommendet to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14
        
        pre = VideoPreprocessor(input_size=input_size, device=preprocess_device, ensure_multiple_of=14, keep_aspect_ratio=True, resize_method='lower_bound')
        prepared_frames = pre.preprocess(frames)

        b, t, c, h, w = prepared_frames.size()
        if print_process_res:
            print(f'Inferring Video Depth at res: {h}x{w}')

        # Handle multiple videos
        if output_raw:
            out_depth = np.zeros((b, t, h, w))
        else:
            out_depth = np.zeros((b, t, frame_height, frame_width))
        
        print_resize_warining = False
        
        for batch in range(b):
            
            cache_size = 0
            mask_indices = torch.tensor(list(range(1, self.cache_size))).to(device)
            input_position = torch.tensor([0, 1]).to(device)

            input_cache = self.setup_cache(h, w, device)
            if not fp32:
                for key in input_cache:
                    input_cache[key] = input_cache[key].half()
                    self.half()
                    prepared_frames = prepared_frames.half()

            # Predict depths 
            depths = []
            times = []
            with torch.no_grad():
                for i in tqdm(range(prepared_frames.shape[1])):
                    input_frame = prepared_frames[:, i, :, :, :].unsqueeze(dim=1).to(device)
                    depth_pred, output_cache = self.forward(
                                                                input_frame,
                                                                input_cache=input_cache,
                                                                mask_indices=mask_indices,
                                                                input_position=input_position,
                                                            )
                    depth_pred = depth_pred.squeeze(1).unflatten(0, (1, 1))
                    depth_pred = depth_pred.squeeze(dim=0)
                    depths.append(depth_pred.cpu())

                    # Update Cache
                    if cache_size == self.cache_size - 1:
                        for key in input_cache:
                            input_cache[key] = torch.cat([input_cache[key][:, :, :, 1:cache_size, :], output_cache[key], torch.zeros_like(output_cache[key])], dim=3)
                    else:
                        for key in input_cache:
                            input_cache[key][:, :, :, cache_size, :] = output_cache[key][:, :, :, 0, :]

                    cache_size += 1
                    cache_size = min(cache_size, self.cache_size - 1)

                    remaining = list(range(cache_size, self.cache_size))
                    padding = [self.cache_size - 1] * (cache_size - 1)
                    
                    mask_indices = torch.tensor(remaining + padding).to(device)
                    input_position = torch.tensor([cache_size, cache_size]).to(device)

                # The depths are in the process resolution.  
                depths = torch.stack(depths, dim=1).float().numpy()
                if output_raw: 
                    out_depth[batch] = depths[batch]
                else:
                    if depths[0][0].shape != (frame_height, frame_width):
                        h, w = depths[0][0].shape
                        gh, gw = (frame_height, frame_width)
                        if not print_resize_warining:
                            print(f'WARNING: Height and width are different: Prediction {h}, {w}, GroundTruth {gh}, {gw}! Resize prediction. ')
                            print_resize_warining = True
                        print('Resizing ...')

                    out_depth[batch] = F.interpolate(torch.from_numpy(depths), size=(frame_height, frame_width), mode='bilinear', align_corners=True)[batch, :, :, :].numpy()
        
        if not return_process_res:
            return out_depth
        else: 
            return out_depth, (h, w)


    @torch.no_grad()
    def lazy_forward(self, image_list, output_dir, device: str, preprocess_device: str, input_size: int = 518,
                    fp32: bool = False, print_process_res : bool = False, output_raw: bool = False,
                    offset: float = 1., save_rgb: bool = False):
        """
        oVDA lazy forward
        -----------------
        Runs oVDA without loading the data complete into the RAM. It loads every frame individually and writes the output directly 
        to the disc. This is used for very long videos. 

        Parameters
        ----------
        image_list : list[str]
            A list of all image-paths or the path to the folder containing images of the video. They will be sorted with natsort.
        output_dir : str
            Path to the output directory. 
        device : str
            torch device string of type: 'cuda:0' defining the cuda device to run on or 'cpu' 
        preprocess_device : str
            torch device string defining the preprocess device. Can be a different one. 
        input_size : int, default=518
            Defining the rought resolution for processing. The exact Resolution will be automatically calculated.
        fp32 : bool, default=False
            Defining if the model is run in fp32 or fp16 (if False). Since fp32 is only marginally better, we recommend to use fp16.
        print_process_res : bool, default=False
            Prints out the resolution the preprocessing has resized the original input to. 
        output_raw : bool, default=False
            If set to true the output will be saved as a seperate prediction per image (.tiff). Note: This might take a lot of 
            free disc space.
            If set to False, it will be saved as colored depth video (.mp4).
        offset : float; default=1.
            The offset to the maximal and minimal depth of the first frame. This will set the vmin and vmax for the entire video.
            In case predictions are not within this range they are marked pink
        save_rgb : bool; default=False
            Saves not only the depth prediction but also the rgb video
        """
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(image_list, str):
            img_str = image_list
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
            image_list = []
            for ext in image_extensions:
                image_list.extend(glob.glob(os.path.join(img_str, ext)))
        
        image_list = natsorted(image_list)
        pre = VideoPreprocessor(input_size=input_size, device=preprocess_device,
                                ensure_multiple_of=14, keep_aspect_ratio=True, resize_method='lower_bound')

        # Load first frame to get ref values
        first_frame = iio.imread(image_list[0])
        pre_first = pre.preprocess(np.expand_dims(first_frame, 0)).to(device).half() if not fp32 else pre.preprocess(np.expand_dims(first_frame, 0)).to(device)
        b, t, c, h, w = pre_first.size()
        
        if print_process_res:
            print(f'Inferring Video Depth at res: {h}x{w}')
        
        self.to(device)
        if not fp32:
            self.half()
            pre_first = pre_first.half()

        # Initialize cache
        cache_size = 0
        mask_indices = torch.tensor(list(range(1, self.cache_size))).to(device)
        input_position = torch.tensor([0, 1]).to(device)

        input_cache = self.setup_cache(h, w, device)
        
        if not fp32:
            for key in input_cache:
                input_cache[key] = input_cache[key].half()
                self.half()

        # Determine depth min/max from first frame
        depth_first, _ = self.forward(pre_first, input_cache, mask_indices, input_position)
        depth_first = depth_first.squeeze(1).unflatten(0, (1, 1))
        dmin, dmax = depth_first.min().item(), depth_first.max().item()
        dmin, dmax = dmin - offset, dmax + offset

        # Video writer setup (if not raw)
        writer = None
        frame_h, frame_w = first_frame.shape[:2]
        if not output_raw:
            out_path = os.path.join(output_dir, "depth_colored.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, 20.0, (frame_w, frame_h))
        
        if save_rgb:
            out_path = os.path.join(output_dir, "rgb.mp4")
            fourcc_rgb = cv2.VideoWriter_fourcc(*'mp4v')
            writer_rgb = cv2.VideoWriter(out_path, fourcc_rgb, 20.0, (frame_w, frame_h))
            

        for idx, img_path in enumerate(tqdm(image_list)):
            frame = iio.imread(img_path)
            proc = pre.preprocess(np.expand_dims(frame, 0))
            proc = proc.to(device).half() if not fp32 else proc.to(device)

            if not fp32:
                proc = proc.half()

            depth_pred, output_cache = self.forward(proc, input_cache, mask_indices, input_position)
            depth_pred = depth_pred.squeeze(1).unflatten(0, (1, 1))
            depth = depth_pred.squeeze().cpu().numpy()

            # Cache update
            if cache_size == self.cache_size - 1:
                for key in input_cache:
                    input_cache[key] = torch.cat([input_cache[key][:, :, :, 1:cache_size, :], output_cache[key], torch.zeros_like(output_cache[key])], dim=3)
            else:
                for key in input_cache:
                    input_cache[key][:, :, :, cache_size, :] = output_cache[key][:, :, :, 0, :]
            
            cache_size = min(cache_size + 1, self.cache_size - 1)
            remaining = list(range(cache_size, self.cache_size))
            padding = [self.cache_size - 1] * (cache_size - 1)
            mask_indices = torch.tensor(remaining + padding, device=device)
            input_position = torch.tensor([cache_size, cache_size], device=device)

            if output_raw:
                out_file = os.path.join(output_dir, f"{idx:05d}.tiff")
                iio.imwrite(out_file, depth.astype("float32"))
            else:
                # Normalize + colorize
                depth_clipped = np.clip(depth, dmin, dmax)
                norm = (depth_clipped - dmin) / (dmax - dmin + 1e-8)
                colored = cm.Spectral(norm)[..., :3]
                mask_oob = (depth < dmin) | (depth > dmax)
                colored[mask_oob] = [1.0, 0.0, 1.0]  # pink for OOR
                colored = (colored * 255).astype(np.uint8)
                colored = cv2.resize(colored, (frame_w, frame_h))
                writer.write(cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
                writer_rgb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if writer:
            writer.release()
        if writer_rgb:
            writer_rgb.release()
        
        print(f"Done. Output saved to {output_dir}")