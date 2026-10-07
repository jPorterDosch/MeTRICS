import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class VideoPreprocessor:
    def __init__(self, input_size=384, device="cpu", ensure_multiple_of=14, keep_aspect_ratio=False, resize_method="lower_bound"):
        self.input_size = input_size
        self.device = device
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.keep_aspect_ratio = keep_aspect_ratio
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    def _to_tensor(self, x):
        '''
        Handels numpy.NDArray or List input and rearranges it to fit with numpy shape. 
        
        Parameters
        ----------
        x : torch.Tensor | np.array[float] | list[]
            Tensor, Array or List of shape (C, H, W) or Video (T, C, H, W) or batch of Videos (B, T, C, H, W)
        
        Returns
        -------
        x : torch.Tensor
            Rearranged torch.Tensor
            shape (C, H, W) or Video (T, C, H, W) or batch of Videos (B, T, C, H, W)
        '''
        if isinstance(x, list):
            x = torch.stack([self._to_tensor(xx) for xx in x], dim=0)
            if x.ndim == 6:  # (B,1,T,3,H,W)
                x = x.squeeze(1)
            return x

        if isinstance(x, torch.Tensor):
            if x.ndim == 3:
                x = rearrange(x, 'c h w -> 1 1 c h w')
            elif x.ndim == 4:
                x = rearrange(x, 't c h w -> 1 t c h w')
            elif x.ndim == 5:
                pass
            else:
                raise ValueError(f"Unsupported tensor shape {x.shape}")
            return x.float().to(self.device)

        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = torch.from_numpy(rearrange(x, 'h w c -> 1 1 c h w'))
            elif x.ndim == 4:
                x = torch.from_numpy(rearrange(x, 't h w c -> 1 t c h w'))
            elif x.ndim == 5:
                x = torch.from_numpy(rearrange(x, 'b t h w c -> b t c h w'))
            else:
                raise ValueError(f"Unsupported numpy shape {x.shape}")
            return x.float().to(self.device)

        raise TypeError("Input must be numpy array, torch tensor, or list thereof.")
    
    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.input_size / height
        scale_width = self.input_size / width

        if self.keep_aspect_ratio:
            if self.resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.resize_method} not implemented")

        if self.resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.input_size)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.input_size)
        elif self.resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.input_size)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.input_size)
        elif self.resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.resize_method} not implemented")

        return (new_width, new_height)

    def preprocess(self, x, mode='bicubic'):
        """
        Preprocessing for the network. It can handle individual images, Videos and batches of Videos. 
        It can also handle numpy or torch.tensor as input. Use bicubic mode for images and nearest for depths.

        Parameters
        ----------
        x : torch.Tensor | np.array[float] | list[]
            Tensor, Array or List of shape (C, H, W) or Video (T, C, H, W) or batch of Videos (B, T, C, H, W)
        mode: {'bicubic', 'nearest'}
            Interpolation mode for torch.functional.interpolate. Use 'bicubic' for images and 'nearest' for depths
        
        Returns
        -------
        x : torch.Tensor
            preprocessed frames. Shape (B, T, C, H, W)
            If only a single frame or a video is given than B, T are 1.
        """
        x = self._to_tensor(x)
        if x.max() > 1.0:
            x = x / 255.0

        B, T, C, H, W = x.shape
        new_w, new_h = self.get_size(W, H)

        # Nur resizen, wenn nötig
        if (H, W) != (new_h, new_w):
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=False)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

        # Normalisieren
        x = (x - self.mean) / self.std
        return x