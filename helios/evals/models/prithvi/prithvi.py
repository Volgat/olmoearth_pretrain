
import logging
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange, repeat
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block
from helios.evals.models.prithvi.prithvi_mae import PrithviMAE
from helios.data.constants import Modality


class PrithviWrapper(nn.Module):

    def __init__(self, weights_path: Path, do_pool=True, temporal_pooling: str = "mean"):
        super().__init__()

        with (weights_path / "prithvi/config.json").open("r") as f:
            config = yaml.safe_load(f)["pretrained_cfg"]

        config["num_frames"] = 1

        self.model = PrithviMAE(**config)
        state_dict = torch.load(weights_path / "prithvi/Prithvi_EO_V2_300M.pt", map_location="cpu")
        # discard fixed pos_embedding weight, following
        # https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/e4aabdc440c8ee703a749def8af5bf4700dee35b/inference.py#L362
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)
        self.image_resolution = config["img_size"]
        self.grid_size = int(config["img_size"] // config["patch_size"][-1])
        self.bands = config["bands"]

        self.inputs_to_prithvi = [Modality.SENTINEL2_L2A.band_order.index(b) for b in self.bands]
        self.do_pool = do_pool
        if temporal_pooling not in ["mean", "max"]:
            raise ValueError(
                f"Expected temporal_pooling to be in ['mean', 'max'], got {temporal_pooling}"
            )
        self.temporal_pooling = temporal_pooling
        self.dim = config["embed_dim"]

    def resize(self, images):
        images = F.interpolate(
            images,
            size=(self.image_resolution, self.image_resolution),
            mode="bilinear",
            align_corners=False,
        )
        return images

    def preproccess(self, images):
        if len(images.shape) == 5:
            # take the mean along the temporal dimension
            images = torch.mean(images, dim=2)
        images = rearrange(images, "b h w c -> b c h w")
        assert images.shape[1] == 13
        images = images[:, self.inputs_to_prithvi, :, :]
        images = self.resize(images)  # (bsz, C, H, W)
        return repeat(images, "b c h w -> b c t h w", t=1)

    def forward(self, s2=None, s1=None, months=None):
        if s2 is None:
            raise ValueError("S2 can't be None for Prithvi")

        if len(s2.shape) == 5:
            outputs_l: List[torch.Tensor] = []
            for timestep in range(s2.shape[3]):
                image = self.preproccess(s2[:, :, :, timestep])
                output = self.model.forward_features(image)[-1]
                # following
                # https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/prithvi_mae.py#L449
                # we remove the class token. This is also the approach they
                # take for classification: https://github.com/IBM/terratorch/blob/main/terratorch/models/scalar_output_model.py#L19
                output = output[:, 1:, :]
                # output shape: (bsz, num_tokens, dim)
                if self.do_pool:
                    output = output.mean(dim=1)
                outputs_l.append(output)
            outputs_t = torch.stack(outputs_l, dim=-1)  # b h w d t
            if self.temporal_pooling == "mean":
                return outputs_t.mean(dim=-1)
            else:
                return torch.amax(outputs_t, dim=-1)
        else:
            s2 = self.preproccess(s2)
            output = self.model.forward_features(s2)[-1]
            output = output[:, 1:, :]
            if self.do_pool:
                return output.mean(dim=1)
            else:
                return output
