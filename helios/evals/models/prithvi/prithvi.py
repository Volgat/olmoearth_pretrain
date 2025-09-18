
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
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample


class PrithviWrapper(nn.Module):

    def __init__(self, weights_path: Path,):
        super().__init__()

        with (weights_path / "config.json").open("r") as f:
            config = yaml.safe_load(f)["pretrained_cfg"]

        config["num_frames"] = 1

        self.model = PrithviMAE(**config)
        state_dict = torch.load(weights_path / "Prithvi_EO_V2_300M.pt", map_location="cpu")
        # discard fixed pos_embedding weight, following
        # https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/e4aabdc440c8ee703a749def8af5bf4700dee35b/inference.py#L362
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)
        self.image_resolution = config["img_size"]
        self.bands = config["bands"]
        self.inputs_to_prithvi = [Modality.SENTINEL2_L2A.band_order.index(b) for b in self.bands]

    def _process_modality_data(
        self, data: torch.Tensor, modality: str
    ) -> list[torch.Tensor]:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]
            modality: What modality data is

        Returns:
            list of tensors of shape [B, C, H, W]
        """
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []

        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")
            data_i = data_i[:, HELIOS_TO_SATLAS[modality], :, :]

            new_height = (
                self.patch_size if original_height == 1 else self.image_resolution
            )

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            if self.use_pretrained_normalizer:
                data_i = self.normalize(data_i, modality)
            data_list.append(data_i)

        return data_list

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> tuple[list[torch.Tensor], str]:
        """Prepare input for the Prithvi model from MaskedHeliosSample."""
        if len(masked_helios_sample.modalities) != 1:
            raise RuntimeError(
                f"Prithvi only supports one modality. Received {len(masked_helios_sample.modalities)}: {masked_helios_sample.modalities}"
            )
        modality = masked_helios_sample.modalities[0]
        if modality != Modality.SENTINEL2_L2A.name:
            raise RuntimeError(f"Prithvi only supports Sentinel2_L2A. Received {modality}")

        data = getattr(masked_helios_sample, modality)
        return self._process_modality_data(data, modality), modality

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the satlas model."""
        processed_inputs, modality = self.prepare_input(masked_helios_sample)
        outputs_list: list[torch.Tensor] = []
        for per_t_input in processed_inputs:
            output = self.model(per_t_input)[-1]
            # output shape for atto: (bsz, 320, 7, 7)
            # output shape for tiny: (bsz, 768, 6, 6)
            if not spatial_pool:
                # then we don't want to keep the spatial dimensions
                output = output.mean(dim=-1).mean(dim=-1)
            else:
                output = rearrange(output, "b c h w -> b h w c")
            outputs_list.append(output.unsqueeze(0))

        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(outputs_list, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(outputs_list, dim=0), dim=0)[0]
        return output_features

    def forward(self, s2=None,  do_pool=True, temporal_pooling: str = "mean"):
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
