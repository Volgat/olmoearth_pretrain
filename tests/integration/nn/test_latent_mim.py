"""Test LatentMIM with loss."""

import logging

import pytest
import torch

from helios.data.constants import Modality, ModalitySpec
from helios.data.transform import TransformConfig
from helios.nn.flexihelios import Encoder, Predictor
from helios.nn.latent_mim import LatentMIM

# from helios.train.loss import PatchDiscriminationLoss
from helios.train.masking import MaskedHeliosSample

# from einops import rearrange

# from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


@pytest.fixture
def modality_band_set_len_and_total_bands(
    supported_modalities: list[ModalitySpec],
) -> dict[str, tuple[int, int]]:
    """Get the number of band sets and total bands for each modality.

    Returns:
        Dictionary mapping modality name to tuple of (num_band_sets, total_bands)
    """
    return {
        modality.name: (
            len(modality.band_sets),
            modality.num_bands,
        )
        for modality in supported_modalities
    }


def test_latentmim_with_loss(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """Test the full end to end forward pass of the model with an exit configuration and loss."""
    # Define supported modalities
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
        modality_band_set_len_and_total_bands["sentinel2_l2a"]
    )
    latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
        "latlon"
    ]
    B, H, W, T, C = masked_sample_dict["sentinel2_l2a"].shape
    x = MaskedHeliosSample(**masked_sample_dict)

    patch_size = 4
    # Shared constants for encoder and predictor
    MAX_PATCH_SIZE = 8
    NUM_HEADS = 2
    MLP_RATIO = 4.0
    MAX_SEQ_LENGTH = 12
    DEPTH = 2
    DROP_PATH = 0.1
    ENCODER_EMBEDDING_SIZE = 16
    DECODER_EMBEDDING_SIZE = 16
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        min_patch_size=1,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        max_sequence_length=MAX_SEQ_LENGTH,
        use_channel_embs=True,
        depth=DEPTH,
        drop_path=DROP_PATH,
    )
    predictor = Predictor(
        supported_modalities=supported_modalities,
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQ_LENGTH,
        drop_path=DROP_PATH,
        learnable_channel_embeddings=True,
    )
    transform = TransformConfig(transform_type="no_transform").build()
    latentmim = LatentMIM(encoder, predictor, transform)
    output = latentmim.forward(x, patch_size)
    # output = predictor.forward(output, x.timestamps, patch_size, input_res)
    patched_H = H // patch_size
    patched_W = W // patch_size
    assert output.sentinel2_l2a is not None
    assert output.sentinel2_l2a_mask is not None
    assert output.latlon is not None
    assert output.latlon_mask is not None
    assert output.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
        predictor.output_embedding_size,
    )
    assert output.sentinel2_l2a_mask.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
    )
    assert output.latlon.shape == (
        B,
        latlon_num_band_sets,
        predictor.output_embedding_size,
    )
    assert output.latlon_mask.shape == (
        B,
        latlon_num_band_sets,
    )
    assert output.worldcover is not None
    assert output.worldcover_mask is not None
    assert output.worldcover.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
        predictor.output_embedding_size,
    )
    assert output.worldcover_mask.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
    )

    # this reflects the forward_model function in latentmim
    # Skip backward pass test for now
    # with torch.no_grad():
    #     logger.info("target encoder running here")
    #     target_output = latentmim.target_encoder.forward(
    #         x.unmask(),
    #         patch_size=patch_size,
    #         token_exit_cfg={
    #             modality: 0 for modality in latentmim.encoder.supported_modality_names
    #         },
    #     )
    # loss_fn.compute(output, target_output).backward()

    # Skip gradient checks for target_encoder as well
    # for name, param in latentmim.target_encoder.named_parameters():
    #     assert param.grad is None, name


def test_latentmim_with_missing_modalities_in_sample(
    masked_sample_dict: dict[str, torch.Tensor], set_random_seeds: None
) -> None:
    """Test the full end to end forward pass of the model with an exit configuration and loss."""
    # supported_modalities = [
    #     Modality.SENTINEL2_L2A,
    #     Modality.LATLON,
    #     Modality.WORLDCOVER,
    # ]
    # worldcover_mask = masked_sample_dict["worldcover_mask"]
    # worldcover_mask[0] = MaskValue.MISSING.value
    # worldcover_mask[3] = MaskValue.MISSING.value
    # masked_sample_dict["worldcover_mask"] = worldcover_mask
    # x = MaskedHeliosSample(**masked_sample_dict)

    # patch_size = 4
    # # Shared constants for encoder and predictor
    # MAX_PATCH_SIZE = 8
    # NUM_HEADS = 2
    # MLP_RATIO = 4.0
    # MAX_SEQ_LENGTH = 12
    # DEPTH = 2
    # DROP_PATH = 0.1
    # ENCODER_EMBEDDING_SIZE = 16
    # DECODER_EMBEDDING_SIZE = 16
    # encoder = Encoder(
    #     supported_modalities=supported_modalities,
    #     embedding_size=ENCODER_EMBEDDING_SIZE,
    #     max_patch_size=MAX_PATCH_SIZE,
    #     num_heads=NUM_HEADS,
    #     mlp_ratio=MLP_RATIO,
    #     max_sequence_length=MAX_SEQ_LENGTH,
    #     use_channel_embs=True,
    #     depth=DEPTH,
    #     drop_path=DROP_PATH,
    # )
    # predictor = Predictor(
    #     supported_modalities=supported_modalities,
    #     encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
    #     decoder_embedding_size=DECODER_EMBEDDING_SIZE,
    #     depth=DEPTH,
    #     mlp_ratio=MLP_RATIO,
    #     num_heads=NUM_HEADS,
    #     max_sequence_length=MAX_SEQ_LENGTH,
    #     drop_path=DROP_PATH,
    #     learnable_channel_embeddings=True,
    # )
    # transform = TransformConfig(transform_type="no_transform").build()
    # latentmim = LatentMIM(encoder, predictor, transform)
    # output = latentmim.forward(x, patch_size)
    # loss_fn = PatchDiscriminationLoss()
    # with torch.no_grad():
    #     logger.info("target encoder running here")
    #     target_output = latentmim.target_encoder.forward(
    #         x.unmask(),
    #         patch_size=patch_size,
    #         token_exit_cfg={
    #             modality: 0 for modality in latentmim.encoder.supported_modality_names
    #         },
    #     )
    #     target_output_no_missing = latentmim.target_encoder.forward(
    #         x.unmask(),
    #         patch_size=patch_size,
    #         token_exit_cfg={
    #             modality: 0 for modality in latentmim.encoder.supported_modality_names
    #         },
    #     )
    # # check if the worldcover output is empty
    # for i in range(4):
    #     logger.info(f"worldcover numel output shape{i}: {output.worldcover[i].numel()}")
    #     # assert output.worldcover[i].numel() == 0
    # loss_missing = loss_fn.compute(output, target_output)
    # # loss_no_missing = loss_fn.compute(output_no_missing, target_output_no_missing)
    # # logger.info(f"loss_missing: {loss_missing}")
    # # logger.info(f"loss_no_missing: {loss_no_missing}")
    # # assert loss_missing == loss_no_missing
    # # loss_no_missing.backward()
    # loss_missing.backward()
    # assert False

    # I wanto check if the loss value would be different with the missing worldcover samples and
    # without them. then try the same thing with all the modalities missing.
    # check many different combinations of masks to double check


def test_latent_mim_forward() -> None:
    """Test that LatentMIM forward pass works."""
    # This test is not implemented yet
    pass
