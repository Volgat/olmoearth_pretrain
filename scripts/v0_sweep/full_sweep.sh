#!/bin/bash

python scripts/v0_sweep/galileo.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
python scripts/v0_sweep/latent_mim.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
python scripts/v0_sweep/contrastive_latent_mim.py launch test_sweep_gal ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=8
