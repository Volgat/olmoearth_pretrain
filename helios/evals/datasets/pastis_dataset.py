"""PASTIS dataset class."""
# Dataset is between September 2018 to November 2019

# load this torch object: /weka/dfive-default/presto_eval_sets/pastis/pastis_train.pt
# print out the shape of the object inside

import torch

pastis_train = torch.load("/weka/dfive-default/presto_eval_sets/pastis/pastis_train.pt")

for item in pastis_train:
    print(item, pastis_train[item].shape)


# images torch.Size([5820, 12, 13, 64, 64])
# months torch.Size([5820, 12])
# targets torch.Size([5820, 64, 64])

# already subset to 64 * 64 images
