
import numpy as np
import torch

from roadsimulator.models.convnet import ConvNet
from roadsimulator.models.utils import get_datasets, get_device


SEED = 1
torch.manual_seed(SEED)

device = get_device()
net = ConvNet(device).to(device)
train_dl, val_dl = get_datasets('data', n_images=60000, seed=SEED, batch_size=1000)
net.train_model(train_dl, val_dl, epochs=100)