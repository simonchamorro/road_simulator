
import numpy as np
import torch

from roadsimulator.models.convnet import ConvNet
from roadsimulator.models.utils import get_datasets, get_device


SEED = 1
torch.manual_seed(SEED)
num_epochs = 500

device = get_device()
print('Creating model')
net = ConvNet(device).to(device)
print('Loading dataset')
train_dl, val_dl = get_datasets('data', n_images=120000, seed=SEED, batch_size=2000)
print('Training for %.0f epochs' % (num_epochs))
net.train_model(train_dl, val_dl, epochs=num_epochs)