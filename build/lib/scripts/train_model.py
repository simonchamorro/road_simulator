
import numpy as np

from scripts.models import ConvNet
from roadsimulator.models.utils import get_datasets

net = ConvNet()
train_x, train_y, val_x, val_y, _, _ = get_datasets('data', n_images=1000)
import pdb; pdb.set_trace()


