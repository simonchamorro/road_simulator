import os
import torch
import numpy as np

from tqdm import tqdm
from scipy.misc import imread
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def get_device():
    if torch.cuda.is_available():  
      dev = "cuda:0" 
    else:  
      dev = "cpu"  
    device = torch.device(dev)

class ImgDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = np.swapaxes(self.x[idx], 0, 1)
        x = np.swapaxes(x, 0, 2)
        y = self.y[idx].index(1)
        return torch.from_numpy(x.astype(np.float32)), y

def get_images(paths, n_images=1000):

    if isinstance(paths, str):
        paths = [paths]

    images = []
    labels = []

    n = 0
    for path in paths:
        if n > n_images: break
        print(path)
        for image_file in tqdm(os.listdir(path)):
            if n > n_images: break
            if '.jpg' not in image_file: continue
            try:
                img = imread(os.path.join(path, image_file))
                itc = image_file[:-4].split('_')
                lbl = [float(itc[3]), float(itc[5])]
                if img is not None:
                    # Normalize
                    images.append(img[:, :].astype('float32') / 255.)
                    labels.append(lbl)
                    n += 1
            except Exception as e:
                pass
    
    # np arrays dont fit in memory for large datasets
    # images = np.array(images)
    # labels = np.array(labels)

    return images, labels


def shuffle_data(X, Y):
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]


def from_continue_to_discrete(Y):

    def one_hot(i, n):
        t = [0 for j in range(n)]
        t[i] = 1
        return t

    # transform direction angle into a 5 dimensions array of 0 and 1
    arr_discr = [i for i in range(-2, 3)]
    threshs = [-100000, -0.7, -0.25, 0.25, 0.7, 100000]
    ns = [0 for i in range(-2, 3)]

    Y_new = []
    for elt in Y:
        e = elt[1]
        for i in range(len(threshs)):
            if threshs[i] < e < threshs[i+1]:
                t = one_hot(i, 5)
                Y_new.append(t)
                ns[i] += 1
                break

    return Y_new, ns


def equilibrate_dataset(X, Y, ns):
    m = min(ns)

    X_by_label = [[] for i in ns]
    Y_by_label = [[] for i in ns]

    for i in range(len(Y)):
        index = np.argmax(Y[i])
        Y_by_label[index].append(Y[i])
        X_by_label[index].append(X[i])

    all_X = []
    all_Y = []

    for i in range(len(ns)):
        x, y = np.array(X_by_label[i]), np.array(Y_by_label[i])
        x, y = shuffle_data(x, y)
        x = x[:m]
        y = y[:m]
        all_X.extend(x)
        all_Y.extend(y)

    return all_X, all_Y


def get_datasets(paths, n_images, seed=0, batch_size=100):

    X, Y = get_images(paths, n_images=n_images)

    # A classifier is better (when talking about performance) than a simple
    # sigmoid
    Y, ns = from_continue_to_discrete(Y)

    # Equilibrate the dataset 
    # TODO fix this for memory efficiency 
    # for now it breaks on big datasets
    # X, Y = equilibrate_dataset(X, Y, ns)

    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, \
                                                          stratify=Y, random_state=seed)
    train_ds = ImgDataset(x_train, y_train)
    valid_ds = ImgDataset(x_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    return train_dl, val_dl
