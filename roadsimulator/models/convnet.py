
import os
import torch
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error


class ConvNet(torch.nn.Module) :
    def __init__(self, device, input_shape=(3,70,250), output_size=5, lr=0.001) :
        super().__init__()

        # TODO add padding 

        self.conv1 = nn.Conv2d(input_shape[0], 1, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, )

        self.conv2 = nn.Conv2d(1, 2, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(2, 2, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(2, 4, kernel_size=3, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(104, 20)
        self.linear2 = nn.Linear(20, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.epochs = 0
        self.loss = 0.0
        self.device = device
        self.lr = lr
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool4(x)
        x = x.view(x.size(0), -1)
        x = self.linear2(self.linear1(x))
        return self.softmax(x)

    def train_model(self, train_dl, val_dl, epochs=150, save_path='/model/'):
        for i in range(epochs):
            self.train()
            sum_loss = 0.0
            total = 0
            j = 0
            e_time = time.time()
            for x, y in train_dl:
                t = time.time()
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(y_pred, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
                j += 1
                print('batch %.0f, loss %.3f, time %.2f' % (j, loss.item()*y.shape[0], time.time()-t))
            self.loss = sum_loss/total
            val_loss, val_acc, val_rmse = self.validation_metrics(val_dl)
            print('-----------------------------------------------------------------------------')
            print("epoch %.0f, train loss %.3f, val loss %.3f, val accuracy %.3f, val rmse %.3f, t %.1f" % ((i+1), self.loss, val_loss, val_acc, val_rmse, time.time()-e_time))
            print('-----------------------------------------------------------------------------')
            if ((i+1) % 5) == 0:
                self.save_final(os.getcwd() + save_path + str(i+1).zfill(4) + '.pth')

    def validation_metrics(self, valid_dl):
        self.eval()
        correct = 0
        total = 0
        sum_loss = 0.0
        sum_rmse = 0.0
        for x, y in valid_dl:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self(x)
            loss = F.cross_entropy(y_pred, y)
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.cpu().unsqueeze(-1)))*y.cpu().shape[0]
        return sum_loss/total, correct/total, sum_rmse/total

    def save_checkpoint(self, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, path)

    def save_final(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        model.train()

    def load_inference(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()