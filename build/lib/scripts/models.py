
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error


class ConvNet(torch.nn.Module) :
    def __init__(self, device, input_shape=(70,250,3), output_size=2, lr=0.001) :
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(1, 2, kernel_size=3, stride=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(2, 2, kernel_size=3, stride=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(2, 4, kernel_size=3, stride=3)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(20, 5)
        self.linear2 = nn.Linear(5, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.epochs = 0
        self.loss = 0.0
        self.device = device
        self.lr = lr
        
    def forward(self, x):
        import pdb; pdb.set_trace()
        return x

    def train_model(self, train_dl, val_dl, epochs=150):
        for i in range(epochs):
            self.train()
            sum_loss = 0.0
            total = 0
            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                y_pred, y_prob = self(x)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(y_pred, y)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
            self.loss = sum_loss/total
            val_loss, val_acc, val_rmse = self.validation_metrics(val_dl)
            if i % 5 == 1:
                print("train loss %.3f, val loss %.3f, val accuracy %.3f, val rmse %.3f, lr %.5f" % (self.loss, val_loss, val_acc, val_rmse, self.lr))
                self.lr = self.lr*self.lr_decay
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def validation_metrics(self, valid_dl):
        self.eval()
        correct = 0
        total = 0
        sum_loss = 0.0
        sum_rmse = 0.0
        for x, y in valid_dl:
            x, y = x.to(self.device), y.to(self.device)
            y_hat, y_prob = self(x)
            loss = F.cross_entropy(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
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
        if self.use_svm:
            svm_filename = path.split('.')[0] + '_svm.pth'
            pickle.dump(self.svm, open(svm_filename, 'wb'))

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
        if self.use_svm:
            svm_filename = path.split('.')[0] + '_svm.pth'
            self.svm = pickle.load(open(svm_filename, 'rb'))