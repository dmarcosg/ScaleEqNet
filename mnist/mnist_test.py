import torch
import  torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import random
from mnist import loadMnistScale, random_scaling

import sys
sys.path.append('../') #Import
from ScaleEqNet import *


#!/usr/bin/env python
__author__ = "Anders U. Waldeland"
__email__ = "anders@nr.no"

"""
A reproduction of the MNIST-classification network described in:
Rotation equivariant vector field networks (ICCV 2017)
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346
https://github.com/dmarcosg/RotEqNet
"""


if __name__ == '__main__':

    # Define network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = ScaleConv(1, 12, [7, 7], 1, padding=3, mode=1)
            self.pool1 = VectorMaxPool(2)
            self.bn1 = VectorBatchNorm(12)

            self.conv2 = ScaleConv(12, 32, [7, 7], 1, padding=3, mode=2)
            self.pool2 = VectorMaxPool(2)
            self.bn2 = VectorBatchNorm(32)

            self.conv3 = ScaleConv(32, 48, [7, 7], 1, padding=3, mode=2)
            self.pool3 = VectorMaxPool(4)
            self.v2m = Vector2Magnitude()
            self.v2a = Vector2Angle()


            self.fc1 = nn.Conv2d(48, 256, 1)  # FC1
            self.fc1bn = nn.BatchNorm2d(256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout2d(0.7)
            self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

            self.afc1 = nn.Conv2d(48, 48, 1)  # FC1
            self.afc1bn = nn.BatchNorm2d(48)
            self.afc1relu = nn.ReLU()
            self.adropout = nn.Dropout2d(0.7)
            self.afc2 = nn.Conv2d(48, 1, 1)  # FC2


        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.conv3(x)
            x = self.pool3(x)

            xm = self.v2m(x)
            xm = self.fc1(xm)
            xm = self.relu(self.fc1bn(xm))
            xm = self.dropout(xm)
            xm = self.fc2(xm)

            #xa = F.torch.cat(x,dim=1)
            xa = self.v2a(x)
            #xa = x[0]
            #xa = self.afc1(xa)
            #xa = self.relu(self.afc1bn(xa))
            xa = self.adropout(xa)
            xa = self.afc2(xa)
            xm = xm.view(xm.size()[0], xm.size()[1])
            xa = xa.view(xa.size()[0], xa.size()[1])

            return xm,xa

    class Net_scalar(nn.Module):
        def __init__(self):
            super(Net_scalar, self).__init__()

            self.conv1 = ScaleConv(1, 12, [7, 7], 1, padding=3, mode=1)
            self.pool1 = nn.MaxPool2d(2)
            self.bn1 = nn.BatchNorm2d(12)

            self.conv2 = ScaleConv(12, 32, [7, 7], 1, padding=3, mode=1)
            self.pool2 = nn.MaxPool2d(2)
            self.bn2 = nn.BatchNorm2d(32)

            self.conv3 = ScaleConv(32, 48, [7, 7], 1, padding=3, mode=1)
            self.pool3 = nn.MaxPool2d(4)

            self.v2m = Vector2Magnitude()
            self.v2a = Vector2Angle()


            self.fc1 = nn.Conv2d(48, 256, 1)  # FC1
            self.fc1bn = nn.BatchNorm2d(256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout2d(0.7)
            self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

            self.afc1 = nn.Conv2d(48, 256, 1)  # FC1
            self.afc1bn = nn.BatchNorm2d(256)
            self.afc1relu = nn.ReLU()
            self.adropout = nn.Dropout2d(0.7)
            self.afc2 = nn.Conv2d(48, 1, 1)  # FC2


        def forward(self, x):
            x = self.conv1(x)
            x = self.v2m(x)
            x = self.pool1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.v2m(x)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.conv3(x)
            x = self.v2m(x)
            x = self.pool3(x)

            xm = self.fc1(x)
            xm = self.relu(self.fc1bn(xm))
            xm = self.dropout(xm)
            xm = self.fc2(xm)

            #xa = self.afc1(x)
            #xa = self.relu(self.afc1bn(xa))
            xa = self.adropout(x)
            xa = self.afc2(xa)
            xm = xm.view(xm.size()[0], xm.size()[1])
            xa = xa.view(xa.size()[0], xa.size()[1])

            return xm,xa

    class Net_std(nn.Module):
        def __init__(self, filter_mult = 3):
            super(Net_std, self).__init__()

            self.conv1 = nn.Conv2d(1, 12*filter_mult, [7, 7], 1, padding=3)
            self.pool1 = nn.MaxPool2d(2)
            self.bn1 = nn.BatchNorm2d(12*filter_mult)

            self.conv2 = nn.Conv2d(12*filter_mult, 32*filter_mult, [7, 7], 1, padding=3)
            self.pool2 = nn.MaxPool2d(2)
            self.bn2 = nn.BatchNorm2d(32*filter_mult)

            self.conv3 = nn.Conv2d(32*filter_mult, 48, [7, 7], 1, padding=3)
            self.pool3 = nn.MaxPool2d(4)


            self.fc1 = nn.Conv2d(48, 256, 1)  # FC1
            self.fc1bn = nn.BatchNorm2d(256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout2d(0.7)
            self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

            #self.afc1 = nn.Conv2d(48*filter_mult, 256*filter_mult, 1)  # FC1
            #self.afc1bn = nn.BatchNorm2d(256*filter_mult)
            #self.afc1relu = nn.ReLU()
            self.adropout = nn.Dropout2d(0.7)
            self.afc2 = nn.Conv2d(48, 1, 1)  # FC2

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.conv3(x)
            x = self.pool3(x)

            xm = self.fc1(x)
            xm = self.relu(self.fc1bn(xm))
            xm = self.dropout(xm)
            xm = self.fc2(xm)

            # xa = self.afc1(x)
            # xa = self.relu(self.afc1bn(xa))
            xa = self.adropout(x)
            xa = self.afc2(xa)
            xm = xm.view(xm.size()[0], xm.size()[1])
            xa = xa.view(xa.size()[0], xa.size()[1])

            return xm, xa








    def test(model, dataset):
        """ Return test-acuracy for a dataset"""
        model.eval()

        true = []
        pred = []
        true_scale = []
        pred_scale = []
        for batch_no in range(len(dataset) / batch_size):
            data, labels, scales = getBatch(dataset)
            outm, outa = model(data)
            out = F.softmax(outm)

            loss = criterion_class(out, labels)
            _, c = torch.max(out, 1)
            true.append(labels.data.cpu().numpy())
            pred.append(c.data.cpu().numpy())
            true_scale.append(scales.data.cpu().numpy())
            pred_scale.append(outa.data.cpu().numpy())
        true = np.concatenate(true, 0)
        pred = np.concatenate(pred, 0)
        pred_scale = np.concatenate(pred_scale, 0)
        true_scale = np.concatenate(true_scale, 0)
        rmse = np.sqrt(np.average((pred_scale-true_scale)**2))
        acc = np.average(pred == true)
        return acc,rmse

    def getBatch(dataset):
        """ Collect a batch of samples from list """

        # Make batch
        data = []
        labels = []
        scales = []
        for sample_no in range(batch_size):
            tmp = dataset.pop()  # Pick top element
            img = tmp[0].astype('float32').squeeze()

            # Train-time random rotation
            #img = random_scaling(img)

            data.append(np.expand_dims(np.expand_dims(img, 0), 0))
            labels.append(tmp[1].squeeze())
            scales.append(tmp[2].squeeze())
        data = np.concatenate(data, 0)
        labels = np.array(labels, 'int32')
        scales = np.array(scales, 'float32')

        data = Variable(torch.from_numpy(data))
        labels = Variable(torch.from_numpy(labels).long())
        scales = Variable(torch.from_numpy(scales))

        if type(gpu_no) == int:
            data = data.cuda(gpu_no)
            labels = labels.cuda(gpu_no)
            scales = scales.cuda(gpu_no)

        return data, labels, scales

    def adjust_learning_rate(optimizer, epoch):
        """Gradually decay learning rate"""
        if epoch == 20:
            lr = start_lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 40:
            lr = start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 60:
            lr = start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr



    def train():
        best_acc = 0
        for epoch_no in range(60):

            #Random order for each epoch
            train_set_for_epoch = train_set[:]
            random.shuffle(train_set_for_epoch)

            #Training
            net.train()
            for batch_no in range(int(len(train_set)/batch_size)):
                # Train
                optimizer.zero_grad()

                data, labels, scales = getBatch(train_set_for_epoch)
                outm, outa = net( data )
                loss_class = criterion_class( outm,labels )
                loss_scale = criterion_scale( outa, scales)
                loss = loss_class + loss_scale
                _, c = torch.max(outm, 1)
                loss.backward()
                optimizer.step()

                #Print training-acc
                if batch_no%10 == -1:
                    print(['Train', 'epoch:', epoch_no, ' batch:', batch_no, ' loss class:', loss_class.data.cpu().numpy()[0], ' loss scale:', loss_scale.data.cpu().numpy()[0],
                           ' acc:', np.average((c == labels).data.cpu().numpy())])


            #Validation
            acc,rmse = test(net, val_set[:])
            print(['Val',  'epoch:', epoch_no,  ' acc:', acc ,' rmse:', rmse])

            #Save model if better than previous
            if acc > best_acc:
                torch.save(net.state_dict(), 'best_model.pt')
                best_acc = acc
                #print('Model saved')

            adjust_learning_rate(optimizer, epoch_no)

    gpu_no = 0  # Set to False for cpu-version
    folders = ['mnist_scale0/','mnist_scale1/','mnist_scale2/','mnist_scale3/','mnist_scale4/','mnist_scale5/']
    accs = []
    rmses = []
    for crossval in range(6):
        # Setup net, loss function, optimizer and hyper parameters
        net = Net()
        criterion_class = nn.CrossEntropyLoss()
        criterion_scale = nn.MSELoss()
        if type(gpu_no) == int:
            net.cuda(gpu_no)

        start_lr = 0.01
        batch_size = 128
        optimizer = optim.Adam(net.parameters(), lr=start_lr)  # , weight_decay=0.01)

        # Load datasets
        train_set, val_set, test_set = loadMnistScale(folder = folders[crossval])
        train()
        # Finally test on test-set with the best model
        net.load_state_dict(torch.load('best_model.pt'))
        acc, rmse = test(net, test_set[:])
        accs.append(acc)
        rmses.append(rmse)
        print(['Test', 'acc:', acc,' rmse:', rmse])
    print(['All test', 'accs:', accs, ' rmses:', rmses])
    print(['Avg acc:', np.average(accs), ' Std acc:', np.std(accs)])
    print(['Avg rms:', np.average(rmses), ' Std acc:', np.std(rmses)])


    
