import numpy as np
import scipy.misc
import sys
sys.path.append('../')
from utils import getGrid, rotate_grid_2D

def loadMnist(mode):
    print(['Loading MNIST', mode, 'images'])
    # Mode = 'train'/'test
    mnist_folder = '/home/diego/PycharmProjects/RotEqNet/mnist/mnist/'

    with open(mnist_folder + mode + '-labels.csv') as f:
        path_and_labels = f.readlines()

    samples = []
    for entry in path_and_labels:
        path = entry.split(',')[0]
        label = int(entry.split(',')[1])
        img = scipy.misc.imread(mnist_folder + path)

        samples.append([img, label])
    return samples




def loadMnistScale(folder = 'mnist_scale0/'):
    def load_and_make_list(mode):
        data = np.load(folder + mode + '_data.npy')
        lbls = np.load(folder + mode + '_label.npy')
        scales = np.load(folder + mode + '_scale.npy')
        data = np.split(data, data.shape[2],2)
        lbls = np.split(lbls, lbls.shape[0],0)
        scales = np.split(scales, scales.shape[0], 0)

        return list(zip(data,lbls,scales))

    train = load_and_make_list('train')
    val = load_and_make_list('val')
    test = load_and_make_list('test')
    return train, val, test

def random_scaling(data):
    scale = np.random.rand() * 0.7 + 0.3  # Random rotation
    data = scipy.misc.imresize(data,scale)
    if len(data) < 28:
        data = np.pad(data, int(round((28-len(data))/2)), mode='constant')

    s = len(data)
    if s > 28:
        data = data[int(round((s-28)/2)):int(round((s-28)/2)+28),int(round((s-28)/2)):int(round((s-28)/2)+28)]
    data = scipy.misc.imresize(data, [28,28])
    return data.astype('float32') / 255. , scale
