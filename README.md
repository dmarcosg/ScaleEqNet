# ScaleEqNet, Scale equavariant CNNs with vector fields

Implementation of the method it:
Scale equivariance in CNNs with vector fields (ICML Workshops 2018),
D Marcos, B Kellenberger, S Lobry, D Tuia
https://arxiv.org/pdf/1807.11783.pdf

Implementation based on the work of Anders U. Waldeland
https://github.com/COGMAR/RotEqNet/

### Usage
To download and setup the MNIST-scale dataset, cd into the mnist folder and run:
```
python download_mnist.py
python make_mnist_scale.py
```
To run the MNIST-scale test:
```
python mnist_test.py
```

This code has been tested with Python 3.6.

The following python packages are required:

```
torch
numpy
scipy
```
