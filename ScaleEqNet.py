import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils import *


class ScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, n_scales_small=5,n_scales_big=3, mode=1, angle_range = 120, output_mode = 2):
        super(ScaleConv, self).__init__()

        kernel_size = ntuple(2)(kernel_size)
        stride = ntuple(2)(stride)
        padding = ntuple(2)(padding)
        dilation = ntuple(2)(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.n_scales_small = n_scales_small
        self.n_scales_big = n_scales_big
        self.n_scales = n_scales_small + n_scales_big
        self.angle_range = angle_range
        self.mode = mode

        # Angles
        self.angles = np.linspace(-angle_range*self.n_scales_small/self.n_scales,
                                  angle_range*self.n_scales_big/self.n_scales, self.n_scales, endpoint=True)


        self.weight1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        # If input is vector field, we have two filters (one for each component)
        if self.mode == 2:
            self.weight2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.mode == 2:
            self.weight2.data.uniform_(-stdv, stdv)

    def _apply(self, func):
        # This is called whenever user calls model.cuda()
        # We intersect to replace tensors and variables with cuda-versions
        super(ScaleConv, self)._apply(func)

    def forward(self, input):


        if self.mode == 1:
            outputs = []
            orig_size = list(input.data.shape[2:4])
            # Input upsampling scales (smaller filter scales)
            input_s = input.clone()
            for n in range(1, self.n_scales_big+1):
                size = [0,0]
                size[0] = int(round(1.26 ** n * orig_size[0]))
                size[1] = int(round(1.26 ** n * orig_size[1]))
                input_s = F.upsample(input_s, size=size,mode='bilinear')
                out = F.conv2d(input_s, self.weight1, None, self.stride, self.padding, self.dilation)
                out = F.upsample(out, size=orig_size,mode='bilinear')
                outputs.append(out.unsqueeze(-1))


            # Input downsampling scales (larger filter scales)
            input_s = input.clone()
            for n in range(0, self.n_scales_small):
                size = [0, 0]
                size[0] = int(round(1.26 ** -n * orig_size[0]))
                size[1] = int(round(1.26 ** -n * orig_size[1]))
                input_s = F.upsample(input_s, size=size,mode='bilinear')
                out = F.conv2d(input_s, self.weight1, None, self.stride, self.padding, self.dilation)
                out = F.upsample(out, size=orig_size,mode='bilinear')
                outputs = [out.unsqueeze(-1)] + outputs



        if self.mode == 2:
            u = input[0]
            v = input[1]
            orig_size = list(u.data.shape[2:4])
            outputs = []

            # Input upsampling scales (smaller filter scales)
            u_s = u.clone()
            v_s = v.clone()
            for n in range(1, self.n_scales_big+1):
                wu = self.weight1
                wv = self.weight2
                n_scale = self.n_scales_small + n - 1
                angle = -self.angles[n_scale] * np.pi / 180
                wru = np.cos(angle).__float__() * wu - np.sin(angle).__float__() * wv
                wrv = np.sin(angle).__float__() * wu + np.cos(angle).__float__() * wv

                size = [0, 0]
                size[0] = int(round(1.26 ** n * orig_size[0]))
                size[1] = int(round(1.26 ** n * orig_size[1]))
                u_s = F.upsample(u_s, size=size,mode='bilinear')
                u_out = F.conv2d(u_s, wru, None, self.stride, self.padding, self.dilation)
                u_out = F.upsample(u_out, size=orig_size,mode='bilinear')
                v_s = F.upsample(v_s, size=size,mode='bilinear')
                v_out = F.conv2d(v_s, wrv, None, self.stride, self.padding, self.dilation)
                v_out = F.upsample(v_out, size=orig_size,mode='bilinear')
                outputs.append((u_out + v_out).unsqueeze(-1))

            # Input downsampling scales (smaller filter scales)
            u_s = u.clone()
            v_s = v.clone()
            for n in range(0, self.n_scales_small):
                wu = self.weight1
                wv = self.weight2
                n_scale = self.n_scales_small - n - 1
                angle = -self.angles[n_scale] * np.pi / 180
                wru = np.cos(angle).__float__() * wu - np.sin(angle).__float__() * wv
                wrv = np.sin(angle).__float__() * wu + np.cos(angle).__float__() * wv

                size = [0, 0]
                size[0] = int(round(1.26 ** -n * orig_size[0]))
                size[1] = int(round(1.26 ** -n * orig_size[1]))
                u_s = F.upsample(u_s, size=size,mode='bilinear')
                u_out = F.conv2d(u_s, wru, None, self.stride, self.padding, self.dilation)
                u_out = F.upsample(u_out, size=orig_size,mode='bilinear')
                v_s = F.upsample(v_s, size=size,mode='bilinear')
                v_out = F.conv2d(v_s, wrv, None, self.stride, self.padding, self.dilation)
                v_out = F.upsample(v_out, size=orig_size,mode='bilinear')
                outputs = [(u_out + v_out).unsqueeze(-1)] + outputs


        # Get the maximum direction (Orientation Pooling)
        strength, max_ind = torch.max(torch.cat(outputs, -1), -1)

        # Convert from polar representation
        angle_map = (max_ind.float() - self.n_scales_small) * np.pi/180. * self.angle_range / len(self.angles)
        u = F.relu(strength) * torch.cos(angle_map)
        v = F.relu(strength) * torch.sin(angle_map)

        return u, v


class VectorMaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False):
        super(VectorMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        # Magnitude
        p = torch.sqrt(v ** 2 + u ** 2)
        # Max pool
        _, max_inds = F.max_pool2d(p, self.kernel_size, self.stride,
                                   self.padding, self.dilation, self.ceil_mode,
                                   return_indices=True)
        # Reshape to please pytorch
        s1 = u.size()
        s2 = max_inds.size()

        max_inds = max_inds.view(s1[0], s1[1], s2[2] * s2[3])

        u = u.view(s1[0], s1[1], s1[2] * s1[3])
        v = v.view(s1[0], s1[1], s1[2] * s1[3])

        # Select u/v components according to max pool on magnitude
        u = torch.gather(u, 2, max_inds)
        v = torch.gather(v, 2, max_inds)

        # Reshape back
        u = u.view(s1[0], s1[1], s2[2], s2[3])
        v = v.view(s1[0], s1[1], s2[2], s2[3])

        return u, v


class Vector2Magnitude(nn.Module):
    def __init__(self):
        super(Vector2Magnitude, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        p = torch.sqrt(v ** 2 + u ** 2)
        return p

class Vector2Angle(nn.Module):
    def __init__(self):
        super(Vector2Angle, self).__init__()

    def forward(self, input):
        u = input[0]
        v = input[1]

        p = torch.atan2(u , v )
        return p


class VectorBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.5, affine=True):

        super(VectorBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()

    def forward(self, input):
        """
        Based on https://github.com/lberrada/bn.pytorch
        """
        if self.training:
            # Compute std
            std = self.std(input)

            alpha = self.weight / (std + self.eps)

            # update running variance
            self.running_var *= (1. - self.momentum)
            self.running_var += self.momentum * std.data ** 2
            # compute output
            u = input[0] * alpha
            v = input[1] * alpha

        else:
            alpha = self.weight.data / torch.sqrt(self.running_var + self.eps)

            # compute output
            u = input[0] * Variable(alpha)
            v = input[1] * Variable(alpha)
        return u, v

    def std(self, input):
        u = input[0]
        v = input[1]

        # Vector to magnitude
        p = torch.sqrt(u ** 2 + v ** 2)

        # Mean
        mu = torch.mean(p, 0, keepdim=True)
        mu = torch.mean(mu, 2, keepdim=True)
        mu = torch.mean(mu, 3, keepdim=True)

        # Variance
        var = (p) ** 2
        # This line should perharps read:
        # var = (p-mu)**2 #?

        var = torch.sum(var, 0, keepdim=True)
        var = torch.sum(var, 2, keepdim=True)
        var = torch.sum(var, 3, keepdim=True)
        std = torch.sqrt(var)

        return std


class VectorUpsampling(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(VectorUpsampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        # Assuming input is vector field
        u = input[0]
        v = input[1]

        u = F.upsample(u, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        v = F.upsample(v, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


        return u, v