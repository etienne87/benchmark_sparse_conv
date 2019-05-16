import torch
import torch.nn as nn
import torch_scatter


class SparseConvNd(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=2, nd=2, depthwise=False):
        super(SparseConvNd, self).__init__()
        self.cin = cin
        self.cout = cout
        self.depthwise = depthwise
        self.kernel_size = kernel_size
        self.nd = nd
        self.stride = stride
        self.rf = kernel_size**nd

        if self.depthwise:
            assert self.cin == self.cout
            self.weights = torch.nn.Parameter(torch.randn(cin, self.rf))
        else:
            self.weights = torch.nn.Parameter(torch.randn(cin, cout * self.rf))

        #Offsets
        lin = torch.arange(0, kernel_size) #padding is kernel_size//2
        lins = [lin for i in range(nd)]
        off = torch.meshgrid(lins)
        xyv = torch.stack(off).view(nd, -1).t()
        n = torch.zeros((self.rf, 1), dtype=torch.long)
        nxyv = torch.cat([n, xyv], dim=1)

        self.register_buffer('offsets', nxyv)

        print('weights: ', self.weights.shape)
        self.bias = torch.nn.Parameter(torch.randn(cout))

    def forward(self, x):
        pts, vals, size = x
        pts, y = sparse_convnd(pts.long(), vals, self.weights, size, self.offsets, self.rf, self.stride, self.depthwise)
        y += self.bias
        return (pts, y, size)


def linear_to_index(linear_index, size):
    """
    linear index to actual index
    """
    index = []
    new_linear = linear_index
    for i in range(len(size) - 1, -1, -1):
        index.append(torch.fmod(new_linear, size[i]).unsqueeze(1))
        new_linear = new_linear.div(size[i])
    index.reverse()
    return torch.cat(index, 1)


def coalesce(idx, value, size, method=0, operation='add'):
    """
    coalesce function inspired by https://github.com/rusty1s/pytorch_sparse/blob/master/torch_sparse/coalesce.py
    """
    assert idx.min() >= 0
    index = idx.t()
    linear_index = 0
    for i in range(len(index)):
        linear_index *= size[i]
        linear_index += index[i]

    op = getattr(torch_scatter, 'scatter_{}'.format(operation))

    if method == 0:
        unique, inv = torch.unique(linear_index, sorted=True, return_inverse=True)
        index = linear_to_index(unique, size)
        value = op(value, inv, dim=0, out=None, dim_size=unique.size(0), fill_value=0)
        if isinstance(value, tuple):
            value = value[0]

    if method == 1:
        unique = torch.unique(linear_index, sorted=True, return_inverse=False)
        index = linear_to_index(unique, size)
        value = op(value, linear_index, dim=0, out=None, dim_size=unique[-1].item() + 1, fill_value=0)
        if isinstance(value, tuple):
            value = value[0]
        value = value[unique]

    return index, value


def sparse_convnd(points, values, weights, size, offsets, rf, stride=2, depthwise=False):
    """"

    :param points: nc format NxC_dim (last 2: height, width)
    :param values: nd format NxD
    :param weights: D, k, k (depthwise weights, 1 output/ channel)
    :param size: output size
    :param offsets,
    :param stride: average pooling during coalescing by shifting indices
    :return:
    """
    nevents, ncols = points.shape
    nevents = points.size(0)
    if depthwise:
        new_values = values * weights
    else:
        new_values = torch.mm(values, weights)

    new_values = new_values.view(nevents * rf, -1)

    with torch.no_grad():
        new_points = points.view(nevents, 1, -1) + offsets.view(1, -1, ncols)
        new_points = new_points.view(-1 , ncols)
        new_points[:, 1:] = new_points[:, 1:] >> (stride-1)

    new_points, new_values = coalesce(new_points, new_values, size)

    return new_points, new_values
