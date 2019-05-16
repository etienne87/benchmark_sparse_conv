from __future__ import print_function
import numpy as np
import torch
from torch import nn
import spconv
from prophesee_utils import td_video
import spconv_ours as spconv2
import time

def timeit(x, func, iter=10):
	torch.cuda.synchronize()
	start = time.time()
	for _ in range(iter):
		y = func(x)
	torch.cuda.synchronize()
	runtime = (time.time()-start)/iter
	return runtime


def mevent_per_second(runtime, nevents):
    return nevents / runtime * 1e-6


if __name__ == '__main__':
    path = "/home/prophesee/work/tmp/davide_18-12-04_2_0_0_td.dat"
    video = td_video.ChronoVideo(path)
    height, width = video.get_size()
    duration = 100
    video.seek_time(int(3*1e6))
    xypt = video.load_delta_t(40000)

    nevents = len(xypt)
    t = xypt['ts']
    t = (t - t[0]) / 100


    #Sparse Tensor setup
    features = torch.empty((nevents, 1), dtype=torch.float)
    features[:, 0] = torch.from_numpy(2*xypt['p'].astype(np.int32) - 1)

    indices = torch.zeros((len(xypt), 4), dtype=torch.int)
    indices[:, 1] = torch.from_numpy(xypt['x'].astype(np.int32))
    indices[:, 2] = torch.from_numpy(xypt['y'].astype(np.int32))
    indices[:, 3] = torch.from_numpy(t.astype(np.int32))
    spatial_shape3d = (width, height, duration)
    spatial_shape2d = (width, height)
    batch_size = 1
    batch_shape = (batch_size, width, height, duration)

    features, indices = features.cuda(), indices.cuda()

    test = 'spconv2'
    do_half = False

    if test == 'spconv':
        net = spconv.SparseSequential(
            #spconv.SparseConv3d(1, 16, 3, 2),  # just like nn.Conv3d but don't support group and all([d > 1, s > 1])

            spconv.SparseConv3d(1, 32, 3, 2, indice_key='subm0'),
            # spconv.SparseConv3d(1, 32, 64, 2, indice_key='subm0'),
            # spconv.SparseConv3d(1, 64, 128, 2, indice_key='subm0'),

        )
        net.cuda()

        if do_half:
            net = net.half()
            features = features.half()

        x = spconv.SparseConvTensor(features, indices, spatial_shape3d, batch_size)
        y = net(x)
        ratio = float(len(y.indices)) / nevents
        print('ratio: ', ratio)

        rt2 = timeit(x, net, 400)
        print('runtime(gpu): ', rt2, ', = mevent/s: ', mevent_per_second(rt2, nevents)) # 48Mev/s
    elif test == 'dense':
        pass

    elif test == 'spconv2':
        # custom made (5, 20, 40 Mev/s using 3, 2, 1 dimensions... quite bad)
        net = spconv2.SparseConvNd(1, 32, 3, stride=1, nd=1)
        net.cuda()
        net.eval()

        if do_half:
            net = net.half()
            features = features.half()

        x = (indices[:,:1 + net.nd], features, batch_shape)
        with torch.no_grad():
            rt3 = timeit(x, net, 1000)
        print('runtime(gpu): ', rt3, ' speed: ', mevent_per_second(rt3, nevents), '(', nevents*1e-6, ' )')


