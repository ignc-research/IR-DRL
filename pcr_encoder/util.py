import os
import pickle

import ghalton
import numpy as np
import torch

sampling_patterns = None


def densSample(d, n):
    b, _, g, _, _ = d.shape
    out = []
    for i in range(b):
        N = torch.zeros(g, g, g).to(d.device)
        add = torch.ones([n]).to(d.device)
        d_ = d[i, 0, :, :, :].view(-1)
        d_sum = d_.sum().item()
        assert (np.isfinite(d_sum))
        if d_sum < 1e-12:
            d_ = torch.ones_like(d_)
        ind = torch.multinomial(d_, n, replacement=True)
        N.put_(ind, add, accumulate=True)
        out.append(N.int())
    out = torch.stack(out, dim=0)
    return out


# Calculates a density for the given pointcloud
def densCalc(x, grid_size):
    def ravel(coords, dims):
        return coords[2, :] + (dims[1] * coords[1, :]) + (dims[1] * dims[2] * coords[0, :])

    res = []
    for i in range(x.size(0)):
        inp = x[i, :, :]
        d = grid_size
        n = inp.size(1)

        ind = ((inp + 0.5) * d - 0.5).round().clamp(0, d - 1).long()
        resf = torch.zeros(d ** 3).to(inp)
        ind = ravel(ind, (d, d, d))
        resf.index_add_(0, ind, torch.ones(ind.size(0)).to(inp))
        res.append(resf.reshape((1, 1, d, d, d)) / n)
    return torch.cat(res, dim=0)


# creates a grid tensor [3 x s x s x s] containing the 3d coordinate of the elements
def meshgrid(s):
    r = torch.arange(s).float()
    x = r[:, None, None].expand(s, s, s)
    y = r[None, :, None].expand(s, s, s)
    z = r[None, None, :].expand(s, s, s)
    return torch.stack([x, y, z], 0)


def fixed_sample(n, count):
    global sampling_patterns
    if sampling_patterns is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sampling_patterns = pickle.load(open(dir_path + "/samplings.pkl", "rb"))

    if n < 100:
        samples = torch.from_numpy(sampling_patterns[n - 1][np.random.randint(10, size=count)])
        samples = samples.permute(1, 2, 0).contiguous().view(2, -1)
    else:
        samples = torch.tensor(ghalton.GeneralizedHalton(2).get(n * count), dtype=torch.float32).t()

    return samples


def count_params(*nets):
    n = 0
    for net in nets:
        for p in net.parameters():
            n += np.prod(p.shape)

    return n
