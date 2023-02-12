import torch
import torch.nn as nn
import torch.nn.functional as fn

from . import layer

class GridAutoEncoderAdaIN(nn.Module):
    def __init__(self, rnd_dim=2, h_dim=62, enc_p=0, dec_p=0, adain_layer=None, filled_cls=True):
        super().__init__()

        self.grid_size = 32
        self.filled_cls = filled_cls

        self.grid_encoder = layer.GridEncoder(
            nn.Sequential(
                nn.Conv2d(3, 8, 1, bias=False),
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.Conv2d(8, 16, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, 1, bias=False),
                nn.BatchNorm2d(32)),
            self.grid_size)

        self.encoder = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.Dropout3d(enc_p),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.MaxPool3d(2),  # 16
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.MaxPool3d(2),  # 8
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ELU(),
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ELU(),
            nn.MaxPool3d(2),  # 4
            nn.Conv3d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.MaxPool3d(2),  # 2
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
            nn.Conv3d(512, 1024, 2, padding=0, bias=False),
            nn.BatchNorm3d(1024),
            nn.ELU(),
        )

        self.decoder = layer.AdaptiveDecoder(nn.ModuleList([
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(512),
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 4
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            nn.Conv3d(512, 256, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(256),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 8
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(256),
            nn.ELU(),
            nn.Conv3d(256, 128, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(128),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 16
            nn.Conv3d(128, 128, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(64),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 32
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, h_dim, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(h_dim)
        ]), max_layer=adain_layer)

        self.generator = layer.PointCloudGenerator(
            nn.Sequential(nn.Conv1d(h_dim + rnd_dim, 64, 1),
                          nn.ELU(),
                          nn.Conv1d(64, 64, 1),
                          nn.ELU(),
                          nn.Conv1d(64, 32, 1),
                          nn.ELU(),
                          nn.Conv1d(32, 32, 1),
                          nn.ELU(),
                          nn.Conv1d(32, 16, 1),
                          nn.ELU(),
                          nn.Conv1d(16, 16, 1),
                          nn.ELU(),
                          nn.Conv1d(16, 8, 1),
                          nn.ELU(),
                          nn.Conv1d(8, 3, 1)),
            rnd_dim=rnd_dim, res=self.grid_size)

        self.density_estimator = nn.Sequential(
            nn.Conv3d(h_dim, 16, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 8, 1, bias=False),
            nn.BatchNorm3d(8),
            nn.ELU(),
            nn.Conv3d(8, 4, 1, bias=False),
            nn.BatchNorm3d(4),
            nn.ELU(),
            nn.Conv3d(4, 2, 1),
        )

        self.adaptive = nn.Sequential(
            nn.Linear(1024, sum(self.decoder.slices))
        )

    def encode(self, x):
        b = x.shape[0]
        x = self.grid_encoder(x)
        z = self.encoder(x).view(b, -1)

        return z

    def generate_points(self, w, n_points=5000, regular_sampling=True):
        b = w.shape[0]
        x_rec = self.decoder(w)

        est = self.density_estimator(x_rec)
        dens = fn.relu(est[:, 0])
        dens_cls = est[:, 1].unsqueeze(1)
        dens = dens.view(b, -1)

        dens_s = dens.sum(-1).unsqueeze(1)
        mask = dens_s < 1e-12
        ones = torch.ones_like(dens_s)
        dens_s[mask] = ones[mask]
        dens = dens / dens_s
        dens = dens.view(b, 1, self.grid_size, self.grid_size, self.grid_size)

        if self.filled_cls:
            filled = torch.sigmoid(dens_cls).round()
            dens_ = filled * dens
            for i in range(b):
                if dens_[i].sum().item() < 1e-12:
                    dens_[i] = dens[i]
        else:
            dens_ = dens

        if regular_sampling:
            cloud, reg = self.generator.forward_fixed_pattern(x_rec, dens_, n_points)
        else:
            cloud, reg = self.generator(x_rec, dens_, n_points)

        return cloud, dens, dens_cls.squeeze(), reg

    def decode(self, z, n_points=5000, regular_sampling=True):
        b = z.shape[0]
        w = self.adaptive(z.view(b, -1))
        return self.generate_points(w, n_points, regular_sampling)

    def forward(self, x, n_points=5000, regular_sampling=True):
        z = self.encode(x)
        return self.decode(z, n_points, regular_sampling)
