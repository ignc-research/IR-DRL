import numpy as np
class RaysCauculator():
    def __init__(self, obs_rays):
        self.obs_rays = obs_rays
        self.indicator = np.zeros((10,), dtype=np.int8)

    def get_indicator(self):
        rays_sum = []
        self.obs_tip = self.obs_rays[0:25]
        self.side_1 = self.obs_rays[25:35]
        self.side_2 = self.obs_rays[35:45]
        self.side_3 = self.obs_rays[45:55]
        self.side_4 = self.obs_rays[55:65]
        self.side_5 = self.obs_rays[65:75]
        self.side_6 = self.obs_rays[75:85]
        self.side_7 = self.obs_rays[85:95]
        self.side_8 = self.obs_rays[95:105]
        self.obs_top = self.obs_rays[105:]
        rays_sum.append(self.obs_tip)
        rays_sum.append(self.side_1)
        rays_sum.append(self.side_2)
        rays_sum.append(self.side_3)
        rays_sum.append(self.side_4)
        rays_sum.append(self.side_5)
        rays_sum.append(self.side_6)
        rays_sum.append(self.side_7)
        rays_sum.append(self.side_8)
        rays_sum.append(self.obs_top)
        for i in range(10):
            if rays_sum[i].min()>=0.99:
                self.indicator[i] = 0
            if 0.5<rays_sum[i].min()<0.99:
                self.indicator[i] = 1
            if rays_sum[i].min()<=0.5:
                self.indicator[i] = 2
        return self.indicator