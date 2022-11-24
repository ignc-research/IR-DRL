import numpy as np
class RaysCauculator():
    def __init__(self, obs_rays):
        self.obs_rays = obs_rays
        self.indicator = np.zeros((24,), dtype=np.int8)

    def get_indicator(self):
        rays_sum = []
        rays_sum.append(self.obs_rays[0:1])
        for i in range(0,50,10):
            rays_sum.append(self.obs_rays[1+i:1+i+10])
        for i in range(0,30,6):
            rays_sum.append(self.obs_rays[51+i:51+i+6])
        for i in range(0,30,6):
            rays_sum.append(self.obs_rays[81+i:81+i+6])
        for i in range(0,80,10):
            rays_sum.append(self.obs_rays[111+i:111+i+10])

        for i in range(len(rays_sum)):
            if rays_sum[i].min()>0.99:
                self.indicator[i] = 0
            elif 0.7<rays_sum[i].min()<=0.99:
                self.indicator[i] = 1
            elif 0.4<rays_sum[i].min()<=0.7:
                self.indicator[i] = 2
            else:
                self.indicator[i] = 3    
        return self.indicator