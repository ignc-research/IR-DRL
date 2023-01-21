import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
font = {
        'size'   : 16}
plt.rc('font', **font)
def smooth(data,weight=0.9):
    last = data[0]
    smoothed = []
    for i in data:
        smoothed_val = last * weight + (1 - weight) * i
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# load raw data
# i.e. 
# ur5_reward[:,0] : steps
# ur5_reward[:,1] : rewards
# ur5_reward[:,2] : steps per episode
# ur5_reward[:,3] : episodes
ur5_all = np.load('./raw/ur5_reward.npy')
kuka_all = np.load('./raw/kuka_reward.npy')

ur5_step = ur5_all[:,0]
ur5_reward = ur5_all[:,1]
ur5_steps_per_episode = ur5_all[:,2]
ur5_episode = ur5_all[:,3]

kuka_step = kuka_all[:,0]
kuka_reward = kuka_all[:,1]
kuka_steps_per_episode = kuka_all[:,2]
kuka_episode = kuka_all[:,3]

#================================================================
# plot rewards for ur5
fig = plt.figure(figsize = (7,5))
ax1 = fig.add_subplot(1, 1, 1)
# plot the raw data
plt.plot(ur5_step, ur5_reward,'g-',alpha = 0.3)
# smoothing
smoothed_reward = np.array(smooth(ur5_reward))
plt.plot(ur5_step, smoothed_reward,'g-',label=u'reward',alpha = 1)
# some descriptions
# plt.title('reward of ur5')
# plt.legend(loc='upper left')
plt.xlabel('step',fontweight='bold')
plt.ylabel('reward',fontweight='bold')
# add episode lable on a second x axis
# because the episode is manually caculated(through steps and steps per episode, i didnt find the episode recorder in tensorboard..)
ax2 = ax1.twiny()
ax2.plot(ur5_episode, ur5_reward, 'r',alpha = 0)
ax2.set_xlabel('episode',fontweight='bold')
# ax2.set_xlim(0, )
# ax2.tick_params(axis='x', which='major', length=3)


# enlarge a part of curve
enable_small_image = False
if enable_small_image:
    #Set the range of horizontal coordinate of the area you want to enlarge
    tx0 = 2e7
    tx1 = 3e7
    #Set the range of vertical coordinates of the area you want to enlarge
    ty0 = 0
    ty1 = 10

    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]

    plt.plot(sx,sy,'black',linewidth = 0.5)
    # Set the placement of the small image, there can be "lower left,lower right,upper right,upper left, upper,center,center left,right,center right,lower center,center"
    axins = inset_axes(ax1, width=1.0, height=1.0, loc='right')
    # plot in small image
    axins.plot(ur5_step, ur5_reward, color='red', ls='-', linewidth = 0.2)
    # range of small image's axises[xmin, xmax, ymin, ymax]
    axins.axis([tx0,tx1,ty0,ty1])
plt.savefig('ur5_training.png')
# plt.show()


#================================================================
# plot rewards for kuka
fig = plt.figure(figsize = (7,5))
ax1 = fig.add_subplot(1, 1, 1)

plt.plot(kuka_step, kuka_reward,'g-',label=u'reward',alpha = 1)
# plt.title('reward of kuka')
# plt.legend(loc='upper left')
plt.xlabel('step',fontweight='bold')
plt.ylabel('reward',fontweight='bold')
# add episode lable on a second x axis
# because the episode is manually caculated(through steps and steps per episode, i didnt find the episode recorder in tensorboard..)
ax2 = ax1.twiny()
ax2.plot(kuka_episode, kuka_reward, 'r',alpha = 0)
ax2.set_xlabel('episode',fontweight='bold')
ax2.set_xticks(np.arange(min(ur5_episode), max(ur5_episode), 50000))
# plt.grid()
plt.savefig('kuka_training.png')


# enlarge a part of curve
enable_small_image = False
if enable_small_image:
    #Set the range of horizontal coordinate of the area you want to enlarge
    tx0 = 3e6
    tx1 = 4e6
    #Set the range of vertical coordinates of the area you want to enlarge
    ty0 = 8
    ty1 = 11
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]

    plt.plot(sx,sy,'black',linewidth = 0.5)
    # Set the placement of the small image, there can be "lower left,lower right,upper right,upper left, upper,center,center left,right,center right,lower center,center"
    axins = inset_axes(ax1, width=1.0, height=1.0, loc='right')
    # plot in small image
    axins.plot(kuka_step, kuka_reward, color='red', ls='-', linewidth = 0.2)
    # range of small image's axises[xmin, xmax, ymin, ymax]
    axins.axis([tx0,tx1,ty0,ty1])
plt.show()



