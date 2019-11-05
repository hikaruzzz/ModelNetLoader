import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def func(x,y):

    return np.power(x,2) + np.power(y,2)


if __name__ == '__main__':
    fig1 = plt.figure()
    
    ax = Axes3D(fig1)
    
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    
    x = np.arange(-1,1,0.1)
    y = np.arange(-1,1,0.1)

    xyz = np.zeros([1, 3, x.shape[0]*y.shape[0]])
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            xyz[0,:,i*y.shape[0] + j] = np.array([x[i],y[j],0])

    points = np.zeros([xyz.shape[0], 1, xyz.shape[2]]) + 1

    xyz = torch.from_numpy(xyz).float()
    points = torch.from_numpy(points).float()
    print(xyz.shape)
    print(points.shape)


    grouped_xyz_norm = np.zeros([])
    dist = xyz[:,0,:] **2 + xyz[:,1,:]**2 + xyz[:,2,:] **2
    radius = [1]
    sigma2 = 0.1 # sigma ** 2, apex : x = radius/2
    gauss = 1 / (np.sqrt(2 * np.pi * sigma2)) * np.exp(
        -(dist - np.sqrt(dist) * radius[0] + radius[0] ** 2 / 4) / (2 * sigma2))

    z = gauss.reshape([20,20])


    sigma2 = 0.3 # sigma ** 2, apex : x = radius/2
    gauss = 1 / (np.sqrt(2 * np.pi * sigma2)) * np.exp(
        -(dist - np.sqrt(dist) * radius[0] + radius[0] ** 2 / 4) / (2 * sigma2))

    z2 = gauss.reshape([20,20])

    x,y = np.meshgrid(x,y)

    ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=plt.cm.coolwarm)
    ax.set_xlabel('x label',color='r')
    ax.set_ylabel('y label',color='g')
    ax.set_zlabel('z label',color='b')

    ax2.plot_surface(x,y,z2,rstride=1,cstride=1,cmap=plt.cm.coolwarm)
    ax2.set_xlabel('x label',color='r')
    ax2.set_ylabel('y label',color='g')
    ax2.set_zlabel('z label',color='b')

    plt.show()