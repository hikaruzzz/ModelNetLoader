import matplotlib.pyplot as plt

import numpy as np
import warnings
import h5py
import os
from torch.utils.data import Dataset
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
class_label = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']


def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(os.path.join(dir, 'ply_data_train0.h5'))
    data_train1, label_train1,Seglabel_train1 = load_h5(os.path.join(dir, 'ply_data_train1.h5'))
    data_train2, label_train2,Seglabel_train2 = load_h5(os.path.join(dir, 'ply_data_train2.h5'))
    data_train3, label_train3,Seglabel_train3 = load_h5(os.path.join(dir, 'ply_data_train3.h5'))
    data_train4, label_train4,Seglabel_train4 = load_h5(os.path.join(dir, 'ply_data_train4.h5'))
    data_test0, label_test0,Seglabel_test0 = load_h5(os.path.join(dir, 'ply_data_test0.h5'))
    data_test1, label_test1,Seglabel_test1 = load_h5(os.path.join(dir, 'ply_data_test1.h5'))
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, rotation = None):
        self.data = data
        self.labels = labels
        self.rotation = rotation

    def __len__(self):
        return len(self.data)

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix).astype(np.float32)

        return rotated_data

    def __getitem__(self, index):
        '''

        :param index:
        :return: data [B,C,S]
                labels [B,1]
        '''
        norm_para = 1
        if self.rotation is not None:
            pointcloud = self.data[index]
            angle = np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
            pointcloud = self.rotate_point_cloud_by_angle(pointcloud, angle)

            cat_data = np.concatenate([pointcloud, pointcloud / norm_para], axis=1)

            return cat_data, self.labels[index]
        else:
            cat_data = np.concatenate([self.data[index], self.data[index] / norm_para], axis=1)

            return cat_data, self.labels[index]


def visible_modelnet(points, labels,is_show=True):
    """
    :param points: # shape = (4096,9)  data_batch[2,:,:]
    :param labels:   label_batch[2,:]
    :return:
    """

    skip = 1  # Skip every n points

    fig = plt.figure(dpi=200)

    ax = fig.add_subplot(111, projection='3d')
    point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
    ax.scatter(points[point_range, 0],  # x
               points[point_range, 1],  # y
               points[point_range, 2],  # z

               cmap='Spectral',
               alpha=1,
               s=2,
               marker=".")
    ax.axis('scaled')  # {equal, scaled}
    plt.title(class_label[labels])
    # print(labels)

    if is_show:
        plt.show()

def show_modelnet():
    root = r'C:\Users\PC\Desktop\data\modelnet\modelnet40'
    import torch
    train_data, train_label, test_data, test_label = load_data(root, classification=True)
    trainDataset = ModelNetDataLoader(train_data, train_label, rotation=None)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(trainDataLoader):
        points, label = batch

        visible_modelnet(points[0],label[0][0])



if __name__ == '__main__':
    show_modelnet()
