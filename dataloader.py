import os
import numpy as np
import torch

from torch.utils.data import Dataset

class Cityscapes_Loader(Dataset):
    '''
    重构了Dataset里面的方法，其中__getitem__()是返回单个图像的，交由DataLoader进行batch size打包
    '''

    def __init__(self,path_root,split,n_classes):
        '''
        只需要传入一个路径，通过__getitem__递归读取每张图
        :param path_root: x/CityScapes/
        :param split:   "test" or "train" or "val" in the dataset
        :param n_classes:  需要缩减至的classes数量
        :param is_transform:
        :param label_type: 使用labelids或instancelds.png
        '''
        self.path_root = path_root
        self.n_classes = n_classes
        self.split = split

        # ./data/
        files_path = os.path.join(path_root, split)
        self.imgs_path_list = self.getFilesList(files_path)

        self.class_map = ['wong', 'cai', 'li']

        file_names = [p.split(os.sep)[-1][:-4] for p in self.imgs_path_list]
        self.saveLabelFile(file_names, self.class_map)
        label_list_pth = './label_list.txt'

        # 读取label txt
        self.label_list = []
        with open(label_list_pth, 'r', encoding='utf-8') as f:
            self.label_list = f.readlines()
            for i in range(len(self.label_list)):
                self.label_list[i] = self.label_list[i].strip('\n')


        assert self.__len__() > 0, "Error: not found image in path = {}".format(self.path_root)

    def __len__(self):
        '''
        :return: 整个split中的所有图片len（包含各city）
        '''
        return len(self.imgs_path_list)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        :return img  img.shape = [3, H, W]
        :return  label.shape=(H,W) , target.shape=(n_classes, H, W)
        """
        img_path = self.imgs_path_list[index]
        # 根据文件列表，从中找到对应名称的label image，

        # 读取 单个img和对应的 label 图
        img = 0
        label = self.label_list[index]

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()

        return img, label # label.shape=(H,W) , target.shape=(n_classes, H, W)

    def getFilesList(self,path_root):
        '''
        获取data/'split'/下面所有图片（把各个city的都结合到一起）
        :param path_root: to data dir
        :return:
        '''
        files_path = os.path.join(path_root,"data",self.split)
        imgs_path_list = []
        for looproot, _, filesnames in os.walk(files_path):
            for filesname in filesnames:
                imgs_path_list.append(os.path.join(looproot, filesname.strip()))

        return imgs_path_list

    def saveLabelFile(self, file_names, class_map):
        '''
        根据固定class表，转换成数字label，保存label.txt,
        :param imgs_path_list: raw images path, like: ['cai_001', 'li_001']
        :param class_map: 固定的classes 表, like: [wong, cai, ...]
        txt如下
            cai_001_1
            li_001_2
            wong_001_0
            wong_002_0
        '''

        names_list = np.array([n.split('_')[0] for n in file_names])
        label_list = np.array([-1 for i in range(len(names_list))])
        # 把names按照class map转成 数字label ['a','c','b'] -> [0,2,1]
        for i, cl in zip(range(len(class_map) + 1), class_map):
            label_list[cl == names_list] = i

        # 拼接file name，生成label txt [wong_001 -> wong_001_1] in each line

        with open('label_list.txt', 'w', encoding='utf-8') as f:
            for i in range(len(file_names)):
                data = file_names[i] + '_' + str(label_list[i]) + '\n'
                f.write(data)


if __name__ == "__main__":

    path_root = '.'

    files_path = os.path.join(path_root, "data", 'train')
    imgs_path_list = []
    for looproot, _, filesnames in os.walk(files_path):
        for filesname in filesnames:
            imgs_path_list.append(os.path.join(looproot, filesname.strip()))

    print(imgs_path_list)
