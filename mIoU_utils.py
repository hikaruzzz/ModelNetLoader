import numpy as np
import torch.nn as nn
import torch


# method 1 for mIoU compute(2D image)
def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


# method 2 for mIoU compute (1D array)
def calc_iou(pred,label,n_class=20):
    '''
    计算一个batch里面的iou和acc，所以return的数据为整个batch 的mean data
    若 iou=1，说明label不存在这个class,且pred也没有这个class。
    若 iou=0，说明label存在该class，但pred没有class
    :param pred: [B, S, n_class]
    :param label: [B, S]
    :return:
    '''

    # assert 20 == len(CLASS_LABELS), "wrong n_class"

    batch_size = pred.shape[0]  # shape=[batchsize,n_point]
    n_point = pred.shape[1]

    cat_table = np.zeros((2,n_class))  # cat_table.shape=[n_class,3]  cat_table[x][0-2]: acc,currency iou,miou

    pred = pred.numpy()
    pred = np.argmax(pred,axis=2)

    label = label.numpy()
    for i in range(batch_size):
        # per batch
        for cla in range(n_class):
            # per class
            acc_m1 = np.zeros(n_point)
            acc_m2 = np.zeros(n_point)-1

            I = np.sum(np.logical_and(pred[i] == cla, label[i] == cla))
            U = np.sum(np.logical_or(pred[i] == cla, label[i] == cla))
            if U != 0:
                iou = I / U
            else: # label[i] == cls全为0，simple中没有这个class
                iou = 1
            cat_table[1][cla] += iou

            acc_m1[pred[i] == cla] = 1
            acc_m2[label[i] == cla] = 1

    cat_table[1] = cat_table[1] / batch_size

    return cat_table[1]

#====================================================================================
# prepare test data
n_class = 5

w = 1
h = 10

x = (np.random.random([1,n_class,1,3])*10).astype(np.float)
y = (np.random.random([1,w,h])*10).astype(np.int)

x = torch.from_numpy(x)
y = torch.from_numpy(y).long()
print("x\n",x)

pred = nn.functional.interpolate(x, size=(w, h), mode='bilinear')
print("interpolate\n",pred)


# how to use method 1
confusion_matrix = np.zeros([n_class,n_class])
for batch in range(10):
    confusion_matrix += get_confusion_matrix(y,pred,(w,h),n_class)

pos = confusion_matrix.sum(1)
res = confusion_matrix.sum(0)
tp = np.diag(confusion_matrix)
IoU_array = (tp / np.maximum(1.0, pos + res - tp))
mean_IoU = IoU_array.mean()


# how to use method 2
pred = pred.view([1,n_class,-1]).permute([0,2,1])
y = y.view([1,-1])

print(pred.permute([0,2,1]).view([1,n_class,w,h]))
mIoU = 0
for i in range(10):
    mIoU += calc_iou(pred,y,n_class)
mIoU = np.mean(mIoU/10)


print("mIoU 1:", mean_IoU)
print("mIoU 2:",mIoU)
