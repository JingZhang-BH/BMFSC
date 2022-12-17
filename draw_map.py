import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.io

size = {'IP':[145,145], 'SV':[512,217], 'PU':[610,340], 'PC':[1096,715]}

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

def draw_map(name, method, pre, num_labeled):
    path_gt = './data/gt/gt_' + name +'.data'
    gt_index = pickle.load(open(path_gt,'rb'))
    m = size[name][0]  # I145，S512,PU610，P1096
    n = size[name][1]  # I145，S217,PU340，P715
    # print('MN', m, n)
    final_result = np.zeros((m, n), dtype=np.int8)
    final_result = final_result.reshape(m * n)
    final_result[gt_index] = pre + 1
    final_result = final_result.reshape(m, n)

    hsi_pic = np.zeros((final_result.shape[0], final_result.shape[1], 3))
    for i in range(final_result.shape[0]):
        for j in range(final_result.shape[1]):
            if final_result[i][j] == 0:
                hsi_pic[i, j, :] = [x / 255.0 for x in [0, 0, 0]]  # 黑色
            if final_result[i][j] == 1:
                hsi_pic[i, j, :] = [x / 255.0 for x in [135, 206, 250]]  # 黑色
            if final_result[i][j] == 2:
                hsi_pic[i, j, :] = [x / 255.0 for x in [139, 0, 0]]
            if final_result[i][j] == 3:
                hsi_pic[i, j, :] = [x / 255.0 for x in [255, 99, 71]]
            if final_result[i][j] == 4:
                hsi_pic[i, j, :] = [x / 255.0 for x in [160, 82, 45]]
            if final_result[i][j] == 5:
                hsi_pic[i, j, :] = [x / 255.0 for x in [205, 133, 63]]
            if final_result[i][j] == 6:
                hsi_pic[i, j, :] = [x / 255.0 for x in [85, 107, 47]]
            if final_result[i][j] == 7:
                hsi_pic[i, j, :] = [x / 255.0 for x in [95, 158, 160]]
            if final_result[i][j] == 8:
                hsi_pic[i, j, :] = [x / 255.0 for x in [34, 139, 34]]
            if final_result[i][j] == 9:
                hsi_pic[i, j, :] = [x / 255.0 for x in [0, 206, 209]]
            if final_result[i][j] == 10:
                hsi_pic[i, j, :] = [x / 255.0 for x in [176, 224, 230]]
            if final_result[i][j] == 11:
                hsi_pic[i, j, :] = [x / 255.0 for x in [70, 130, 180]]
            if final_result[i][j] == 12:
                hsi_pic[i, j, :] = [x / 255.0 for x in [100, 149, 237]]
            if final_result[i][j] == 13:
                hsi_pic[i, j, :] = [x / 255.0 for x in [138, 43, 226]]
            if final_result[i][j] == 14:
                hsi_pic[i, j, :] = [x / 255.0 for x in [218, 112, 214]]
            if final_result[i][j] == 15:
                hsi_pic[i, j, :] = [x / 255.0 for x in [255, 182, 193]]
            if final_result[i][j] == 16:
                hsi_pic[i, j, :] = [x / 255.0 for x in [220, 20, 60]]
            if final_result[i][j] == 17:
                hsi_pic[i, j, :] = [x / 255.0 for x in [25, 25, 112]]
            if final_result[i][j] == 18:
                hsi_pic[i, j, :] = [x / 255.0 for x in [30, 144, 255]]
            if final_result[i][j] == 19:
                hsi_pic[i, j, :] = [x / 255.0 for x in [95, 158, 160]]

    resultmap_path = './result/'+name+'/'+name+'_' + str(num_labeled) + '_s_'+str(method)+'.png'
    classification_map(hsi_pic, final_result, 300, resultmap_path)

def metric(name, predict):
    path_gt = './data/gt/' + name + '_gt.pickle'
    gt = pickle.load(open(path_gt, 'rb'))
    C = metrics.confusion_matrix(gt, predict)
    A = np.diag(C) / np.sum(C, 1, dtype=np.float)
    k = metrics.cohen_kappa_score(gt, predict)
    OA = metrics.accuracy_score(gt, predict)
    CLASS_NUM = np.max(gt) + 1
    num_per_class = np.bincount(gt)
    print('num_per_class:', num_per_class)
    print('num_per_class - shot:', num_per_class - 5)
    print('CLASS_NUM:', CLASS_NUM, '\nC:', C, '\nOA:', OA, '\nA:', A, '\nk:', k)
    for i in range(CLASS_NUM):
        print ("Class " + str(i) + ": " + "{:.2f}".format(100 * A[i]))

def metric_DCFSL(name, predict):
    if name=='PU':
        gt_path = './dataset/PaviaU_gt.mat'
        groundt = scipy.io.loadmat(gt_path)
        gt = groundt['paviaU_gt']
    elif name=='PC':
        gt_path = './dataset/Pavia_gt.mat'
        groundt = scipy.io.loadmat(gt_path)
        gt = groundt['pavia_gt']
    elif name=='IP':
        gt_path = './dataset/Indian_pines_gt.mat'
        groundt = scipy.io.loadmat(gt_path)
        gt = groundt['indian_pines_gt']
    elif name=='SV':
        gt_path = './dataset/Salinas_gt.mat'
        groundt = scipy.io.loadmat(gt_path)
        gt = groundt['salinas_gt']
    print(gt.shape, predict.shape)
    gt = np.reshape(gt, gt.shape[0]*gt.shape[1])
    predict = np.reshape(predict, predict.shape[0] * predict.shape[1])
    print(gt.shape, predict.shape)
    C = metrics.confusion_matrix(gt, predict)
    A = np.diag(C) / np.sum(C, 1, dtype=np.float)
    k = metrics.cohen_kappa_score(gt, predict)
    OA = metrics.accuracy_score(gt, predict)
    CLASS_NUM = np.max(gt) + 1
    num_per_class = np.bincount(gt)
    print('num_per_class:', num_per_class)
    print('num_per_class - shot:', num_per_class - 5)
    print('CLASS_NUM:', CLASS_NUM, '\nC:', C, '\nOA:', OA, '\nA:', A, '\nk:', k)
    for i in range(CLASS_NUM):
        print ("Class " + str(i) + ": " + "{:.2f}".format(100 * A[i]))


def main():
    name = 'SV'
    # path = './result/5-shot/' + name + '_5_s_DCFSL.pickle'
    # pre = pickle.load(open(path, 'rb'))
    pre = pickle.load(open('./result/SV_1_s_CFSL.pickle', 'rb'))
    prediction = pre[4:-4, 4:-4]
    metric_DCFSL(name, prediction)
    # draw_map(name, 'BMFSC', pre, 5)


if __name__=="__main__":
    main()