import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from collections import defaultdict
import pickle
import h5py


def get_data():
    root = './dataset/'
    root1 = './data/'

    band_selected = pickle.load(open('./data/bands.pickle', 'rb'))
    winsize = 9
    bands = 100
    samplesperclass = 300

    # im_, gt_ = 'salinas_corrected', 'salinas_gt'
    # im_, gt_ = 'contest2013', 'contest2013_gt'
    # im_, gt_ = 'pavia', 'pavia_gt'
    # im_, gt_ = 'paviaU', 'paviaU_gt'
    # im_, gt_ = 'Chikusei', 'Chikusei_gt'  #chi,GroundT
    im_, gt_ = 'indian_pines_corrected', 'indian_pines_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'


    band_chosen = {"salinas_corrected": 'S', 'contest2013': 'H', 'pavia': 'PC', 'paviaU': 'PU',
                   'Chikusei': 'C', 'indian_pines_corrected': 'I', 'Botswana': 'B', 'KSC': 'K'}

    result_path = root1 + im_ + str(winsize) + '_w_' + str(bands) + 'b_' + str(samplesperclass) + 's' + '.pickle'
    result_path1 = root1 + im_ + '_all' + '.pickle'

    # input_mat = h5py.File('./dataset/Chikusei.mat', 'r')
    # input_image = input_mat['chikusei'][:].transpose()
    # Chikusei_gt = loadmat('./dataset/Chikusei_gt.mat')
    # gt_hsi = Chikusei_gt['GT']
    # output_image = gt_hsi[0, 0]['gt']
    # print("bands:", band_selected['C'])
    # input_image = np.array(input_image)
    # input_image = input_image[:, :, band_selected['C']]
    # input_image = np.array(input_image)
    # print(input_image.shape)
    # print(output_image.shape)


    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    input_image = loadmat(img_path)[im_]
    output_image = loadmat(gt_path)[gt_]
    print(input_image.shape, output_image.shape)
    print("bands:", band_selected[band_chosen[im_]])
    input_image = np.array(input_image)
    input_image = input_image[:,:, band_selected[band_chosen[im_]]]

    dataset = defaultdict(list)
    #label = dict()
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            if output_image[i][j] != 0:
                data = []
                for r in range(-(winsize//2), winsize//2+1):
                    for c in range(-(winsize//2), winsize//2+1):
                        # print('i,r,j,c:',i,r,j,c)
                        if 0 <= i + r < input_image.shape[0] and 0 <= j + c < input_image.shape[1]:
                            data.extend(list(input_image[i + r][j + c])[:bands])
                            # print(len(list(input_image[i+r][j+c])[:100]))
                        elif 0 <= i + r < input_image.shape[0]:
                            data.extend(list(input_image[i + r][j - c])[:bands])
                        elif 0 <= j + c < input_image.shape[1]:
                            data.extend(list(input_image[i - r][j + c])[:bands])
                        else:
                            data.extend(list(input_image[i - r][j - c])[:bands])
                dataset[output_image[i][j]].append(data)
                #print(len(data))
                #label.append(output_image[i][j])
    #print(len(dataset),len(label),len(dataset[0]))
    #print(dataset.keys())

    # sorted(dataset.keys())
    num = 0
    for i in dataset.keys():
        print(i,len(dataset[i]))
        num += len(dataset[i])
    print(num)
    idx = 0
    dataset_array = np.zeros((num, winsize, winsize, bands), dtype=np.float32)
    label = np.zeros((num), dtype=np.int32)
    batch = num // 500
    sample_idx = np.zeros((len(dataset.keys()), batch+1), dtype=np.int32)
    print(sample_idx.shape)
    for j in range(batch + 1):
        for i in dataset.keys():
            sample_num = len(dataset[i]) // batch
            if j<batch:
                sample_idx[i - 1][j + 1] = j * sample_num + sample_num
                for k in range(sample_idx[i-1][j], sample_idx[i-1][j+1]):
                    tmp = np.array(dataset[i][k])
                    tmp = tmp.reshape(winsize, winsize, bands)
                    dataset_array[idx] = tmp
                    label[idx] = i-1
                    idx += 1
            elif j==batch:
                for k in range(sample_idx[i-1][j], len(dataset[i])):
                    tmp = np.array(dataset[i][k])
                    tmp = tmp.reshape(winsize, winsize, bands)
                    dataset_array[idx] = tmp
                    label[idx] = i-1
                    idx += 1
    print(sample_idx)
    print(idx)
    print(dataset_array.shape)
    print(label.shape)
    dict = {'data': dataset_array, 'Labels': label}

    pickle.dump(dict, open(result_path1, 'wb'), protocol=2)

    # dataset_mini = {}
    # for i in dataset.keys():
    #     if (len(dataset[i])) >= samplesperclass:
    #         # dataset_mini[num] = dataset[i + 1]
    #         dataset_mini[i] = dataset[i][:samplesperclass]
    #         print("dataset_mini[num] = dataset[i]", num, i)
    #     else:
    #         temp = np.array(dataset[i])
    #         tmp1 = np.tile(temp, (5, 1))
    #         dataset_mini[i] = tmp1[:samplesperclass]
    #         print("dataset_mini[num] = dataset[i]", num, i)
    #         print(len(dataset_mini[num]))
    #
    # print(dataset_mini.keys())
    # for i in dataset_mini.keys():
    #     print("mini", i, len(dataset_mini[i]))
    #     num += len(dataset_mini[i])
    #
    # dataset_array = np.zeros((len(dataset.keys()), samplesperclass, winsize, winsize, bands), dtype=np.float32)
    # for i in range(len(dataset_mini.keys())):
    #     print(i)
    #     out = np.zeros((samplesperclass, winsize, winsize, bands), dtype=np.float32)
    #     for j in range(len(dataset_mini[i + 1])):
    #         tmp = np.array(dataset_mini[i + 1][j])
    #         tmp = tmp.reshape(winsize, winsize, bands)
    #         out[j] = tmp
    #     dataset_array[i] = out
    #
    # print(dataset_array.shape)
    #
    # pickle.dump(dataset_array, open(result_path, 'wb'), protocol=2)

    return dataset

if __name__ == '__main__':
    get_data()