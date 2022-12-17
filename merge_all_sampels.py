import pickle
import numpy as np

win_size = 9
root = "data/all/w" + str(win_size) + "/"
result_path = root + "HBKC_w" + str(win_size) + "_all.pickle"
root1 = "data/win" + str(win_size) + "/"
tail = "_w_100b_100s.pickle"
Houston = pickle.load(open(root1 + "contest2013" + str(win_size) + tail,'rb'))
Chikusei = pickle.load(open(root1 + "Chikusei" + str(win_size) + tail,'rb'))
KSC = pickle.load(open(root1 + "KSC" + str(win_size) + tail,'rb'))
Botswana = pickle.load(open(root1 + "Botswana" + str(win_size) + tail,'rb'))
print(Houston.shape, Chikusei.shape, KSC.shape, Botswana.shape)

train = np.zeros((Houston.shape[0] + Chikusei.shape[0] + Botswana.shape[0] + KSC.shape[0], 100, win_size, win_size, 100))

idx = 0
for i in range(Houston.shape[0]):
     train[idx] = Houston[i]
     idx+=1

print(idx)
for i in range(Botswana.shape[0]):
     train[idx] = Botswana[i]
     idx += 1

print(idx)
for i in range(KSC.shape[0]):
     train[idx] = KSC[i]
     idx += 1

print(idx)
for i in range(Chikusei.shape[0]):
     train[idx] = Chikusei[i]
     idx += 1

# print(idx)
print(train.shape)
way = Houston.shape[0] + Chikusei.shape[0] + Botswana.shape[0] + KSC.shape[0]
sample_num = way * 100
train = np.reshape(train, (sample_num, win_size, win_size, 100))
train_labels = np.empty((1, way, 100), dtype=np.int32)
labels_tmp = np.arange(way)
train_labels[0] = np.expand_dims(labels_tmp, axis=1)
print('train_labels', train_labels.shape)
train_labels = np.reshape(train_labels, [sample_num])

print(train.shape, train_labels.shape)
dataset = {"data":train, 'lable':train_labels}
pickle.dump(dataset, open(result_path, 'wb'), protocol=2)
