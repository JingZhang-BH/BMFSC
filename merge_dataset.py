import pickle
import numpy as np

win_size = 13
root = "data/win" + str(win_size) + "/"
result_path = root + "HBKC_w" + str(win_size) + ".pickle"
Houston = pickle.load(open(root + "contest2013" + str(win_size) + "_w_100b_100s.pickle",'rb'))
Chikusei = pickle.load(open(root + "Chikusei" + str(win_size) + "_w_100b_100s.pickle",'rb'))
KSC = pickle.load(open(root + "KSC" + str(win_size) + "_w_100b_100s.pickle",'rb'))
Botswana = pickle.load(open(root + "Botswana" + str(win_size) + "_w_100b_100s.pickle",'rb'))
print(Houston.shape[0], Chikusei.shape[0], KSC.shape[0], Botswana.shape[0])
# print(Chikusei.shape[0], KSC.shape[0], Botswana.shape[0])

# train = np.zeros((Chikusei.shape[0] + KSC.shape[0] + Botswana.shape[0], smp_num_per_class, win, win, 100))
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

pickle.dump(train, open(result_path, 'wb'), protocol=2)
