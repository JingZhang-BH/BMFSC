import pickle
import numpy as np
IP = pickle.load(open("data/indian_pines_corrected_all.pickle",'rb'))  #target
# print(IP.shape)

IP_data = IP['data']
IP_label = IP['Labels']
print(IP_data.shape, IP_label.shape)
print(min(IP_label), max(IP_label))

for i in range(16):
    print(np.array(np.where(IP_label == i)).shape)

# query_labels = IP_label - 1
# print(IP_label)
# print(query_labels)
# # Houston_bands = pickle.load(open("data/win9/contest20139_w_100b_100s.pickle",'rb'))
# # HBKC = pickle.load(open("data/win9/HBKC.pickle",'rb'))
# #
# for i in range(query_labels.shape[0]):
#     print(IP_label[i] == query_labels[i] + 1)
#
# def onehottify_2d_array(a):
#     """
#     https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
#     :param a: 2-dimensional array.
#     :return: 3-dim array where last dim corresponds to one-hot encoded vectors.
#     """
#
#     # https://stackoverflow.com/a/46103129/ @Divakar
#     def all_idx(idx, axis):
#         grid = np.ogrid[tuple(map(slice, idx.shape))]
#         grid.insert(axis, idx)
#         return tuple(grid)
#
#     num_columns = a.max() + 1
#     out = np.zeros(a.shape + (num_columns,), dtype=int)
#     out[all_idx(a, axis=2)] = 1
#     return out
#
# a = np.ones(5, dtype = np.int32)
# c = np.arange(5)
# d = a + c
# d = np.reshape(d, (1, 5))
# print(d.shape)
# b = onehottify_2d_array(d)
# print(b.shape)
# print(b)