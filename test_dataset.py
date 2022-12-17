import os
import sys
import numpy as np
import pickle



def onehottify_2d_array(a):
    """
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    :param a: 2-dimensional array.
    :return: 3-dim array where last dim corresponds to one-hot encoded vectors.
    """

    # https://stackoverflow.com/a/46103129/ @Divakar
    def all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    num_columns = a.max() + 1
    out = np.zeros(a.shape + (num_columns,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out

def onehottify_way_determined(a, way):
    """
        https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
        :param a: 2-dimensional array.
        :return: 3-dim array where last dim corresponds to parameter 'way'.
        """

    # https://stackoverflow.com/a/46103129/ @Divakar
    def all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    num_columns = way
    out = np.zeros(a.shape + (num_columns,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out

class DataSet(object):

    def __init__(self, path, seed):
        """
        Constructs a HSI dataset for use in episodic training.
        :param path: Path to miniImageNet data files.
        :param seed: Random seed to reproduce batches.
        """
        np.random.seed(seed)

        self.image_height = 9
        self.image_width = 9
        self.bands = 100

        tail = '_all.pickle'
        path_test = os.path.join(path, 'salinas_corrected'+tail)
        self.test_set = pickle.load(open(path_test, 'rb'))

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_bands(self):
        return self.bands

    def get_sample_num(self):
        images = self.test_set
        test_data = images['data']
        sample_num = test_data.shape[0]
        return sample_num

    def get_iter(self):
        samper_num_per_batch = self.get_samper_num_per_batch()
        images = self.test_set
        test_data = images['data']
        sample_num = test_data.shape[0]
        iter_num = sample_num // samper_num_per_batch
        res_num = sample_num % samper_num_per_batch
        return iter_num, res_num

    def get_samper_num_per_batch(self):
        samper_num_per_batch = 500
        return samper_num_per_batch

    def _sample_batch(self, test_data, test_label, iter):
        """
        Sample a k-shot batch from images.
        :param images: Data to sample from [way, samples, h, w, c] (either of train, val, test)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: A list [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * (shot or eval_samples), h, w, c]
            * Labels: [tasks_per_batch, way * (shot or eval_samples), way]
                      (one-hot encoded in last dim)
        """
        # Set up empty arrays
        i, r = self.get_iter()
        samper_num_per_batch = self.get_samper_num_per_batch()
        if iter < i:
            query_images = test_data[samper_num_per_batch * iter:samper_num_per_batch * (iter + 1), :, :, :]
            query_labels = test_label[samper_num_per_batch * iter:samper_num_per_batch * (iter + 1)]
        elif iter == i:
            query_images = test_data[samper_num_per_batch * iter:samper_num_per_batch * iter + r, :, :, :]
            query_labels = test_label[samper_num_per_batch * iter:samper_num_per_batch * iter + r]
        # print(query_images.shape, query_labels.shape)
        return [query_images, query_labels]


    def _shuffle_batch(self, train_images, train_labels):
        """
        Randomly permute the order of the second column
        :param train_images: [tasks_per_batch, way * shot, height, width, channels]
        :param train_labels: [tasks_per_batch, way * shot, way]
        :return: permuted images and labels.
        """
        for i in range(train_images.shape[0]):
            permutation = np.random.permutation(train_images.shape[1])
            train_images[i, ...] = train_images[i, permutation, ...]
            train_labels[i, ...] = train_labels[i, permutation, ...]
        return train_images, train_labels

    def get_batch(self, shot, way):
        """
        Returns a batch of tasks from miniImageNet. Values are np.float32 and scaled to [0,1]
        :param source: one of `train`, `test`, `validation` (i.e. from which classes to pick)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * shot, height, width, channels]
            * Labels: [tasks_per_batch, way * shot, way]
                      (one-hot encoded in last dim)
        """

        images = self.test_set
        test_data = images['data']
        test_label = images['lable']
        iter, res = self.get_iter()
        samper_num_per_batch = self.get_samper_num_per_batch()

        support_images = np.empty((1, way, shot, self.image_height, self.image_width, self.bands), dtype=np.float32)
        support_labels = np.empty((1, way, shot), dtype=np.int32)
        query_images = np.empty((iter, 1, samper_num_per_batch, self.image_height, self.image_width, self.bands), dtype=np.float32)
        query_images_last = np.empty((1, res, self.image_height, self.image_width, self.bands), dtype=np.float32)
        query_labels = np.empty((iter, 1, samper_num_per_batch), dtype=np.int32)
        query_labels_last = np.empty((1, res), dtype=np.int32)

        # sample_idx_all = np.zeros([way, shot])
        for i in range(way):
            idx_tmp = np.array(np.where(test_label == i))
            idx_tmp = np.reshape(idx_tmp, (idx_tmp.shape[1]))
            idx = np.random.choice(idx_tmp, size=shot, replace=False)
            # sample_idx_all[i] = idx
            support_images[0][i] = test_data[idx, :, :, :]

        labels_tmp = np.arange(way)
        support_images = np.reshape(support_images, [1, way * shot, self.image_height, self.image_width, self.bands])
        support_images = support_images.astype(dtype=np.float32)
        support_labels[0] = np.expand_dims(labels_tmp, axis=1)
        support_labels = np.reshape(support_labels, [1, way * shot])
        support_labels = onehottify_2d_array(support_labels)


        query_labels_onehot = np.empty((iter, 1, samper_num_per_batch, way), dtype=np.int32)
        for i in range(iter):
            query_images[i][0], query_labels[i][0] = self._sample_batch(test_data, test_label, i)
            query_labels_onehot[i][0] = onehottify_way_determined(np.reshape(query_labels[i][0], (1, samper_num_per_batch)), way)

        query_images_last[0], query_labels_last[0] = self._sample_batch(test_data, test_label, iter)
        query_labels_last = onehottify_way_determined(query_labels_last, way)

        return [support_images, support_labels, query_images, query_labels_onehot, query_images_last, query_labels_last]

