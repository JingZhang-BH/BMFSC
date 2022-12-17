import tensorflow as tf
from block import conv3d_pool_block, conv2d_pool_block, conv1x1_block, conv2d_transpose_layer, dense_layer, dense_block

def extract_features_3d_nor(images, output_size, use_batch_norm, dropout_keep_prob):
    # print('&&&&&&&& images.shape: ', images.shape)
    images = tf.reshape(images, [-1, images.shape[1], images.shape[2], images.shape[3], 1])
    # images = tf.reshape(images, [-1, images.shape[3], images.shape[1], images.shape[2]])

    # print('&&&&&&&& images.shape: ', images.shape)
    h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_01')  # filters = 64
    # print('&&&&&&&& h.shape: ', h.shape)
    # h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_02')
    # h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1') #filters = 64
    # print('&&&&&&&& h.shape: ', h.shape)
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(1, 1, 1), padding='valid', name=('pool_1'))
    h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_03')
    # h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1') #filters = 64
    # print('&&&&&&&& h.shape: ', h.shape)
    # print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_2')
    # print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(2, 2, 1), padding='valid', name=('pool_1'))
    # print('&&&&&&&& h.shape: ', h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_3')
    # print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_4')
    # print('&&&&&&&& h.shape: ', h.shape)
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(2, 2, 1), padding='valid', name=('pool_2'))

    # print(h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_5')
    # print('&&&&&&&& h.shape: ', h.shape)

    # print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(1, 1, 1), padding='valid', name=('pool_2'))
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[1, 1, 2], strides=(1, 1, 2), padding='valid', name=('pool_3'))
    # print(h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_6')
    h = conv3d_pool_block(h, use_batch_norm, 8, dropout_keep_prob, 'valid', 'fe_block_7')
    # print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(2, 2, 2), padding='valid', name=('pool_3'))
    # print('&&&&&&&& h.shape: ', h.shape)
    # flatten output
    # h = tf.contrib.layers.flatten(h)
    # print('########################', images.shape)
    h = tf.contrib.layers.flatten(h)
    # print('&&&&&&&& h.shape: ', h.shape)
    # print("##########", h.shape)
    # print('feature_test', h.shape)
    # tmp = tf.reshape(h, (10, 192))
    #
    # h_print = tf.Print(tmp,[tmp, tmp.shape, 'feature'],
    #                                 message='Debug message:', summarize=20000)
    # h = h_print
    # print('get_features:45', h.shape)  #64,64,64,64:(?,25600)    64,32,16,8:(?.3200)
    return h


def extract_features_3d_9(images, output_size, use_batch_norm, dropout_keep_prob):
    # images = tf.reshape(images, [-1, images.shape[1], images.shape[2], images.shape[3], 1])
    images = tf.expand_dims(images, -1)

    h = conv3d_pool_block(images, use_batch_norm, 128, dropout_keep_prob, 'same', 'fe_block_01')  # filters = 64
    h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'same', 'fe_block_02')
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 4], strides=(1, 1, 2), padding='valid', name=('pool_1'))

    h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'same', 'fe_block_03')
    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'same', 'fe_block_2')
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 4], strides=(1, 1, 2), padding='valid', name=('pool_2'))

    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'same', 'fe_block_3')
    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'same', 'fe_block_4')
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 4], strides=(1, 1, 2), padding='valid', name=('pool_3'))

    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'same', 'fe_block_5')
    h = conv3d_pool_block(h, use_batch_norm, 8, dropout_keep_prob, 'same', 'fe_block_7')
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 4], strides=(2, 2, 4), padding='valid', name=('pool_4'))

    h = tf.contrib.layers.flatten(h)
    return h

def extract_features_DFSL(images, output_size, use_batch_norm, dropout_keep_prob):
    images = tf.expand_dims(images, -1)

    h_conv1_1 = conv3d_pool_block(images, use_batch_norm, 8, dropout_keep_prob, 'same', 'h_conv1_1')  # filters = 64
    h_conv1_2 = conv3d_pool_block(h_conv1_1, use_batch_norm, 8, dropout_keep_prob, 'same', 'h_conv1_2')
    h_conv1_3 = conv3d_pool_block(h_conv1_2, use_batch_norm, 8, dropout_keep_prob, 'same', 'h_conv1_3')
    h_conv1 = conv3d_pool_block(h_conv1_3, use_batch_norm, 8, dropout_keep_prob, 'same', 'h_conv1_4') + h_conv1_1
    print('h_conv1.shape', h_conv1.shape)

    h_pool1 = tf.layers.max_pooling3d(inputs=h_conv1, pool_size=[2, 2, 4], strides=(2, 2, 4), padding='same', name=('pool_1'))
    print('h_pool1.shape', h_pool1.shape)

    h_conv2_1 = conv3d_pool_block(h_pool1, use_batch_norm, 16, dropout_keep_prob, 'same', 'h_conv2_1')  # filters = 64
    h_conv2_2 = conv3d_pool_block(h_conv2_1, use_batch_norm, 16, dropout_keep_prob, 'same', 'h_conv2_2')
    h_conv2_3 = conv3d_pool_block(h_conv2_2, use_batch_norm, 16, dropout_keep_prob, 'same', 'h_conv2_3')
    h_conv2 = conv3d_pool_block(h_conv2_3, use_batch_norm, 16, dropout_keep_prob, 'same', 'h_conv2_4') + h_conv2_1
    print('h_conv2.shape', h_conv2.shape)

    h_pool2 = tf.layers.max_pooling3d(inputs=h_conv2, pool_size=[2, 2, 4], strides=(2, 2, 4), padding='same',
                                      name=('pool_2'))
    print('h_pool2.shape', h_pool2.shape)

    h_conv3 = conv3d_pool_block(h_pool2, use_batch_norm, 32, dropout_keep_prob, 'VALID', 'h_conv3')
    print('h_conv3.shape', h_conv3.shape)

    y_conv = tf.reshape(h_conv3, [-1, 5 * 32])

    print('y_conv.shape', y_conv.shape)

    return y_conv

def extract_features_3d_99(images, output_size, use_batch_norm, dropout_keep_prob):
    # print('&&&&&&&& images.shape: ', images.shape)
    images = tf.reshape(images, [-1, images.shape[1], images.shape[2], images.shape[3], 1])
    # images = tf.reshape(images, [-1, images.shape[3], images.shape[1], images.shape[2]])

    print('&&&&&&&& images.shape: ', images.shape)
    h = conv3d_pool_block(images, use_batch_norm, 128, dropout_keep_prob, 'valid', 'fe_block_01')  # filters = 64
    # print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_02')
    # h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1') #filters = 64
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(1, 1, 1), padding='valid', name=('pool_1'))
    h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_03')
    # h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1') #filters = 64
    print('&&&&&&&& h.shape: ', h.shape)
    # print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_2')
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(2, 2, 1), padding='valid', name=('pool_1'))
    print('&&&&&&&& h.shape: ', h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_3')
    print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_4')
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(2, 2, 1), padding='valid', name=('pool_2'))
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[1, 1, 2], strides=(1, 1, 1), padding='valid', name=('pool_2'))
    # print(h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_5')
    print('&&&&&&&& h.shape: ', h.shape)
    # h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_6')
    # print('&&&&&&&& h.shape: ', h.shape)
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[1, 1, 2], strides=(1, 1, 2), padding='valid', name=('pool_3'))
    # print(h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 8, dropout_keep_prob, 'valid', 'fe_block_7')
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(2, 2, 2), padding='valid', name=('pool_3'))
    print('&&&&&&&& h.shape: ', h.shape)
    # flatten output
    # h = tf.contrib.layers.flatten(h)
    # print('########################', images.shape)
    h = tf.contrib.layers.flatten(h)
    print('&&&&&&&& h.shape: ', h.shape)
    # print("##########", h.shape)
    # print('feature_test', h.shape)
    # tmp = tf.reshape(h, (10, 192))
    #
    # h_print = tf.Print(tmp,[tmp, tmp.shape, 'feature'],
    #                                 message='Debug message:', summarize=20000)
    # h = h_print
    # print('get_features:45', h.shape)  #64,64,64,64:(?,25600)    64,32,16,8:(?.3200)
    return h

def extract_features_3d(images, output_size, use_batch_norm, dropout_keep_prob):
    # print('&&&&&&&& images.shape: ', images.shape)
    images = tf.reshape(images, [-1, images.shape[1], images.shape[2], images.shape[3], 1])
    # images = tf.reshape(images, [-1, images.shape[3], images.shape[1], images.shape[2]])

    print('&&&&&&&& images.shape: ', images.shape)
    h = conv3d_pool_block(images, use_batch_norm, 256, dropout_keep_prob, 'valid', 'fe_block_01')  # filters = 64
    print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 128, dropout_keep_prob, 'valid', 'fe_block_02')
    # h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1') #filters = 64
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.average_pooling3d(inputs=h, pool_size=[1, 1, 2], strides=(1, 1, 1), padding='valid', name=('pool_1'))
    h = conv3d_pool_block(h, use_batch_norm, 128, dropout_keep_prob, 'valid', 'fe_block_03')
    # h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1') #filters = 64
    print('&&&&&&&& h.shape: ', h.shape)
    print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_2')
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.average_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(1, 1, 1), padding='valid', name=('pool_1'))
    # print('&&&&&&&& h.shape: ', h.shape)

    h = conv3d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_3')
    print('&&&&&&&& h.shape: ', h.shape)
    h = conv3d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_4')
    print('&&&&&&&& h.shape: ', h.shape)
    h = tf.layers.average_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(1, 1, 1), padding='valid', name=('pool_2'))
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[1, 1, 2], strides=(1, 1, 1), padding='valid', name=('pool_2'))
    # print(h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_5')
    print('&&&&&&&& h.shape: ', h.shape)
    # h = conv3d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_6')
    # print('&&&&&&&& h.shape: ', h.shape)
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[1, 1, 2], strides=(1, 1, 2), padding='valid', name=('pool_3'))
    # print(h.shape)

    h = conv3d_pool_block(h, use_batch_norm, 8, dropout_keep_prob, 'valid', 'fe_block_7')
    h = tf.layers.average_pooling3d(inputs=h, pool_size=[2, 2, 2], strides=(1, 1, 2), padding='valid', name=('pool_3'))
    print('&&&&&&&& h.shape: ', h.shape)
    # flatten output
    # h = tf.contrib.layers.flatten(h)
    # print('########################', images.shape)
    h = tf.contrib.layers.flatten(h)
    print('&&&&&&&& h.shape: ', h.shape)
    # print("##########", h.shape)
    # print('feature_test', h.shape)
    # tmp = tf.reshape(h, (10, 192))
    #
    # h_print = tf.Print(tmp,[tmp, tmp.shape, 'feature'],
    #                                 message='Debug message:', summarize=20000)
    # h = h_print
    # print('get_features:45', h.shape)  #64,64,64,64:(?,25600)    64,32,16,8:(?.3200)
    return h


def no_extract_features(images, output_size, use_batch_norm, dropout_keep_prob):
    """
    Based on the architecture described in 'Matching Networks for One-Shot Learning'
    http://arxiv.org/abs/1606.04080.pdf.

    :param images: batch of images.
    :param output_size: dimensionality of the output features.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :return: features.
    """

    print('########################', images.shape)
    # images = tf.array(images)
    # print('########################', images.shape)
    h = images[:,4, 4, :]
    h = tf.contrib.layers.flatten(h)
    print("##########", h.shape)
    # tmp = tf.reshape(h, [80, 46])
    #
    # h_print = tf.Print(tmp,[tmp, tmp.shape, 'feature'],
    #                            message='feature:', summarize=20000)
    # h = h_print
    return h

def extract_features_2d(images, output_size, use_batch_norm, dropout_keep_prob):
    h = conv2d_pool_block(images, use_batch_norm, 64, dropout_keep_prob, 'valid', 'fe_block_1')  # filters = 64
    # h = tf.layers.max_pooling3d(inputs=h, pool_size=[2, 1, 4], strides=(1, 1, 4), padding='valid', name=('pool_1'))
    # print('HERE:get_feature 37')
    h = conv2d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_2')
    h = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2], strides=2, padding='valid', name=('2d_pool'))
    # h = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2, 4], strides=(2, 2, 4), padding='valid', name=('pool_1'))
    # print(h.shape)

    h = conv2d_pool_block(h, use_batch_norm, 32, dropout_keep_prob, 'valid', 'fe_block_3')
    h = conv2d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_4')
    h = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2,], strides=2, padding='valid', name=('2d_pool_2'))
    # print(h.shape)

    h = conv2d_pool_block(h, use_batch_norm, 16, dropout_keep_prob, 'valid', 'fe_block_5')
    h = conv2d_pool_block(h, use_batch_norm, 8, dropout_keep_prob, 'valid', 'fe_block_6')


    # 4X conv2d + pool blocks
    # h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'valid','fe_block_2d_1')
    # h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid','fe_block_2d_2')
    # h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid','fe_block_2d_3')
    # h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_2d_4')

    # flatten output
    h = tf.contrib.layers.flatten(h)

    # dense layer
    # h = dense_block(h, output_size, use_batch_norm, dropout_keep_prob, 'fe_dense')

    return h

# def extract_features(images, output_size, use_batch_norm, dropout_keep_prob):
#     """
#     Based on the architecture described in 'Matching Networks for One-Shot Learning'
#     http://arxiv.org/abs/1606.04080.pdf.
#
#     :param images: batch of images.
#     :param output_size: dimensionality of the output features.
#     :param use_batch_norm: whether to use batch normalization or not.
#     :param dropout_keep_prob: keep probability parameter for dropout.
#     :return: features.
#     """
#
#     # # 4X conv2d + pool blocks
#     # h = conv2d_pool_block(images, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_1')
#     # h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_2')
#     # h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_3')
#     # h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_4')
#     #
#     # # flatten output
#     # h = tf.contrib.layers.flatten(h)
#     #
#     # return h
#
#     """
#     for Pavia, 8,8,8,16,16,32
#     for IP,Sa, 64,32,32,16,16,8
#     """
#
#
#     # 5X conv2d + pool blocks
#     # print("get_features:30", images.shape, output_size)
#     # h = conv1x1_block(images, use_batch_norm, dropout_keep_prob, 'valid', 'fe_block_1')
#     # print("THERE:", h.shape)
#
#     # print('get_features:33', images.shape)
#     images = tf.reshape(images, [-1, images.shape[1], images.shape[2], images.shape[3], 1])
#     # print('get_features:35', images.shape)
#     h_conv1_1 = conv3d_pool_block(images, use_batch_norm, 8, dropout_keep_prob, 'same', 'fe_block_01') #filters = 64
#     # print('HERE:get_feature 37')
#     h_conv1_2 = conv3d_pool_block(h_conv1_1, use_batch_norm, 8, dropout_keep_prob, 'same', 'fe_block_02')
#     h_conv1_3 = conv3d_pool_block(h_conv1_2, use_batch_norm, 8, dropout_keep_prob, 'same', 'fe_block_03')
#     h_conv1 = conv3d_pool_block(h_conv1_3, use_batch_norm, 8, dropout_keep_prob, 'same', 'fe_block_0')
#     h_conv1 += h_conv1_1
#     h_pool1 = tf.layers.max_pooling3d(inputs=h_conv1, pool_size=[2, 2, 4], strides=(2, 2, 4), padding='same', name=('pool_1'))
#     print(h_pool1.shape)
#
#     h_conv2_1 = conv3d_pool_block(h_pool1, use_batch_norm, 16, dropout_keep_prob, 'same', 'fe_block_11')  # filters = 64
#     # print('HERE:get_feature 37')
#     h_conv2_2 = conv3d_pool_block(h_conv2_1, use_batch_norm, 16, dropout_keep_prob, 'same', 'fe_block_12')
#     h_conv2_3 = conv3d_pool_block(h_conv2_2, use_batch_norm, 16, dropout_keep_prob, 'same', 'fe_block_13')
#     h_conv2 = conv3d_pool_block(h_conv2_3, use_batch_norm, 16, dropout_keep_prob, 'same', 'fe_block_1')
#     h_conv2 += h_conv2_1
#     h_pool2 = tf.layers.max_pooling3d(inputs=h_conv2, pool_size=[2, 2, 4], strides=(2, 2, 4), padding='same',name=('pool_2'))
#     print(h_pool2.shape)
#
#     h_conv3 = tf.keras.layers.Conv3D(32, kernel_size = (3,3,3),strides=[1, 1, 1], padding='VALID')(h_pool2)
#     # print('HERE:get_feature 37')
#     print(h_conv3.shape)
#
#     # h = conv3d_pool_block(h, use_batch_norm, 8, dropout_keep_prob, 'valid', 'fe_block_7')
#
#     # flatten output
#     h = tf.contrib.layers.flatten(h_conv3)
#     # print('get_features:45', h.shape)  #64,64,64,64:(?,25600)    64,32,16,8:(?.3200)
#     return h
