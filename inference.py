import tensorflow as tf
from block import dense_layer, sample_normal


def inference_block(inputs, d_theta, output_units, name):
    """
    Three dense layers in sequence.
    :param inputs: batch of inputs.
    :param d_theta: dimensionality of the intermediate hidden layers.
    :param output_units: dimensionality of the output.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    h = dense_layer(inputs, d_theta, tf.nn.elu, True, name + '1')
    h = dense_layer(h, d_theta, tf.nn.elu, True, name + '2')
    h = dense_layer(h, output_units, None, True, name + '3')
    # print('inference: 17',h.shape)
    return h


def infer_classifier(features, labels, d_theta, num_classes):
    """
    Infer a linear classifier by concatenating vectors for each class.
    :param features: tensor (tasks_per_batch x num_features) feature matrix
    :param labels:  tensor (tasks_per_batch x num_classes) one-hot label matrix
    :param d_theta: Integer number of features on final layer before classifier.
    :param num_classes: Integer number of classes per task.
    :return: Dictionary containing output classifier layer (including means and
    :        log variances for weights and biases).
    """

    classifier = {}
    class_weight_means = []
    class_weight_logvars = []
    class_bias_means = []
    class_bias_logvars = []
    # print('inference 37:  labels', labels.shape)
    for c in range(num_classes):
        class_mask = tf.equal(tf.argmax(labels, 1), c)  #tf.argmax在axis=1的时候，将每一行最大元素所在的索引记录下来
        # class_mask = tf.equal(labels, c)
        class_features = tf.boolean_mask(features, class_mask)
        # print('inference: 45:  class_features', class_features.shape)  # (?,3200)

        # Pool across dimensions
        nu = tf.expand_dims(tf.reduce_mean(class_features, axis=0), axis=0)  #均值
        # print('inference: 43', nu.shape)

        class_weight_means.append(inference_block(nu, d_theta, d_theta, 'weight_mean'))
        class_weight_logvars.append(inference_block(nu, d_theta, d_theta, 'weight_log_variance'))
        class_bias_means.append(inference_block(nu, d_theta, 1, 'bias_mean'))
        class_bias_logvars.append(inference_block(nu, d_theta, 1, 'bias_log_variance'))


    classifier['weight_mean'] = tf.transpose(tf.concat(class_weight_means, axis=0))
    classifier['bias_mean'] = tf.reshape(tf.concat(class_bias_means, axis=1), [num_classes, ])
    classifier['weight_log_variance'] = tf.transpose(tf.concat(class_weight_logvars, axis=0))
    classifier['bias_log_variance'] = tf.reshape(tf.concat(class_bias_logvars, axis=1), [num_classes, ])
    return classifier

def _post_process(pooled, units):
    """
    Process a pooled variable through 2 dense layers
    :param pooled: tensor of rank (1 x num_features).
    :param units: integer number of output features.
    :return: tensor of rank (1 x units)
    """
    h = dense_layer(pooled, units, tf.nn.elu, True, 'post_process_dense_1')
    h = dense_layer(h, units, tf.nn.elu, True, 'post_process_dense_2')

    return h
