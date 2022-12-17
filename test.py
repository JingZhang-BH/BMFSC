import numpy as np
import tensorflow as tf
import argparse
from get_features import no_extract_features, extract_features_2d, extract_features_3d_9, extract_features_3d_99, extract_features_3d_nor
from inference import infer_classifier
from block import sample_normal, multinoulli_log_density, print_and_log, get_log_files, sample_theta
from dataset import DataSet
import time
from sklearn import metrics

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
parse_command_line: command line parser
"""

def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--dim", type=int, default=144,     #3*3 176    #9*9  168  #nor 176
                        help="Size of the feature extractor output.")
    parser.add_argument("--shot", type=int, default=5,
                        help="Number of training examples.")
    parser.add_argument("--way", type=int, default=16,
                        help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None,
                        help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    parser.add_argument("--test_way", type=int, default=None,
                        help="Way to be used at evaluation time. If not specified 'way' will be used.")
    parser.add_argument("--tasks_per_batch", type=int, default=1,
                        help="Number of tasks per batch.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples from q.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001,
                        help="Learning rate.")
    # todo 4: iterations
    parser.add_argument("--iterations", type=int, default=15000, help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
                        help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default='./checkpoint/2022-06-14-09-28-58/best_validation',
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=10,
                        help="Frequency of summary results (in iterations).")
    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir)

    print_and_log(logfile, "Options: %s\n" % args)

    # Load training and eval data
    data = DataSet(path="data/", seed=42)
    # set the feature extractor based on the dataset
    feature_extractor_fn = extract_features_3d_9
    # print('classify:109')

    # evaluation samples
    # todo 2：调整query个数
    eval_samples_train = 40
    eval_samples_test = 40

    # testing parameters
    test_iterations = 10
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               data.get_image_height(),
                                               data.get_image_width(),
                                               data.get_bands()],
                                  name='train_images')
    # print('train_images:', train_images.shape)
    test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              data.get_image_height(),
                                              data.get_image_width(),
                                              data.get_bands()],
                                 name='test_images')
    # todo D: 修改labels的占位符
    train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               args.way],
                                  name='train_labels')
    test_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              args.way],
                                 name='test_labels')
    # train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
    #                                            None,  # shot
    #                                            1],
    #                               name='train_labels')
    # test_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
    #                                           None,  # num test images
    #                                           1],
    #                              name='test_labels')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    # Relevant computations for a single task
    def evaluate_task(inputs):
        train_inputs, train_outputs, test_inputs, test_outputs = inputs
        # print('classify: 149   train_outputs:', train_outputs.shape, 'test_outputs:', test_outputs.shape)
        print('classify: 147, train_inputs:', train_inputs.shape)
        with tf.variable_scope('shared_features'):
            # extract features from train and test data
            # print("classify:150")
            # print('classify: 151, train_inputs:', train_inputs.shape)
            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.dim,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)
            print("classify:171  features_train", features_train.shape)
            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.dim,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)
            print('classify:176', features_test.shape)
        # Infer classification layer from q
        with tf.variable_scope('classifier'):
            classifier = infer_classifier(features_train, train_outputs, args.dim, args.way) # 字典

        # print('classify:178', classifier)
        # Local reparameterization trick
        # Compute parameters of q distribution over logits
        weight_mean, bias_mean = classifier['weight_mean'], classifier['bias_mean']
        # print('classify_144:', weight_mean.shape, bias_mean.shape)  #classify_144: (192, 16) (16,)

        #todo : 去掉
        weight_mean, bias_mean = sample_theta(weight_mean, bias_mean, K = 5, alpha = 2)
        # weight_mean_print = tf.Print(weight_mean, [weight_mean, weight_mean.shape, 'feature'],
        #                    message='weight_mean:', summarize=20000)
        # weight_mean = weight_mean_print

        weight_log_variance, bias_log_variance = classifier['weight_log_variance'], classifier['bias_log_variance']

        # print('classify:s2')
        print('features_test.shape', features_test.shape)
        print('weight_mean.shape', weight_mean.shape)
        logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean

        # print('logits_mean_test:', logits_mean_test.shape)

        logits_log_var_test = \
            tf.log(tf.matmul(features_test ** 2, tf.exp(weight_log_variance)) + tf.exp(bias_log_variance))
        # print('logits_log_var_test:', logits_log_var_test.shape)

        logits_sample_test = sample_normal(logits_mean_test, logits_log_var_test, args.samples)

        # print('classify:157', logits_sample_test.shape)

        #todo C:  确定此处不影响
        test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [args.samples, 1, 1])
        # print(tf.expand_dims(test_outputs, 0).shape)
        # print('classify 190   test_labels_tiled:', test_labels_tiled.shape)
        # # 't2' is a tensor of shape [2, 3, 5]
        # shape(expand_dims(t2, 0)) == > [1, 2, 3, 5]
        # shape(expand_dims(t2, 2)) == > [2, 3, 1, 5]
        # shape(expand_dims(t2, 3)) == > [2, 3, 5, 1]
        task_log_py = multinoulli_log_density(inputs=test_labels_tiled, logits=logits_sample_test)
        # print('task_log_py', task_log_py.shape)
        # print('tf.reduce_logsumexp(logits_sample_test, axis=0).shape', tf.reduce_logsumexp(logits_sample_test, axis=0).shape)
        # print('tf.log(L)', tf.log(L))

        averaged_predictions = tf.reduce_logsumexp(logits_sample_test, axis=0) - tf.log(L)
        task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                        tf.argmax(averaged_predictions, axis=-1)), tf.float32))
        task_score = tf.reduce_logsumexp(task_log_py, axis=0) - tf.log(L)
        task_loss = -tf.reduce_mean(task_score, axis=0)
        C = tf.contrib.metrics.confusion_matrix(tf.argmax(test_outputs, axis=-1), tf.argmax(averaged_predictions, axis=-1), num_classes=args.way, dtype=tf.int32)
        # print('C:', C.shape)
        A = tf.diag_part(C) / tf.reduce_sum(C, 1)
        # print('A:', A.shape)
        A_print = tf.Print(A, [A, A.shape, 'Acc_of_each_class'], message='A:', summarize=20000)
        A = A_print
        # print('A:', A.shape)
        return [task_loss, task_accuracy]

    # tf mapping of batch to evaluation function
    # print('classify:205, train_images:', train_images.shape)
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(train_images, train_labels, test_images, test_labels),
                             dtype=[tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_accuracies = batch_output
    # print('classify:225, train_images:', batch_output)
    loss = tf.reduce_mean(batch_losses)
    accuracy = tf.reduce_mean(batch_accuracies)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        def test_model(model_path, load=True):
            if load:
                saver.restore(sess, save_path=model_path)
            test_iteration = 0
            test_iteration_accuracy = []
            test_start = time.time()
            while test_iteration < test_iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                    data.get_batch('test', test_args_per_batch, args.test_shot, args.test_way,
                                   eval_samples_test)
                feedDict = {train_images: train_inputs, test_images: test_inputs,
                            train_labels: train_outputs, test_labels: test_outputs,
                            dropout_keep_prob: 1.0}
                iter_acc = sess.run([accuracy], feedDict)
                test_iteration_accuracy.append(iter_acc)
                test_iteration += 1
                test_end = time.time()
            test_accuracy = np.array(test_iteration_accuracy).mean() * 100.0
            confidence_interval_95 = \
                (196.0 * np.array(test_iteration_accuracy).std()) / np.sqrt(len(test_iteration_accuracy))
            print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                          .format(test_accuracy, confidence_interval_95, model_path))
            print_and_log(logfile, 'Test Time: {:5.3f}'.format(test_end-test_start))

        if args.mode == 'test':
            test_model(args.test_model_path)

    logfile.close()


if __name__ == "__main__":
    tf.app.run()
