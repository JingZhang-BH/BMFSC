import tensorflow as tf
import argparse
from get_features import extract_features_3d_9
from inference import infer_classifier
from block import sample_normal, multinoulli_log_density, print_and_log, get_test_log_files, sample_theta
from test_dataset import DataSet as TestDataSet
import numpy as np


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.enable_eager_execution()

"""
parse_command_line: command line parser
"""

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["IP", "SV", "PC", "PU","HS","BO","KSC","CK"],
                        default="SV", help="Dataset to use")
    parser.add_argument("--part", type=int,
                        default=0, help="Part of Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--dim", type=int, default=144,
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
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=100000,
                        help="Number of training iterations.")
    # parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
    #                     help="Directory to save trained models.")
    parser.add_argument("--testlogdir", "-c", default='./testlog',
                        help="Directory to save testlog.")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default='./checkpoint/2022-06-14-09-28-58/best_validation',
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=200,
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

    logfile, lognumfile = get_test_log_files(args.dataset, args.testlogdir)

    print_and_log(logfile, "Options: %s\n" % args)

    # Load training and eval data

    test_data = TestDataSet(path="data/", seed=42)

    # set the feature extractor based on the dataset
    feature_extractor_fn = extract_features_3d_9

    # testing parameters
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               test_data.get_image_height(),
                                               test_data.get_image_width(),
                                               test_data.get_bands()],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              test_data.get_image_height(),
                                              test_data.get_image_width(),
                                              test_data.get_bands()],
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
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    # L = tf.constant(test_data.get_sample_num(), dtype=tf.float32, name="num_samples")
    L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    # Relevant computations for a single task
    def evaluate_task(inputs):
        train_inputs, train_outputs, test_inputs, test_outputs = inputs

        with tf.variable_scope('shared_features'):
            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.dim,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)

            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.dim,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)

        # Infer classification layer from q
        with tf.variable_scope('classifier'):
            classifier = infer_classifier(features_train, train_outputs, args.dim, args.way) # 字典


        # Local reparameterization trick
        # Compute parameters of q distribution over logits
        weight_mean, bias_mean = classifier['weight_mean'], classifier['bias_mean']

        weight_mean, bias_mean = sample_theta(weight_mean, bias_mean, K=5, alpha=1)

        weight_log_variance, bias_log_variance = classifier['weight_log_variance'], classifier['bias_log_variance']

        # print('classify:s2')

        # weight_log_variance, bias_log_variance = classifier['weight_log_variance'], classifier['bias_log_variance']

        # print(weight_mean.shape)  ##(192,16)
        # print(features_test.shape)   ##(1419,192)
        # print('classify:s2')
        logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean  #(1419,16)
        # print('logits_mean_test', logits_mean_test.shape)
        # print('classify:s3')
        logits_log_var_test = \
            tf.log(tf.matmul(features_test ** 2, tf.exp(weight_log_variance)) + tf.exp(bias_log_variance))
        # print('logits_log_var_test', logits_log_var_test.shape)

        # logits_sample_test = sample_normal(logits_mean_test, logits_log_var_test, test_data.get_sample_num())
        logits_sample_test = sample_normal(logits_mean_test, logits_log_var_test, args.samples)
        # print(logits_sample_test.shape)
        # tmp_logits_sample_test = tf.reshape(logits_sample_test, (10, 1419, 16))
        # print(tmp_logits_sample_test.shape)
        # #
        # logits_sample_test_print = tf.Print(tmp_logits_sample_test,
        #                                 [tmp_logits_sample_test, tmp_logits_sample_test.shape, 'logits_sample_test_print'],
        #                                 message='Debug message:', summarize=1000)
        # logits_sample_test = logits_sample_test_print

        #todo C:  确定此处不影响
        #test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [test_data.get_sample_num(), 1, 1])
        test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [args.samples, 1, 1])
        # print('classify 190   test_labels_tiled:', test_labels_tiled.shape)
        # # 't2' is a tensor of shape [2, 3, 5]
        # shape(expand_dims(t2, 0)) == > [1, 2, 3, 5]
        # shape(expand_dims(t2, 2)) == > [2, 3, 1, 5]
        # shape(expand_dims(t2, 3)) == > [2, 3, 5, 1]
        task_log_py = multinoulli_log_density(inputs=test_labels_tiled, logits=logits_sample_test)
        # print('task_log_py', task_log_py.shape)
        averaged_predictions = tf.reduce_logsumexp(logits_sample_test, axis=0) - tf.log(L)
        predicted_label = tf.argmax(averaged_predictions, axis=-1)
        # predicted_label_tmp = tf.reshape(predicted_label,(test_data.get_sample_num(),1))
        #
        # predicted_label_print = tf.Print(predicted_label,
        #                             [predicted_label, predicted_label.shape, 'predicted_label'],
        #                             message='Debug message:', summarize=10000)
        # predicted_label = predicted_label_print

        # inter_sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run()
        # print(inter_sess.run(predicted_label))

        # print('predicted_label:', predicted_label.shape)
        # tf.Print(predicted_label, [predicted_label], summarize=2000)
        task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                        predicted_label), tf.float32))
        task_score = tf.reduce_logsumexp(task_log_py, axis=0) - tf.log(L)
        task_loss = -tf.reduce_mean(task_score, axis=0)

        return [task_loss, task_accuracy]

    # tf mapping of batch to evaluation function
    # print('classify:205, train_images:', train_images.shape)
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(train_images, train_labels, test_images, test_labels),
                             dtype=[tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_accuracies = batch_output
    # print('batch_accuracies:', batch_accuracies.shape)
    # loss = tf.reduce_mean(batch_losses)
    accuracy = tf.reduce_mean(batch_accuracies)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        test_iteration_accuracy = []

        def test_model(model_path, load=True):
            if load:
                saver.restore(sess, save_path=model_path)
            batch_num, res = test_data.get_iter()
            samp_num = test_data.get_sample_num()
            test_accuracy_all_batch = []
            support_images, support_labels, query_images, query_labels, query_images_last, query_labels_last = \
                test_data.get_batch(args.test_shot, args.test_way)
            print("########  support_images, support_labels, query_images, query_labels, query_images_last, query_labels_last:",
                  support_images.shape,support_labels.shape, query_images.shape, query_labels.shape,
                  query_images_last.shape, query_labels_last.shape)
            score_all_batch = 0
            samper_num_per_batch = test_data.get_samper_num_per_batch()
            # print('samper_num_per_batch:', samper_num_per_batch)
            for i in range(batch_num + 1):
                if i == batch_num:
                    feedDict = {train_images: support_images, test_images: query_images_last,
                                     train_labels: support_labels, test_labels: query_labels_last,
                                     dropout_keep_prob: 1.0}
                    iter_acc = sess.run(accuracy, feedDict)
                    test_accuracy_all_batch.append(iter_acc)
                    score_all_batch = score_all_batch + iter_acc * res

                else:
                    feedDict = {train_images: support_images, test_images: query_images[i],
                                train_labels: support_labels, test_labels: query_labels[i],
                                dropout_keep_prob: 1.0}
                    iter_acc = sess.run(accuracy, feedDict)
                    test_accuracy_all_batch.append(iter_acc)
                    score_all_batch = score_all_batch + iter_acc * samper_num_per_batch

            test_accuracy = score_all_batch / samp_num * 100.0
            print('测试集样本准确率：', test_accuracy)
            test_iteration_accuracy.append(test_accuracy)

        for test_iter in range(10):
            test_model(args.test_model_path)

        confidence_interval_95 = \
            (196.0 * np.array(test_iteration_accuracy).std()) / np.sqrt(len(test_iteration_accuracy))
        mean_test_accuracy = np.array(test_iteration_accuracy).mean()
        print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                      .format(mean_test_accuracy, confidence_interval_95, args.test_model_path))
        print_and_log(lognumfile, '{0:5.3f}'.format(mean_test_accuracy))

    logfile.close()


if __name__ == "__main__":
    tf.app.run()