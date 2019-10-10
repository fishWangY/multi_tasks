import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from scipy import misc

from net.inception import inception_v4
from net.inception_utils import inception_arg_scope
from preprocess.inception_preprocess import get_data, load_train_data, name_dict1, name_dict2


slim = tf.contrib.slim


class MultiParser(object):
    def __init__(self, config, sess):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.data_dir1 = config.data_dir1
        self.data_dir2 = config.data_dir2
        self.train_dir = config.train_dir
        self.pretrain_dir = config.pretrain_dir
        self.learning_rate = config.lr
        self.beta = config.beta1
        self.num_classes1 = config.num_classes1
        self.num_classes2 = config.num_classes2
        self.epoch = config.epoch
        self.epoch_step = config.epoch_step
        self.save_freq = config.save_freq
        self.test_dir1 = config.test_data_dir1
        self.test_dir2 = config.test_data_dir2

        self.sess = sess
        self._build_model()
        self.saver_backbone = tf.train.Saver(self.backbone_var)
        self.saver_all = tf.train.Saver(self.backbone_var + self.fc1_var + self.fc2_var)

    def _build_model(self):

        # define input and labels
        self.input1 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images1')
        self.label1 = tf.placeholder(tf.int64, [None, ], name='label1')
        self.input2 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images2')
        self.label2 = tf.placeholder(tf.int64, [None, ], name='label2')

        # inception V4 backbone
        with slim.arg_scope(inception_arg_scope(weight_decay=0.00004)):
            feature1, _ = inception_v4(self.input1, reuse=tf.AUTO_REUSE)
            feature2, _ = inception_v4(self.input2, reuse=tf.AUTO_REUSE)

        # head1
        logit1 = slim.fully_connected(feature1, self.num_classes1, activation_fn=None, scope='Logit1')
        self.predict1 = tf.argmax(tf.nn.softmax(logit1, axis=-1, name='predict1'), axis=1, name='index1')
        self.onehot1 = slim.one_hot_encoding(self.label1, self.num_classes1)

        # head2
        logit2 = slim.fully_connected(feature2, self.num_classes2, activation_fn=None, scope='Logit2')
        self.predict2 = tf.argmax(tf.nn.softmax(logit2, axis=-1, name='predict2'), axis=1, name='index2')
        self.onehot2 = slim.one_hot_encoding(self.label2, self.num_classes2)

        # cal softmax cross entropy loss
        self.loss1 = tf.losses.softmax_cross_entropy(self.onehot1, logit1)
        self.loss2 = tf.losses.softmax_cross_entropy(self.onehot2, logit2)

        self.summary1 = tf.summary.scalar("loss1", self.loss1)
        self.summary2 = tf.summary.scalar("loss2", self.loss2)
        self.all_sum = tf.summary.merge([self.summary1, self.summary2])

        self.all_vars = tf.all_variables()

        self.backbone_var = [var for var in self.all_vars if 'InceptionV4' in var.name]
        self.fc1_var = [var for var in self.all_vars if 'InceptionV4' not in var.name and '1' in var.name]
        self.fc2_var = [var for var in self.all_vars if 'InceptionV4' not in var.name and '2' in var.name]

    def train(self):
        """Train inception v4"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.optimizer1 = tf.train.AdamOptimizer(self.lr, beta1=self.beta) \
            .minimize(self.loss1, var_list=self.backbone_var + self.fc1_var)
        self.optimizer2 = tf.train.AdamOptimizer(self.lr, beta1=self.beta) \
            .minimize(self.loss2, var_list=self.backbone_var + self.fc2_var)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)

        # self.sess.run(tf.variables_initializer(var_list=self.add_var))
        # restore parameters from pretrain models
        self.saver_backbone.restore(self.sess, self.pretrain_dir)

        counter = 1
        start_time = time.time()

        # staring training
        for epoch in range(self.epoch):
            # loading dataset 1. And split to train/eval
            data1 = get_data(self.data_dir1)
            train_data1 = data1[int(0.1 * len(data1)):]
            eval_data1 = data1[: int(0.1 * len(data1))]
            test_data1 = get_data(self.test_dir1)

            # loading dataset 2. And split to train/eval
            data2 = get_data(self.data_dir2)
            train_data2 = data2[int(0.1 * len(data2)):]
            eval_data2 = data2[: int(0.1 * len(data2))]
            test_data2 = get_data(self.test_dir2)

            batch_idxs = min(len(train_data1) // self.batch_size, len(train_data2) // self.batch_size)
            lr = self.learning_rate if epoch < self.epoch_step else self.learning_rate*(self.epoch - epoch)/(self.epoch-self.epoch_step)

            for idx in range(0, batch_idxs):

                # loading dataset1 And dataset2 batch images and labels
                images1, labels1 = load_train_data(
                    data1[idx * self.batch_size:(idx + 1) * self.batch_size], name_dict1, self.image_size)

                images2, labels2 = load_train_data(
                    data2[idx * self.batch_size:(idx + 1) * self.batch_size], name_dict2, self.image_size
                )

                # Update training parameters
                summary_str, _, classify_loss1, _, classify_loss2 = self.sess.run(
                    [self.all_sum, self.optimizer1, self.loss1, self.optimizer2, self.loss2],
                    feed_dict={self.input1: images1, self.label1: labels1,
                               self.input2: images2, self.label2: labels2, self.lr: lr})

                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f classify_loss1 : %4.4f classify_loss2 : %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, classify_loss1, classify_loss2)))

                if np.mod(counter, self.save_freq) == 2:
                    self.save(self.train_dir, counter)

                    # starting eval when saved checkpoints
                    eval_accuracy1 = self.eval(self.sess, eval_data1, name_dict1,
                                                     self.input1, self.predict1)
                    eval_accuracy2 = self.eval(self.sess, eval_data2, name_dict2,
                                                        self.input2, self.predict2)
                    txt_path = os.path.join(self.train_dir, 'eval', 'eval.txt')
                    if not os.path.exists(os.path.dirname(txt_path)):
                        os.makedirs(os.path.dirname(txt_path))
                    with open(txt_path, 'a') as txt_file:
                        txt_file.write('steps: {}, eval accuracy 1 is {}, eval accuracy 2 is {}'.format(
                            str(counter), eval_accuracy1, eval_accuracy2) + '\n')

                    # starting test when saved checkpoints
                    test_accuracy1 = self.eval(self.sess, test_data1, name_dict1,
                                                     self.input1, self.predict1)
                    test_accuracy2 = self.eval(self.sess, test_data2, name_dict2,
                                                        self.input2, self.predict2)
                    txt_path = os.path.join(self.train_dir, 'test', 'test.txt')
                    if not os.path.exists(os.path.dirname(txt_path)):
                        os.makedirs(os.path.dirname(txt_path))
                    with open(txt_path, 'a') as txt_file:
                        txt_file.write('steps: {}, test accuracy 1 is {}, test accuracy 2 is {}'.format(
                            str(counter), test_accuracy1, test_accuracy2) + '\n')

    def save(self, checkpoint_dir, step):
        model_name = "inception_v4.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver_all.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_checkpoints(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        self.saver_all.restore(self.sess, checkpoint_dir)

    # test models
    def eval(self, sess, eval_data_list, map_dict, tensor_input, tensor_predict):

        # predict
        predict_true, batch_size = 0.0, self.batch_size
        n_samples = len(eval_data_list)
        n_batches = int(np.ceil(float(n_samples) / float(batch_size)))
        map_dict = {v: k for k, v in map_dict.items()}
        for i in tqdm(range(n_batches)):
            image_batch, image_list_batch = [], []
            for j in range(i * batch_size, min(n_samples, (i + 1) * batch_size)):
                image = misc.imread(eval_data_list[j])
                image = misc.imresize(image, (299, 299))
                image = image / 127.5 - 1
                image_batch.append(image.astype(np.float32))
                image_list_batch.append(eval_data_list[j])

            predict = sess.run(tensor_predict, feed_dict={tensor_input: image_batch})
            predicts = predict.tolist()

            for index in range(len(predicts)):
                ground_truth = image_list_batch[index].split('/')[-2]
                predict = map_dict[predicts[index]]

                if ground_truth == predict:
                    predict_true += 1

        return predict_true / n_samples
