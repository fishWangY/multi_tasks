import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
from scipy import misc
import tensorflow as tf

from config import Config
from net.MultiParser import MultiParser
from preprocess.inception_preprocess import name_dict1, name_dict2


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir1', type=str, default=None, help='where dataset 1. test data saved.')
    parser.add_argument('--data_dir2', type=str, default=None, help='where dataset 2. test data saved.')
    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='Path where .ckpt file is saved')
    parser.add_argument('--out_dir', type=str, default='./models/test/',
                        help='Path where test summary file is saved')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=str, default="1")
    parser.add_argument('--gpu_fraction', type=float, default=0.2)

    return parser.parse_args(argv)


# calculate prediction for samples
def cal_predict(sess, input, model_predict, data_dir, map_dict,  batch_size):

    image_list = []

    for label in os.listdir(data_dir):
        labels = os.path.join(data_dir, label)
        image_list += glob.glob(os.path.join(labels, '*.jpg'))

    # predict
    predict_true = 0.0
    n_samples = len(image_list)
    n_batches = int(np.ceil(float(n_samples) / float(batch_size)))
    for i in tqdm(range(n_batches)):
        image_batch, image_list_batch = [], []
        for j in range(i * batch_size, min(n_samples, (i + 1) * batch_size)):
            image = misc.imread(image_list[j])
            image = image / 255.0
            image_batch.append(image.astype(np.float32))
            image_list_batch.append(image_list[j])

        predict = sess.run(model_predict, feed_dict={input: image_batch})
        predicts = predict.tolist()

        for index in range(len(predicts)):
            ground_truth = image_list_batch[index].split('/')[-2]
            predict = map_dict[predicts[index]]

            if ground_truth == predict:
                predict_true += 1

    return predict_true / n_samples


# write test result to tf summary
class SummaryWriter:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.eval_steps = set()
        self._load_history()
        self.writer = None
        self.acc_summary = None

    def _write_history(self, step):
        with open(os.path.join(self.save_dir, 'eval_history.txt'), 'a') as txt_file:
            txt_file.write(str(step) + '\n')
            self.eval_steps.add(step)

    def _load_history(self):
        file_path = os.path.join(self.save_dir, 'eval_history.txt')
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r') as txt_file:
            for line in txt_file.readlines():
                if not line.strip('\n'):
                    continue
                self.eval_steps.add(int(line.strip('\n')))

    def build(self):
        with tf.name_scope('test'):
            self.writer = tf.summary.FileWriter(self.save_dir)

            # Scalar acc
            self.accuracy1 = tf.placeholder(dtype=tf.float32, shape=[])
            self.accuracy2 = tf.placeholder(dtype=tf.float32, shape=[])
            self.acc_summary1 = tf.summary.scalar('accuracy1', self.accuracy1)
            self.acc_summary2 = tf.summary.scalar('accuracy2', self.accuracy2)
            self.acc_summary = tf.summary.merge([self.acc_summary1, self.acc_summary2])

    def write(self, sess, acc_sub1, acc_sub2, step):
        summary = sess.run(self.acc_summary, feed_dict={self.accuracy1: acc_sub1, self.accuracy2: acc_sub2})
        self.writer.add_summary(summary, global_step=step)
        self._write_history(step)

    def get_latest_step(self):
        if not self.eval_steps:
            return -1
        return max(self.eval_steps)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    train_parameters = Config('./', './', './')

    summary_writer = SummaryWriter(args.out_dir)
    start_step = summary_writer.get_latest_step()

    # Get all checkpoint paths
    paths = glob.glob(os.path.join(args.model_dir, '*.meta'))
    paths.sort(key=lambda x: (len(x), x))

    with tf.Graph().as_default():

        summary_writer.build()
        if args.gpu_fraction is None:
            gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=args.gpus)
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
                                        visible_device_list=args.gpus)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)

        with tf.Session(config=config) as sess:

            cnnModel = MultiParser(train_parameters, sess)
            for model_meta_path in paths:
                # Skip checkpoint if it was evaluated before
                model_path = model_meta_path[:len(model_meta_path) - 5]
                step = int(model_path.split('-')[-1])
                if step <= start_step:
                    continue
                # loading checkpoints
                cnnModel.load_checkpoints(model_path)

                # predict dataset 1.
                dict_reverse1 = {v: k for k, v in name_dict1.items()}
                accuracy1 = cal_predict(sess, cnnModel.input1, cnnModel.predict1,
                                              args.data_dir1, dict_reverse1, args.batch_size)

                # predict clothDoor
                dict_reverse2 = {v: k for k, v in name_dict2.items()}
                accuracy2 = cal_predict(sess, cnnModel.input2, cnnModel.predict2,
                                                 args.data_di2, dict_reverse2, args.batch_size)
                summary_writer.write(sess, accuracy1, accuracy2, step)
                print 'dataset 1. accuracy is {}, dataset 1. accuracy is {}'.format(accuracy1, accuracy2)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))