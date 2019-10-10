import sys
import argparse
import tensorflow as tf

from config import Config
from net.MultiParser import MultiParser


def parser_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir1', type=str, default=None, help='Task 1. Dataset directory')
    parser.add_argument('--data_dir2', type=str, default=None, help='Task 2. Dataset directory')
    parser.add_argument('--train_dir', type=str, default=None, help='Where training dir.')
    parser.add_argument('--gpus', type=str, default="2")
    parser.add_argument('--gpu_fraction', type=float, default=1.0)

    return parser.parse_args(argv)


def main(args):
    if not tf.gfile.Exists(args.train_dir):
        tf.gfile.MakeDirs(args.train_dir)

    # config training process
    train_config = Config(args.data_dir1, args.data_dir2, args.train_dir)

    # initial CNN Model And training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction, visible_device_list=args.gpus)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        model = MultiParser(train_config, sess)
        model.train()


if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))