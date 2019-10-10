# config training process


class Config:
    def __init__(self, data_dir1, data_dir2, train_dir):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.train_dir = train_dir
        self.batch_size = 16
        self.image_size = 299
        self.backbone = 'inception_v4'
        self.pretrain_dir = './models/pretrain/inception_v4.ckpt'
        self.lr = 0.0001
        self.beta1 = 0.5
        self.epoch = 80
        self.epoch_step = 10
        self.save_freq = 500
        self.test_data_dir1 = './'
        self.test_data_dir2 = './'
        self.num_classes1 = 10
        self.num_classes2 = 10
