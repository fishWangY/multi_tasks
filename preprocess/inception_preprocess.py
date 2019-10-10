import os
import glob
import numpy as np
import imgaug.augmenters as iaa
from scipy import misc, ndimage
import threading
import Queue
q = Queue.Queue()


name_dict1 = {'name0': 0, 'name1': 1, 'name2': 2, 'name3': 3, 'name4': 4, 'name5': 5}
name_dict2 = {'name0': 0, 'name1': 1, 'name2': 2, 'name3': 3, 'name4': 4, 'name5': 5}


def get_data(data_dir):
    file_names = []
    for label in os.listdir(data_dir):
        file_names += glob.glob(os.path.join(data_dir, label, '*.jpg'))

    np.random.shuffle(file_names)

    return file_names


def load_train_data(image_list, map_dict, img_size):

    images, labels = [], []
    for index in range(len(image_list)):
        thread = threading.Thread(target=process, name='thread' + str(index), args=(image_list[index], img_size))
        thread.start()
        images.append(q.get())

        labels.append(map_dict[image_list[index].split('/')[-2]])

    return np.array(images).astype(np.float32), np.array(labels).astype(np.int64)


# image auguments by imgaug :https://github.com/aleju/imgaug
def process(image_path, img_size):

    image = misc.imread(image_path)

    # random rotate
    angle = np.random.uniform(low=-30, high=30)
    image = ndimage.rotate(image, angle)

    # random flip left_right
    flip_flag = np.random.randint(low=0, high=2)
    if flip_flag == 1:
        image = np.flip(image, axis=1)

    # color distored
    some_times = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            some_times(iaa.Add((-10, 10), per_channel=0.5)),
            some_times(iaa.AddToHueAndSaturation((-20, 20)))
        ],
        random_order=True
    )
    image = seq.augment_image(image)
    # scale normalize(ie: resize image)
    image = misc.imresize(image, (img_size, img_size))
    image = image/127.5 - 1

    q.put(image)
