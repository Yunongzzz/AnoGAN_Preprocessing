import os
import random
import shutil

from util import bach_class_reorganize


def data_split(img_path, reorganize_img_path, c0_train_ratio, c1_train_ratio):
    bach_class_reorganize(img_path=img_path, reorganize_img_path=reorganize_img_path)

    train_path = os.path.join(reorganize_img_path, 'train')
    test_path = os.path.join(reorganize_img_path, 'test')

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    class_0_path = os.path.join(reorganize_img_path, 'class_0')
    class_1_path = os.path.join(reorganize_img_path, 'class_1')

    class_0_data_names = os.listdir(class_0_path)
    class_1_data_names = os.listdir(class_1_path)

    num_class_0_data = len(class_0_data_names)
    num_class_1_data = len(class_1_data_names)
    num_train_class_0 = int(num_class_0_data * c0_train_ratio)
    num_train_class_1 = int(num_class_1_data * c1_train_ratio)

    train_class_0_names = random.sample(class_0_data_names, num_train_class_0)
    test_class_0_names = list(set(class_0_data_names) - set(train_class_0_names))

    train_class_1_names = random.sample(class_1_data_names, num_train_class_1)
    test_class_1_names = list(set(class_1_data_names) - set(train_class_1_names))

    for i in train_class_0_names:
        shutil.move(os.path.join(class_0_path, i), train_path)
    for j in train_class_1_names:
        shutil.move(os.path.join(class_1_path, j), train_path)

    for m in test_class_0_names:
        shutil.move(os.path.join(class_0_path, m), test_path)
    for n in test_class_1_names:
        shutil.move(os.path.join(class_1_path, n), test_path)

    os.rmdir(class_0_path)
    os.rmdir(class_1_path)