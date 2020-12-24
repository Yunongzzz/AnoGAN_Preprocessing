import shutil
import tensorflow as tf
import os

from PIL import Image


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def log2n(n):
    if n > 1:
        exp_n = 1 + log2n(n / 2)
    else:
        exp_n = 0

    return exp_n


def str_to_bool():
    str_bool_dic = {'True': True,
                    'False': False}

    return str_bool_dic


def single_patch_store(patch_path, img, height, width):
    patch_name = os.path.basename(img)
    img_name = os.path.basename(os.path.dirname(img))

    im = Image.open(img)
    imgwidth, imgheight = im.size
    img_patch_index = 0
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            img_patch = im.crop(box)
            saved_path = os.path.join(patch_path, img_name + '_' + patch_name + '_' + str(img_patch_index) + ".jpg")
            img_patch.save(saved_path, "JPEG")
            img_patch_index += 1


def patch_extract_save(img_path, patch_path, ori_patch_size_val):
    height = ori_patch_size_val
    width = ori_patch_size_val

    for i in os.listdir(img_path):
        img = os.path.join(img_path, i)
        single_patch_store(patch_path, img, height, width)


def bach_class_reorganize(img_path, reorganize_img_path):
    class_0_path = os.path.join(reorganize_img_path, 'class_0')
    class_1_path = os.path.join(reorganize_img_path, 'class_1')

    if not os.path.exists(class_0_path):
        os.mkdir(class_0_path)
    if not os.path.exists(class_1_path):
        os.mkdir(class_1_path)

    benign_path = os.path.join(img_path, 'Benign')
    normal_path = os.path.join(img_path, 'Normal')
    for b in os.listdir(benign_path):
        shutil.copy(os.path.join(benign_path, b), class_0_path)
    for n in os.listdir(normal_path):
        shutil.copy(os.path.join(normal_path, n), class_0_path)

    insitu_path = os.path.join(img_path, 'InSitu')
    invasive_path = os.path.join(img_path, 'Invasive')

    for i in os.listdir(insitu_path):
        shutil.copy(os.path.join(insitu_path, i), class_1_path)
    for j in os.listdir(invasive_path):
        shutil.copy(os.path.join(invasive_path, j), class_1_path)
