import tensorflow as tf
import numpy as np
import math
import os

from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


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


def patch_extract(img_path, patch_path, tf_path, ori_win_shape=(512, 512)):
    writer = tf.io.TFRecordWriter(os.path.join(tf_path, samp + '.tfrecords'))
    for i in os.listdir(img_path):
        images = Image.open(os.path.join(img_path, i))
        patch_height = images.shape[0]
        patch_width = images.shape[1]
        patch_depth = images.shape[2]
        image_format = 'png'
        image_name = os.path.basename(i)
        img_array = np.array(images)
        samp = extract_patches_2d(img_array, ori_win_shape)  ## shape be (512,512)

        '''writing tfrecord'''
        feature = {'height': _int64_feature(patch_height),
                   'width': _int64_feature(patch_width),
                   'depth': _int64_feature(patch_depth),
                   'image/format': _bytes_feature(image_format[0].encode('utf8')),
                   'image_name': _bytes_feature(image_name.encode('utf8')),
                   'image/encoded': _bytes_feature(samp)}

        Example = tf.train.Example(features=tf.train.Features(feature=feature))
        Serialized = Example.SerializeToString()
        writer.write(Serialized)
        # for j in range(len(re_imgs)):
        #     img_patch = Image.fromarray(re_imgs[j])
        #     img_patch.save(os.path.join(patch_path, img_name, str(j) + '_' + str(ori_win_shape[0]) + '.png'))


def img_resize(resize_patch_path, resize_shape=(256, 256)):
    resize_patch_path = os.path.join(resize_patch_path, 'resize_' + str(resize_shape[0]))
    if not os.path.exists(resize_patch_path):
        os.mkdir(resize_patch_path)

    for i in os.listdir(resize_patch_path):
        img_patch = Image.open(os.path.join(resize_patch_path, i))
        resize_patch = img_patch.resize(resize_shape)

        resize_patch.save(os.path.join(resize_patch_path, i + '_' + str(resize_shape[0]) + '.png'))


def pGAN_prep(resize_patch_path, ori_win_shape=(512, 512), resize_f_shape=(4, 4)):
    resize_steps = log2n(ori_win_shape[0]) - log2n(resize_f_shape[0])
    for i in range(resize_steps - 1):
        resolution_index = resize_steps
        resize_shape_num = math.pow(2, resolution_index)
        img_resize(resize_patch_path=resize_patch_path,
                   resize_shape=(resize_shape_num, resize_shape_num))
        resolution_index -= 1


def pGAN_tfrecord_create(tf_path, pGAN_img_path):
    image_string = list()
    image_name = pGAN_img_path
    image_patch_size = list()
    image_format = list()

    for file in pGAN_img_path:
        patch_size = file.shape
        img_str = open(file, 'rb').read()
        img_format = file.format

        image_patch_size.append(patch_size)
        image_string.append(img_str)
        image_format.append(img_format)

    patch_height = image_patch_size[0][0]
    patch_width = image_patch_size[0][1]
    patch_depth = image_patch_size[0][2]

    res_num = log2n(patch_height)
    writer = tf.io.TFRecordWriter(os.path.join(tf_path, 'resolution_index_' + str(res_num) + '.tfrecords'))
    '''writing tfrecord'''
    feature = {'height': _int64_feature(patch_height),
               'width': _int64_feature(patch_width),
               'depth': _int64_feature(patch_depth),
               'image/format': _bytes_feature(image_format[0].encode('utf8')),
               'image_name': _bytes_feature(image_name.encode('utf8')),
               'image/encoded': _bytes_feature(image_string)}

    Example = tf.train.Example(features=tf.train.Features(feature=feature))
    Serialized = Example.SerializeToString()
    writer.write(Serialized)


img_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/sam'
patch_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Patches'
resize_patch_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Resize/Benign'
tf_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/tfrecord/Benign'
patch_extract(img_dir, patch_dir, ori_win_shape=(512, 512))
#pGAN_prep(patch_dir, ori_win_shape=(512, 512), resize_f_shape=(4, 4))
#pGAN_tfrecord_create(tf_dir, patch_dir)
