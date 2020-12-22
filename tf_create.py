import tensorflow as tf
import os
import math

from PIL import Image

from util import log2n, _int64_feature, _bytes_feature


def patch_extract_to_tf(tf_path, input_img_path, height, width):
    img_level = str(log2n(height))

    img = Image.open(input_img_path)
    img_name = os.path.basename(input_img_path)
    img_name = img_name.split('.')[0]
    imgwidth, imgheight = img.size

    tf_writer = tf.io.TFRecordWriter(os.path.join(tf_path, img_name + '_L' + img_level + '.tfrecords'))

    patch_str = list()
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            image_patch = img.crop(box)
            image_patch_str = image_patch.tobytes()
            patch_str.append(image_patch_str)

    for k in range(len(patch_str)):
        img_str = patch_str[k]
        image_name = img_name + 'patch_' + str(k)
        image_format = 'jpeg'
        feature = {'height': _int64_feature(height),
                   'width': _int64_feature(width),
                   'depth': _int64_feature(3),
                   'image/format': _bytes_feature(image_format.encode('utf8')),
                   'image_name': _bytes_feature(image_name.encode('utf8')),
                   'image/encoded': _bytes_feature(img_str)}

        Example = tf.train.Example(features=tf.train.Features(feature=feature))
        Serialized = Example.SerializeToString()
        tf_writer.write(Serialized)
    tf_writer.close()


def patch_resize_to_tf(tf_path, input_img_path, height, width, resize_shape):
    img_level = str(log2n(resize_shape[0]))

    img = Image.open(input_img_path)
    img_name = os.path.basename(input_img_path)
    img_name = img_name.split('.')[0]
    imgwidth, imgheight = img.size

    tf_writer = tf.io.TFRecordWriter(os.path.join(tf_path, img_name + '_L' + img_level + '.tfrecords'))

    patch_resize_str = list()
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            image_patch = img.crop(box)
            resize_patch = image_patch.resize(resize_shape)
            resize_patch_str = resize_patch.tobytes()
            patch_resize_str.append(resize_patch_str)

    for k in range(len(patch_resize_str)):
        img_str = patch_resize_str[k]
        image_name = img_name + 'patch_' + str(k) + 'level_' + img_level
        image_format = 'jpeg'
        feature = {'height': _int64_feature(height),
                   'width': _int64_feature(width),
                   'depth': _int64_feature(3),
                   'image/format': _bytes_feature(image_format.encode('utf8')),
                   'image_name': _bytes_feature(image_name.encode('utf8')),
                   'image/encoded': _bytes_feature(img_str)}

        Example = tf.train.Example(features=tf.train.Features(feature=feature))
        Serialized = Example.SerializeToString()
        tf_writer.write(Serialized)
    tf_writer.close()

def tf_create(img_path, tf_path, ori_patch_size_val, final_patch_size_val):
    tf_patch_path = os.path.join(tf_path, 'level_' + str(log2n(ori_patch_size_val)))

    if not os.path.exists(tf_patch_path):
        os.mkdir(tf_patch_path)

    patch_height = ori_patch_size_val
    patch_width = ori_patch_size_val

    for i in os.listdir(img_path):
        patch_extract_to_tf(tf_patch_path, os.path.join(img_path, i), patch_height, patch_width)

    resize_steps = log2n(ori_patch_size_val) - log2n(final_patch_size_val)
    res_index = resize_steps + 2

    for step in range(resize_steps):
        res_index = res_index - 1
        resize_num = math.pow(2, res_index)
        resize_num = int(resize_num)
        resized_shape = (resize_num, resize_num)

        tf_level_path = os.path.join(tf_path, 'level_' + str(log2n(resize_num)))

        if not os.path.exists(tf_level_path):
            os.mkdir(tf_level_path)

        for j in os.listdir(img_path):
            patch_resize_to_tf(tf_level_path, os.path.join(img_path, j), 512, 512, resized_shape)