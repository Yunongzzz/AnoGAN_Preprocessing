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


def crop(path, input, height, width):
    fn = os.path.basename(input)
    fbn = os.path.basename(os.path.dirname(input))

    im = Image.open(input)
    imgwidth, imgheight = im.size
    k = 0
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            a = im.crop(box)
            a.save(os.path.join(path, fbn + '_' + fn + '_' + str(k) + ".jpg"), "JPEG")
            k += 1