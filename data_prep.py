from data_split import data_split
from tf_create import tf_create
from util import str_to_bool, bach_class_reorganize, patch_extract_save


def anoGAN_bach_prep(is_reorganize, is_split, return_patches, is_tfrecord_create,
                     img_path, reorganize_img_path, patch_path, tf_path,
                     ori_patch_size_val, final_patch_size_val, c0_train_ratio, c1_train_ratio):

    str_bool_dic = str_to_bool()

    is_reorganize = str_bool_dic[is_reorganize]
    is_split = str_bool_dic[is_split]
    return_patches = str_bool_dic[return_patches]
    is_tfrecord_create = str_bool_dic[is_tfrecord_create]

    if is_reorganize:
        bach_class_reorganize(img_path, reorganize_img_path)

    if is_split:
        data_split(img_path, reorganize_img_path, c0_train_ratio, c1_train_ratio)

    if return_patches:
        patch_extract_save(img_path, patch_path, ori_patch_size_val)

    if is_tfrecord_create:
        tf_create(img_path, tf_path, ori_patch_size_val, final_patch_size_val)