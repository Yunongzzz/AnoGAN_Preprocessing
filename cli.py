import argparse

from data_prep import anoGAN_bach_prep


def make_arg_parser():
    parser = argparse.ArgumentParser(description='proGANanomaly command line arguments description',
                                     epilog='epilog')

    parser.add_argument('-n', '--is_reorg',
                        type=str,
                        default='True',
                        required=True,
                        help='Whether or not reorganizing the original microscopy images into 2 classes')

    parser.add_argument('-l', '--is_split',
                        type=str,
                        default='True',
                        required=True,
                        help='Whether or not splitting data into training and testing sets')

    parser.add_argument('-u', '--is_patch_save',
                        type=str,
                        default='True',
                        required=True,
                        help='Whether or not saving extracted image patches with the highest resolution level')

    parser.add_argument('-e', '--is_tf_create',
                        type=str,
                        default='True',
                        required=True,
                        help='Whether or not creating tfrecords')

    parser.add_argument('-i', '--image_path',
                        type=str,
                        required=True,
                        help='Path to the input microscopy images')

    parser.add_argument('-g', '--reorg_img_path',
                        type=str,
                        required=True,
                        help='Path to the reorganized 2 classes input microscopy images')

    parser.add_argument('-t', '--tf_path',
                        type=str,
                        required=True,
                        help='Path to where the tfrecords are stored')

    parser.add_argument('-a', '--patch_path',
                        type=str,
                        required=True,
                        help='Path to where the extracted image patches with the highest resolution level')

    parser.add_argument('-p', '--patch_size_val',
                        type=int,
                        default=512,
                        required=True,
                        help='Original size of the image patches extracted')

    parser.add_argument('-f', '--least_level_val',
                        type=int,
                        default=4,
                        required=True,
                        help='The least image patches level required for the model training')

    parser.add_argument('-c', '--c0_train',
                        type=float,
                        default=1.0,
                        required=True,
                        help='Ratio to split class 0 samples into training dataset')

    parser.add_argument('-r', '--c1_train',
                        type=float,
                        default=0.0,
                        required=True,
                        help='Ratio to split class 1 samples into training dataset')

    return parser

def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    anoGAN_bach_prep(is_reorganize=args.is_reorg,
                     is_split=args.is_split,
                     return_patches=args.is_patch_save,
                     is_tfrecord_create=args.is_tf_create,
                     img_path=args.image_path,
                     reorganize_img_path=args.reorg_img_path,
                     patch_path=args.patch_path,
                     tf_path=args.tf_path,
                     ori_patch_size_val=args.patch_size_val,
                     final_patch_size_val=args.least_level_val,
                     c0_train_ratio=args.c0_train,
                     c1_train_ratio=args.c1_train)