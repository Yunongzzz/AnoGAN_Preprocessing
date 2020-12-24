from PIL import Image
import os
import shutil
import numpy as np

data_dir = '/Users/29124/Desktop/SU/Project/'


def data_reorg(img_dir):
    pos_dir = os.path.join(img_dir, 'class_1')
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)

    neg_dir = os.path.join(img_dir, 'class_0')
    if not os.path.exists(neg_dir):
        os.mkdir(neg_dir)

    benign_dir = os.path.join(img_dir, 'Benign')
    normal_dir = os.path.join(img_dir, 'Normal')
    insitu_dir = os.path.join(img_dir, 'InSitu')
    invasive_dir = os.path.join(img_dir, 'Invasive')

    for i in os.listdir(benign_dir):
        file_name = os.path.join(benign_dir, i)
        shutil.move(file_name, neg_dir)

    for j in os.listdir(normal_dir):
        file_name = os.path.join(normal_dir, j)
        shutil.move(file_name, neg_dir)

    for m in os.listdir(insitu_dir):
        file_name = os.path.join(insitu_dir, m)
        shutil.move(file_name, pos_dir)

    for n in os.listdir(invasive_dir):
        file_name = os.path.join(invasive_dir, n)
        shutil.move(file_name, pos_dir)


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
            saved_path = os.path.join(patch_path, img_name + '_' +
                                      patch_name + '_' + str(img_patch_index) + ".jpg")
            img_patch.save(saved_path, "JPEG")
            img_patch_index += 1

def patch(image_dir, height, width):
    neg_dir = os.path.join(image_dir, 'class_0')
    pos_dir = os.path.join(image_dir, 'class_1')

    patch_dir= os.path.join(image_dir, 'patch_dir')
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)

    for i in os.listdir(neg_dir):
        file_name = os.path.join(neg_dir, i)
        single_patch_store(patch_dir, file_name, height, width)
    for j in os.listdir(pos_dir):
        file_name = os.path.join(pos_dir,j)
        single_patch_store(patch_dir,file_name,height,width)


patch(data_dir, 512, 512)