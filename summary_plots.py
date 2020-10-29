# !/usr/local/biotools/python/3.4.3/bin/python3
__author__ = "..."
__email__ = "..."
__status__ = "Dev"

import openslide
import tensorflow as tf
import os
import argparse
import sys
import pwd
import time
import subprocess
import re
import shutil
import glob
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import math
import io
import re
import matplotlib
#from skimage.filters import threshold_otsu
#from skimage.color import rgb2lab,rgb2hed
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from dataset_utils import *
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.geometry import geo
from descartes.patch import PolygonPatch
import xml.etree.ElementTree as ET
from xml.dom import minidom

'''function to check if input files exists and valid'''

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img





def create_binary_mask_new(input_label_file,svs_file,patch_dir,patch_level):

    fn=os.path.basename(svs_file)
    OSobj = openslide.OpenSlide(svs_file)
    divisor = int(OSobj.level_dimensions[0][0]/3000)
     
    patch_sub_size_x=int(OSobj.level_dimensions[0][0]/divisor)
    patch_sub_size_y=int(OSobj.level_dimensions[0][1]/divisor)
    img = OSobj.get_thumbnail((patch_sub_size_x, patch_sub_size_y))
    img = img.convert('RGB')
    img.save(os.path.join(patch_dir , fn + "_1.png"), "png")
    np_img = np.array(img)

    patch_sub_size_y = np_img.shape[0]
    patch_sub_size_x = np_img.shape[1]
    f, ax = plt.subplots(frameon=False)
    f.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.set_xlim(0, patch_sub_size_x)
    ax.set_ylim(patch_sub_size_y, 0)
    ax.imshow(img)
    
    poly_included_0 = []
    poly_included_1 = []
    fobj=open(input_label_file)
    header = fobj.readline()
    for i in fobj:
        i = i.strip()
        arr1 = i.split("\t")
        pca1 = int(arr1[2])/255
        arr1[1]=arr1[1].replace(".png","")
        arr = arr1[1].split("_")
        x1 = int(arr[len(arr)-5])/divisor
        x2 = int(arr[len(arr)-4])/divisor
        y1 = int(arr[len(arr)-2])/divisor
        y2 = int(arr[len(arr)-1])/divisor

        poly_included_0.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))
        poly_included_1.append(pca1)
    fobj.close() 
    
    for j in range(0, len(poly_included_1)):
        #print(poly_included_1[j])
        #print(poly_included_0[j])
        #sys.exit(0)
        patch1 = PolygonPatch(poly_included_0[j], facecolor=[0, 0, 0], edgecolor="green", alpha=poly_included_1[j], linewidth=1,zorder=2)
        ax.add_patch(patch1)
    ax.set_axis_off()
    DPI = f.get_dpi()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    f.set_size_inches(patch_sub_size_x / DPI, patch_sub_size_y / DPI)
    f.savefig(os.path.join(patch_dir , fn + "_2.png"), pad_inches='tight')

    images = [Image.open(x) for x in [os.path.join(patch_dir , fn + "_1.png"), os.path.join(patch_dir , fn + "_2.png")]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save(os.path.join(patch_dir , fn + ".png"),"png")
    #os.remove(os.path.join(patch_dir , fn + "_1.png"))
    #os.remove(os.path.join(patch_dir , fn + "_2.png"))
    
def main():

    input_label_file="normalized_pca.txt"
    svs_file="183533.svs"
    patch_dir="..."
    patch_level=0
    '''creating binary mask to inspect areas with tissue and performance of threshold''' 
    create_binary_mask_new(input_label_file,svs_file,patch_dir,patch_level)



if __name__ == "__main__":
    main()
