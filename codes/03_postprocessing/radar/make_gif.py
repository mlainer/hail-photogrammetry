from PIL import Image, ImageDraw
import os
import glob
import numpy as np

images =[ ]
path = '/scratch/mlainer/hail/plot_mzc/'

all_files = glob.glob(path+'*.png')
all_files.sort()

for file in all_files:
    im = Image.open(file, mode='r')
    images.append(im)

images[0].save('mzc.gif',save_all=True, append_images=images[1:], optimize=False, duration=10, loop=0)
