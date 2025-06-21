from charset_normalizer.utils import iana_name

from metrics import *
import glob
import time
import timeit
import numpy as np
import cv2
import cv2 as cv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import nibabel as nib
import skimage.morphology as morphology
# This code aims to help you visualize a lived and tracked video and test how changing parameters impact the tracking
# Last update (28/03/2025): add a red point to follow the mouse's centroid in the live video




##################################################### Analyses #########################################################
framerate = 10
number_of_images = 100
path_to_segs = "../segs/"
images_paths = sorted(glob.glob(f"{path_to_segs}*.png"))[:number_of_images]

segs = []
for path in images_paths:
    image = read_image(path)
    segs.append(largest_connected_component((image>100) * 1.0))

x,y,t = read_csv_traj(f"{path_to_segs}test_csv")
aligned_segs = align_segmentations(segs)
mean_value, __ = evaluate_aligned_segmentations(aligned_segs, framerate)
print(f"Dice moyen selon la frame rate {framerate}: {mean_value}")



# for i in range(len(aligned_segs)):
#     save_image(aligned_segs[i], f"../segs_centered/centered_{i:03d}.png")



volume = np.stack(aligned_segs, axis=0) * 255.0  # shape: (N, H, W)
save_nifti_image(volume, "segmentation_3d.nii.gz")

# for i, image in enumerate(aligned_segs):
#     skelet = extract_mouse_skelet(image)
#     save_image(skelet, f"../skelets/skelet_{i:03d}.png")