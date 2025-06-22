from charset_normalizer.utils import iana_name

from metrics import *
import numpy as np
from utils import load_segmentations_from_video, remove_empty_segmentations

# This code aims to help you visualize a lived and tracked video and test how changing parameters impact the tracking
# Last update (28/03/2025): add a red point to follow the mouse's centroid in the live video




##################################################### Analyses #########################################################
framerate = 10
path_to_video = "../results/Video1_TO0/old_tracking_Video1_TO0.avi"


segs = load_segmentations_from_video(path_to_video)
segs_copy = [seg * 1.0 for seg in segs]

# Supprimer les images vides
segs_copy = remove_empty_segmentations(segs_copy)
aligned_segs = align_segmentations(segs_copy)
mean_value, __ = evaluate_aligned_segmentations(aligned_segs, framerate)
print(f"Dice moyen selon la frame rate {framerate}: {mean_value}")


