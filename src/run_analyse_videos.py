from metrics import *
from utils import *
import numpy as np

video_to_analyse = input("Please enter the segmentation video path to analyse: ")
csv_to_analyse = input("Please enter the associated csv path to analyse: ")


### Import CSV file ###

x,y,t = read_csv_traj(csv_to_analyse)

### Compute the fraction of abherent speeds ###

v_thr = 200 # cm/s

abherent_speed_ratio = count_speeds_above_threshold(x,y,t,v_thr)

print(f"{abherent_speed_ratio*100} % of the speeds are above {v_thr} cm/s")

### Compute the fraction of abherent turning angles ###

angle_thr = np.pi/2 # radians

abherent_angle_ratio = count_bad_angles(x,y,t,angle_thr)

print(f"{abherent_angle_ratio*100} % of the speeds are above {angle_thr} rad")


