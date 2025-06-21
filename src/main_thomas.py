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

def save_nifti_image(image, output):
    xform = np.eye(4)
    nib_image = nib.nifti1.Nifti1Image(image, xform)
    nib.save(nib_image, output)

def read_csv_traj(path_and_name):
    df = pd.read_csv(f"{path_and_name}.csv")
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    t = df.t.to_numpy()

    return x,y,t


def read_image(path):
    image = Image.open(path)
    return np.array(image)


def save_image(image, output_path):
    image_pil = Image.fromarray(image)
    image_pil.save(output_path)


def save_traj_to_csv(x, y, t, path_and_name='test_csv'):
    df = pd.DataFrame({'x': x, 'y': y, 't': t})

    df.to_csv(f"{path_and_name}.csv")


def track(input_video):
    segs = []
    if glob.glob(input_video):
        print("Video found")
    else:
        print("No video found, check input_video name")

    # Register the starting time of the function
    start = timeit.default_timer()

    # Open the video
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Creation of the background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=1, varThreshold=16, detectShadows=True
    )

    # Initialize position and time
    time_arr, x_pos, y_pos = np.array([]), np.array([]), np.array([])

    t = 0

    # Retrieving framerate and frame count
    resolution = 512, 512
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - framerate

    # Variable for break
    pause = False

    # Frame processing loop
    for _ in tqdm(range(0, length)):

        # If the video is paused, we wait
        while pause:
            k = cv2.waitKey(50) & 0xFF  # Waiting to avoid infinite loop
            if k == ord(" "):  # Press "Space" to resume
                pause = False

        # Register the time at which the current frame starts to be processed
        frame_start = time.time()

        # Read current frame
        ret, frm = cap.read()
        if not ret:
            break

        ##################################
        ### Mouse detection + tracking ###
        ##################################

        # Resize current frame
        frm = cv2.resize(frm, resolution, interpolation=cv2.INTER_AREA)

        # Apply a Gaussian blur (not sure if it is necessary ? Maybe it's better for the background substraction)
        kernelSize = (25, 25)
        frameBlur = cv2.GaussianBlur(frm, kernelSize, 0)

        # Apply background subtraction
        thresh = fgbg.apply(frameBlur, learningRate=0.0009)

        # Calculation of the centroid (center of mass). This will be considered as the position of the mouse.
        M = cv2.moments(thresh)
        if M["m00"] == 0:
            continue

        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        # Draw a red dot on the centroid
        cv2.circle(
            frm, (x, y), 10, (0, 0, 255), -1
        )  # Center (x, y), radius 10, color red (0, 0, 255), fill (-1)

        # Save positions and timestamps
        t += 1 / framerate
        time_arr = np.append(time_arr, t)
        x_pos = np.append(x_pos, x)
        y_pos = np.append(y_pos, y)

        #######################################
        ### Display of the video + tracking ###
        #######################################

        # Concatenate videos: Original on the left, tracked on the right
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat(
            [frm, thresh_colored]
        )  # Fusionner les deux vid�os

        gray = cv2.cvtColor(thresh_colored, cv2.COLOR_BGR2GRAY)
        segs.append(gray)
        # Processing time
        frame_end = time.time()
        frame_processing_time = frame_end - frame_start

    # Close the window and free the memory
    cap.release()
    # cv2.destroyAllWindows()

    stop = timeit.default_timer()
    print("Time: ", stop - start)

    return x_pos, y_pos, time_arr, segs


        ############################## tracking et sauvegarde de la souris ####################################################
x_pos, y_pos, time_arr, segs = track("/home/scarneiro/Documents/hackathon/Hackathon_videos/to_0/Video1_TO0.avi")
i_max = 400
for i in range(len(segs)):
    save_image(segs[i], f"../segs/seg_{i:03d}.png")
    if i ==i_max:
        break

save_traj_to_csv(x_pos[:i_max], y_pos[:i_max], time_arr[:i_max], path_and_name='../segs/test_csv')

print(len(segs))

