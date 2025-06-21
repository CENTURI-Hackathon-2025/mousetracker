
import numpy as np
import matplotlib.pyplot
import pandas as pd
from tqdm import tqdm
import glob
import time
import timeit
import numpy as np
import cv2
import matplotlib.pyplot as plt

def track(input_video):

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

    # Window Initialization
    # The video + tracking will be displayed in this window
    # cv2.namedWindow("Original (left) | Tracked (right)", cv2.WINDOW_NORMAL)

    # Resize the window to a specific size (e.g. 1024x512)
    # window_width = 1024  # Window width
    # window_height = 512  # Window heigh
    # cv2.resizeWindow(
    #     "Original (left) | Tracked (right)", window_width, window_height
    # )

    segs = []

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
        )  # Fusionner les deux vidéos

        gray = cv2.cvtColor(thresh_colored, cv2.COLOR_BGR2GRAY)
        segs.append(gray)

        # Adjust the size of the concatenated image to fill the window without distortion
        # combined_resized = cv2.resize(
        #     combined,
        #     (window_width, window_height),
        #     interpolation=cv2.INTER_AREA,
        # )

        # Show the combined video in one window
        # cv2.imshow("Original (left) | Tracked (right)", combined_resized)

        # Processing time
        frame_end = time.time()
        frame_processing_time = frame_end - frame_start

        # Calculating waiting time
                
        # wait_time = max(
        #     1, int((1000 / framerate) - (frame_processing_time * 1000))
        # )

        # Manage keyboard inputs
        # k = cv2.waitKey(wait_time) & 0xFF
        # if k == ord("q"):  # Quit
            # break
        # elif k == ord(" "):  # Break
            # pause = True

    # Close the window and free the memory
    cap.release()
    # cv2.destroyAllWindows()

    stop = timeit.default_timer()
    print("Time: ", stop - start)

    return x_pos, y_pos, time_arr, segs