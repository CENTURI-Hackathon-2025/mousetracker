from track_algorithms import *
import argparse
import os


# video_file = input("Enter video file path: ")
# result_directory = input("Enter result directory path: ")
# method = input("Enter the method: ")

video_file = "/home/scarneiro/Documents/hackathon/Hackathon_videos/to_0/Video1_TO0.avi"
result_directory = "../results"
method = "new_tracking"



file_to_save = f"{result_directory}/{video_file.split('/')[-1].split('.')[0]}"
print(os.listdir(result_directory))
print(file_to_save)
if video_file.split('/')[-1].split('.')[0] not in os.listdir(result_directory):
    os.mkdir(file_to_save)

movie_file = f"{file_to_save}/{method}_{video_file.split('/')[-1]}"
csv_file = f"{file_to_save}/{method}_{video_file.split('/')[-1].split('.')[0]}"
if  method == "old_tracking":
    old_tracking(video_file, movie_file, csv_file)
elif method == "new_tracking":
    new_tracking(video_file, movie_file, csv_file)
