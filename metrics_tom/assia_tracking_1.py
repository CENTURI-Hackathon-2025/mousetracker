# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:03:05 2025

@author: AKourgli
"""

import cv2 as cv
import numpy as np
from skimage import morphology
import time

def track_motion(video_name, output_video_name="output_binary_video.avi"):
    """
    Détecte et suit le mouvement dans une vidéo
    Crée une vidéo des frames binaires
    Retourne:
        output_video_name (nom du fichier vidéo créé)
        smoothed_trajectory (liste des centres lissés)
    """
    video = cv.VideoCapture(video_name)
    if not video.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return None, None
    
    # Création du fond avec médiane
    list_video = []
    NBack = 200 # Nombre de frames utilisés pour détecter le background
    for _ in range(NBack):
        ret, cur_frame = video.read()
        if ret:
            list_video.append(cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY))
    
    if not list_video:
        print("Erreur: Aucune frame lue pour le fond")
        video.release()
        return None, None
        
    background = np.uint8(np.median(list_video, axis=0))
    video.release()
    
    # Réouverture de la vidéo pour traitement principal
    video = cv.VideoCapture(video_name)
    if not video.isOpened():
        print("Erreur: Impossible de rouvrir la vidéo")
        return None, None
    
    # Obtenir les propriétés de la vidéo pour créer la vidéo de sortie
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)
    
    # Créer le writer pour la vidéo de sortie
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_video = cv.VideoWriter(output_video_name, fourcc, fps, (width, height), False)
    
    Tb = 15 # Taille du filtre Gaussien pour lisser les frames
    trajectory_length = 15 #la longueur de la fenêtre de lissage utilisée pour calculer la trajectoire lissée de la souris ente 5 et 30
    trajectory = []
    smoothed_trajectory = []
    radius_history = []

    # Création des fenêtres
    # cv.namedWindow("Difference Frame", cv.WINDOW_NORMAL)
    # cv.namedWindow("Threshold Frame", cv.WINDOW_NORMAL)
    # cv.namedWindow("Tracking with Smooth Trajectory", cv.WINDOW_NORMAL)

    ret, cur_frame = video.read()  
    if not ret:
        print("Erreur: Impossible de lire la première frame")
        video.release()
        out_video.release()
        return None, None
        
    gray_image = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)  
    First_frame = cv.GaussianBlur(background, (Tb, Tb), 0)  

    while True:  

        iteration_start_time = time.time()

        ret, cur_frame = video.read()
        if not ret:
            break
            
        gray_image = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)  
        gray_frame = cv.GaussianBlur(gray_image, (Tb, Tb), 0)  
     
        diff_frame = cv.absdiff(First_frame, gray_frame)
        thresh_frame = cv.threshold(diff_frame, 20, 255, cv.THRESH_BINARY)[1]  
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thresh_frame = cv.dilate(thresh_frame, kernel, iterations=3) # Iterations entre 2 (objet fin) et 8 (grossier)
        #thresh_frame = morphology.skeletonize(thresh_frame, method='lee')
        
        # Écrire la frame binaire dans la vidéo de sortie
        out_video.write(thresh_frame)
        
        cont, _ = cv.findContours(thresh_frame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        current_center = None
        current_radius = 0
        
        for cur in cont:  
            if cv.contourArea(cur) < 10: # plus petit contour à considérer
                continue
                
            ((x, y), radius) = cv.minEnclosingCircle(cur)
            center = (int(x), int(y))
            radius = int(radius)
            
            if radius > 5:  # Rayon minimum
                current_center = center
                current_radius = radius
        
        # Traitement du lissage
        if current_center:
            trajectory.append(current_center)
            radius_history.append(current_radius)
            
            if len(trajectory) > trajectory_length:
                trajectory.pop(0)
                radius_history.pop(0)
            
            avg_x = int(np.mean([p[0] for p in trajectory]))
            avg_y = int(np.mean([p[1] for p in trajectory]))
            avg_radius = int(np.mean(radius_history))
            
            smoothed_center = (avg_x, avg_y)
            smoothed_trajectory.append(smoothed_center)
            
            # Dessin du cercle lissé
            # cv.circle(cur_frame, smoothed_center, avg_radius, (0, 255, 0), 2)
            # Dessin du centre lissé
            # cv.circle(cur_frame, smoothed_center, 3, (0, 0, 255), -1)
        
        # Dessin de la trajectoire lissée
        if len(smoothed_trajectory) > 1:
            for i in range(1, len(smoothed_trajectory)):
                thickness = int(2 * i/len(smoothed_trajectory)) + 1
                cv.line(cur_frame, smoothed_trajectory[i-1], smoothed_trajectory[i], (255, 0, 0), thickness)
        
        # Affichage des résultats
        # cv.imshow("Difference Frame", diff_frame)
        # cv.imshow("Threshold Frame", thresh_frame)
        # cv.imshow("Tracking with Smooth Trajectory", cur_frame)
        
        # Contrôle de la vitesse et gestion de la fermeture
        # wait_key = cv.waitKey(30)
        # if wait_key == ord('q'):
        #     break

        iteration_end_time = time.time()
        
        processing_time = iteration_end_time-iteration_start_time

        if processing_time>1/fps:

            print(f"WARNING: Frame took too long to process: {processing_time} > 0.04")
    
    video.release()
    out_video.release()  # Fermer la vidéo de sortie
    # cv.destroyAllWindows()
    return output_video_name, smoothed_trajectory

# Exemple d'utilisation
if __name__ == "__main__":
    output_video, trajectory = track_motion("Video2_TO0.avi")
    if output_video is not None:
        print(f"Vidéo binaire créée : {output_video}")
        print(f"Longueur de la trajectoire : {len(trajectory)}")