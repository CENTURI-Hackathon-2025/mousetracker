# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:03:05 2025

@author: AKourgli
"""

import cv2 as cv
import numpy as np
import os
import time
from skimage import morphology

def track_motion(video_name, output_video_name="output_binary_video.avi", trajectory_file="trajectory.txt"):
    """
    Détecte et suit le mouvement dans une vidéo avec des améliorations significatives
    Crée une vidéo des frames binaires et sauvegarde la trajectoire
    
    Args:
        video_name (str): Chemin vers la vidéo d'entrée
        output_video_name (str): Nom du fichier vidéo de sortie
        trajectory_file (str): Fichier pour sauvegarder la trajectoire
        
    Returns:
        tuple: (output_video_name, smoothed_trajectory)
    """
    # ==============================
    # PARAMÈTRES CONFIGURABLES
    # ==============================
    config = {
        'background_frames': 200,   # Nombre de frames pour créer le fond
        'gaussian_kernel': 15,      # Taille du noyau gaussien (doit être impair)
        'dilate_iterations': 3,     # Itérations de dilatation (2-8)
        'min_contour_area': 20,     # Surface minimale du contour
        'min_radius': 3,            # Rayon minimal pour considérer une détection
        'trajectory_length': 15,    # Longueur de la fenêtre de lissage
        'threshold_value': 20,      # Seuil pour la binarisation
        'morph_kernel_size': 5,     # Taille du noyau morphologique (impair)
        'output_fps': 25,           # FPS de la vidéo de sortie
        'display_scale': 0.7,       # Échelle d'affichage des fenêtres
        'enable_skeletonize': False # Activer/désactiver le squeletisation
    }
    
    # ==============================
    # INITIALISATION
    # ==============================
    start_time = time.time()
    print(f"Début du traitement de la vidéo: {video_name}")
    
    # Ouvrir la vidéo
    video = cv.VideoCapture(video_name)
    if not video.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return None, None
    
    # Vérifier et créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_video_name), exist_ok=True)
    
    # ==============================
    # CRÉATION DU FOND DYNAMIQUE
    # ==============================
    print("Création du fond d'arrière-plan...")
    background_frames = []
    frame_count = 0
    
    while frame_count < config['background_frames']:
        ret, frame = video.read()
        if not ret:
            break
            
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        background_frames.append(gray_frame)
        frame_count += 1
    
    if not background_frames:
        print("Erreur: Aucune frame lue pour le fond")
        video.release()
        return None, None
    
    # Création du fond avec médiane et flou gaussien
    background = np.uint8(np.median(background_frames, axis=0))
    background = cv.GaussianBlur(background, (config['gaussian_kernel'], config['gaussian_kernel']),0)
    
    # Réinitialiser la vidéo
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    
    # ==============================
    # PRÉPARATION SORTIE
    # ==============================
    # Obtenir les propriétés de la vidéo
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS) or config['output_fps']
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"Résolution: {width}x{height}, FPS: {fps:.1f}, Total frames: {total_frames}")
    
    # Créer le writer pour la vidéo de sortie
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_video = cv.VideoWriter(output_video_name, fourcc, fps, (width, height), False)
    
    # Préparer le fichier de trajectoire
    trajectory_writer = open(trajectory_file, 'w') if trajectory_file else None
    
    # ==============================
    # VARIABLES DE SUIVI
    # ==============================
    trajectory = []          # Positions brutes
    smoothed_trajectory = [] # Positions lissées
    radius_history = []      # Historique des rayons
    frame_counter = 0        # Compteur de frames
    
    # ==============================
    # FENÊTRES D'AFFICHAGE
    # ==============================
    # cv.namedWindow("Difference", cv.WINDOW_NORMAL)
    # cv.namedWindow("Threshold", cv.WINDOW_NORMAL)
    # cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
    
    # # Redimensionner les fenêtres
    # display_width = int(width * config['display_scale'])
    # display_height = int(height * config['display_scale'])
    # cv.resizeWindow("Difference", display_width, display_height)
    # cv.resizeWindow("Threshold", display_width, display_height)
    # cv.resizeWindow("Tracking", display_width, display_height)
    
    # ==============================
    # NOYAUX MORPHOLOGIQUES
    # ==============================
    kernel_size = config['morph_kernel_size']
    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size * 2, kernel_size * 2))
    
    # ==============================
    # BOUCLE PRINCIPALE
    # ==============================
    print("Début du traitement des frames...")
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        frame_counter += 1
        if frame_counter % 50 == 0:
            print(f"Traitement frame {frame_counter}/{total_frames}")
        
        # Conversion en niveaux de gris et flou gaussien
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.GaussianBlur(gray_frame, (config['gaussian_kernel'], config['gaussian_kernel']), 0)
        
        # Soustraction de fond et seuillage
        diff_frame = cv.absdiff(background, gray_frame)
        _, thresh_frame = cv.threshold(diff_frame, config['threshold_value'], 255, cv.THRESH_BINARY)
        
        # Opérations morphologiques améliorées
        thresh_frame = cv.morphologyEx(thresh_frame, cv.MORPH_CLOSE, close_kernel)
        thresh_frame = cv.dilate(thresh_frame, dilate_kernel, iterations=config['dilate_iterations'])
        
        # Sauvegarde de la frame binaire
        out_video.write(thresh_frame)
        
        # Option: squeletisation
        if config['enable_skeletonize']:
            thresh_frame = morphology.skeletonize(thresh_frame, method='lee')
            thresh_frame = (thresh_frame * 255).astype(np.uint8)
        
       
        # Détection des contours
        contours, _ = cv.findContours(thresh_frame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Liste des centres détectés (pour multi-objet)
        current_centers = []
        
        # Traitement des contours
        for contour in contours:
            # Filtre par aire
            area = cv.contourArea(contour)
            if area < config['min_contour_area']:
                continue
                
            # Filtre par circularité
            perimeter = cv.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # if circularity < 0.1:  # Éliminer les formes très allongées
            #     continue
                
            # Cercle englobant
            ((x, y), radius) = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            if radius >= config['min_radius']:
                current_centers.append((center, radius))
        
        # Suivi de l'objet principal (le plus gros)
        main_center = None
        main_radius = 0
        
        if current_centers:
            # Trouver l'objet avec le plus grand rayon
            main_center, main_radius = max(current_centers, key=lambda x: x[1])
            
            # Mise à jour de la trajectoire
            trajectory.append(main_center)
            radius_history.append(main_radius)
            
            # Limite de l'historique
            if len(trajectory) > config['trajectory_length']:
                trajectory.pop(0)
                radius_history.pop(0)
            
            # Calcul de la moyenne mobile
            avg_x = int(np.mean([p[0] for p in trajectory]))
            avg_y = int(np.mean([p[1] for p in trajectory]))
            avg_radius = int(np.mean(radius_history))
            
            smoothed_center = (avg_x, avg_y)
            smoothed_trajectory.append(smoothed_center)
            
            # Sauvegarde de la position
            if trajectory_writer:
                trajectory_writer.write(f"{frame_counter},{avg_x},{avg_y},{avg_radius}\n")
            
            # Dessin du cercle et du centre
            # cv.circle(frame, smoothed_center, avg_radius, (0, 255, 0), 2)
            # cv.circle(frame, smoothed_center, 3, (0, 0, 255), -1)
        
        # Dessin de la trajectoire
        # if len(smoothed_trajectory) > 1:
        #     for i in range(1, len(smoothed_trajectory)):
        #         # Épaisseur proportionnelle à la récence
        #         thickness = int(2 * i/len(smoothed_trajectory)) + 1
        #         cv.line(frame, smoothed_trajectory[i-1], smoothed_trajectory[i], (255, 0, 0), thickness)
        
        # Affichage des FPS en temps réel
        # elapsed_time = time.time() - start_time
        # current_fps = frame_counter / elapsed_time if elapsed_time > 0 else 0
        # fps_text = f"FPS: {current_fps:.1f} | Objets: {len(current_centers)}"
        #cv.putText(frame, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Affichage des résultats
        # cv.imshow("Difference", diff_frame)
        # cv.imshow("Threshold", thresh_frame)
        
        # Redimensionner l'image de suivi pour l'affichage
        # display_frame = cv.resize(frame, (display_width, display_height))
        # cv.imshow("Tracking", display_frame)
        
        # Contrôle
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv.waitKey(-1)  # Pause
    
    # ==============================
    # FIN DE TRAITEMENT
    # ==============================
    # Libération des ressources
    video.release()
    out_video.release()
    if trajectory_writer:
        trajectory_writer.close()
    cv.destroyAllWindows()
    
    # Calcul des performances
    total_time = time.time() - start_time
    avg_fps = frame_counter / total_time if total_time > 0 else 0
    
    print("\n" + "="*50)
    print(f"Traitement terminé!")
    print(f"Frames traitées: {frame_counter}/{total_frames}")
    print(f"Temps total: {total_time:.2f} secondes")
    print(f"FPS moyen: {avg_fps:.2f}")
    print(f"Vidéo binaire créée: {output_video_name}")
    print(f"Trajectoire sauvegardée: {trajectory_file}")
    print("="*50)
    
    return output_video_name, smoothed_trajectory

# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres d'entrée
    input_video = "Video2_TO0.avi"
    output_video = "results/binary_output.avi"
    trajectory_file = "results/trajectory_data.csv"
    
    # Exécution du tracking
    output, trajectory = track_motion(input_video, output_video, trajectory_file)
