import glob
import time
import timeit
import numpy as np
import cv2
import cv2 as cv
from tqdm import tqdm
import os
from utils import *
import skimage.morphology as morphology

def old_tracking(input_video, tracking_video, csv_file):
    if glob.glob(input_video):
        print("Video found")
    else:
        print("No video found, check input_video name")

    # Register the starting time of the function
    start = timeit.default_timer()

    # Open the video
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    # Obtenir les propriétés de la vidéo pour créer la vidéo de sortie
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    # Créer le writer pour la vidéo de sortie
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_video = cv.VideoWriter(tracking_video, fourcc, fps, (width, height), False)


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
        )  # Fusionner les deux vidéos

        gray = cv2.cvtColor(thresh_colored, cv2.COLOR_BGR2GRAY)
        # Écrire la frame binaire dans la vidéo de sortie
        out_video.write(gray)

        # Processing time
        frame_end = time.time()
        frame_processing_time = frame_end - frame_start

    # Close the window and free the memory
    cap.release()
    # cv2.destroyAllWindows()

    stop = timeit.default_timer()
    print("Time: ", stop - start)
    out_video.release()  # Fermer la vidéo de sortie
    cv.destroyAllWindows()
    save_traj_to_csv(x_pos, y_pos, time_arr, csv_file)


def new_tracking(video_name, output_video_name, csv_file):
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
        'background_frames': 200,  # Nombre de frames pour créer le fond
        'gaussian_kernel': 15,  # Taille du noyau gaussien (doit être impair)
        'dilate_iterations': 3,  # Itérations de dilatation (2-8)
        'min_contour_area': 20,  # Surface minimale du contour
        'min_radius': 3,  # Rayon minimal pour considérer une détection
        'trajectory_length': 15,  # Longueur de la fenêtre de lissage
        'threshold_value': 20,  # Seuil pour la binarisation
        'morph_kernel_size': 5,  # Taille du noyau morphologique (impair)
        'output_fps': 25,  # FPS de la vidéo de sortie
        'display_scale': 0.7,  # Échelle d'affichage des fenêtres
        'enable_skeletonize': False  # Activer/désactiver le squeletisation
    }

    # ==============================
    # INITIALISATION
    # ==============================
    start_time = time.time()
    print(f"Début du traitement de la vidéo: {video_name}")
    x_list = []
    y_list = []
    t_list = []
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
    background = cv.GaussianBlur(background, (config['gaussian_kernel'], config['gaussian_kernel']), 0)

    # Réinitialiser la vidéo
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    framerate = int(video.get(cv2.CAP_PROP_FPS))

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


    # ==============================
    # VARIABLES DE SUIVI
    # ==============================
    trajectory = []  # Positions brutes
    smoothed_trajectory = []  # Positions lissées
    radius_history = []  # Historique des rayons
    frame_counter = 0  # Compteur de frames


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


            x_list.append(avg_x)
            y_list.append(avg_y)
            t_list.append(frame_counter/framerate)

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
    cv.destroyAllWindows()

    # Calcul des performances
    total_time = time.time() - start_time
    avg_fps = frame_counter / total_time if total_time > 0 else 0

    print("\n" + "=" * 50)
    print(f"Traitement terminé!")
    print(f"Frames traitées: {frame_counter}/{total_frames}")
    print(f"Temps total: {total_time:.2f} secondes")
    print(f"FPS moyen: {avg_fps:.2f}")
    print(f"Vidéo binaire créée: {output_video_name}")
    print("=" * 50)
    save_traj_to_csv(np.array(x_list), np.array(y_list), np.array(t_list), csv_file)
