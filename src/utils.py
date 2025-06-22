import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2


def remove_empty_segmentations(list_segs):
    """Supprime les segmentations totalement vides."""
    return [seg for seg in list_segs if np.sum(seg) > 0]


def load_segmentations_from_video(video_path, threshold=100, frame_step=1):
    """
    Charge une vidéo et renvoie une liste d'images binaires (segmentations),
    en ne prenant qu'une image toutes les `frame_step` frames.

    Args:
        video_path (str): Chemin vers la vidéo.
        threshold (int): Seuil de binarisation.
        frame_step (int): Intervalle entre les frames à conserver.

    Returns:
        list of np.ndarray: Liste des segmentations binaires.
    """
    cap = cv2.VideoCapture(video_path)
    segs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Conversion en niveau de gris si besoin
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Binarisation
            _, binary = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)
            segs.append(binary.astype(bool))

        frame_idx += 1

    cap.release()
    return segs


def read_csv_traj(path_and_name):
    df = pd.read_csv(f"{path_and_name}.csv")
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    t = df.t.to_numpy()

    return x,y,t


def save_traj_to_csv(x, y, t, path_and_name='test_csv'):
    df = pd.DataFrame({'x': x, 'y': y, 't': t})

    df.to_csv(f"{path_and_name}.csv")


def read_image(path):
    image = Image.open(path)
    return np.array(image)


def save_image(image, output_path):
    image_pil = Image.fromarray(image)
    image_pil.save(output_path)

