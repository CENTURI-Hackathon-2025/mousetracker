import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import rotate, center_of_mass
from skimage.morphology import skeletonize
from skimage.transform import resize
import matplotlib.pyplot as plt
# def b0_error_numpy_2D(x, y):
#
#     # _, ncc_pred = label(, return_num=True, connectivity=2)
#     # return ncc_pred
#
# def b1_error_numpy_2D(y_pred):
#     y_pred_inverse = np.ones(y_pred.shape) - y_pred
#     _, ncc_pred = label(y_pred_inverse, return_num=True, connectivity=1)
#     b1_pred = ncc_pred - 1
#
#     return b1_pred

def dice(seg1, seg2):
    intersection = np.logical_and(seg1, seg2).sum()
    return 2 * intersection / (seg1.sum() + seg2.sum())

def extract_orientation(seg):
    """Retourne l'orientation principale (en degrés) de l'objet segmenté."""
    props = regionprops(label(seg.astype(int)))
    if not props:
        return 0  # ou np.nan si aucune région
    centroid = props[0].centroid
    return -props[0].orientation * 180 / np.pi, centroid  # skimage donne en radians

def reorient_seg(seg, angle):
    return rotate(seg, angle=angle, reshape=False, order=0)

def recenter_seg(seg, output_shape):
    """Centre la segmentation dans une image de taille `output_shape`."""
    com = center_of_mass(seg)
    target_center = np.array(output_shape) / 2
    shift = target_center - com

    # Création d'une image vide centrée
    centered = np.zeros(output_shape, dtype=bool)

    # Coordonnées originales
    coords = np.argwhere(seg)
    shifted_coords = (coords + shift).astype(int)

    # Évite les dépassements
    for y, x in shifted_coords:
        if 0 <= y < output_shape[0] and 0 <= x < output_shape[1]:
            centered[y, x] = True

    return centered


def dice(seg1, seg2):
    intersection = np.logical_and(seg1, seg2).sum()
    return 2 * intersection / (seg1.sum() + seg2.sum())


def align_segmentations(list_segs, plot=True):
    # Taille de référence (supposée identique pour tous les inputs)
    shape_ref = list_segs[0].shape
    aligned_segs = []
    seg = list_segs[0]
    angle, __ = extract_orientation(seg)
    aligned = reorient_seg(seg, angle)
    centered = recenter_seg(aligned, shape_ref)
    aligned_segs.append(centered)
    for i in range(1, len(list_segs)):
        seg_previous = aligned_segs[i-1]
        seg = list_segs[i]
        angle, __ = extract_orientation(seg)
        aligned = reorient_seg(seg, angle)
        centered = recenter_seg(aligned, shape_ref)
        seg = compare_two_segments_angles(centered, seg_previous)
        aligned_segs.append(seg)
    return aligned_segs

def evaluate_aligned_segmentations(list_segs, frame_compared_rate=1):
    # We can change with other metrics to evaluate like the shape
    scores = [
        dice(list_segs[i], list_segs[i - frame_compared_rate])
        for i in range(frame_compared_rate, len(list_segs))
    ]
    return np.mean(scores), scores

def compare_two_segments_angles(seg1, seg2):
    # pas de flip car si on fait un flip cest plus une question de recalage de la forme de la souris
    dice_1 = dice(seg1, seg2)
    reoriented_seg1 = reorient_seg(seg1, 180)
    dice_2 = dice(reoriented_seg1, seg2)
    if dice_1 > dice_2:
        return seg1
    else:
        return reoriented_seg1

def largest_connected_component(binary_image):
    # Étiqueter les composantes connexes
    labeled_image = label(binary_image)

    # Obtenir les régions et trouver la plus grande (hors arrière-plan)
    regions = regionprops(labeled_image)
    if not regions:
        return np.zeros_like(binary_image)  # Retourne une image vide si rien n'est trouvé

    largest_region = max(regions, key=lambda r: r.area)

    # Créer un masque de la plus grande composante
    largest_component = (labeled_image == largest_region.label)

    return largest_component


def extract_mouse_skelet(seg):
    skelet = skeletonize(seg)
    return skelet

