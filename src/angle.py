import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def read_csv_traj(path_and_name):
    df = pd.read_csv(f"{path_and_name}.csv")
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    t = df.t.to_numpy()

    return x,y,t

def plot_histo(liste, titre, seuil):

    plt.hist(liste, bins=10, color='skyblue', edgecolor='black')

    # Positions des intervalles
    pi_2 = np.pi / 2
    neg_pi_2 = -np.pi / 2 # remappé dans [0, 2π] ≈ 4.71

    # Afficher des bandes verticales transparentes
    plt.axvspan(pi_2 - seuil, pi_2 + seuil, color='red', alpha=0.3, label='+π/2 ± seuil')
    plt.axvspan(neg_pi_2 - seuil, neg_pi_2 + seuil, color='green', alpha=0.3, label='−π/2 ± seuil')
    plt.title(titre)
    plt.xlabel("Valeurs")
    plt.ylabel("Fréquence")
    plt.show()
def compute_speeds(x,t):
    dx=np.diff(x)
    dt = np.diff(t)
    v = dx / dt
    return v
def compute_norm(vx, vy):
    return(np.sqrt(vx ** 2 + vy ** 2))
def angle_oriente(x1,y1,x2,y2):
    """Renvoie l'angle orienté entre deux vecteurs 2D en radians"""
    cross_product = x1 * y2 - y1 * x2  # produit vectoriel en 2D
    dot_product= x1 * x2 + y1 * y2  # produit scalaire
    angle = np.arctan2(cross_product, dot_product)
    return angle  # en radians, dans [-π, π]
def compute_angles(x,y,t):
    vx=compute_speeds(x,t)
    vy = compute_speeds(y,t)
    angles = angle_oriente(vx[:-1], vy[:-1], vx[1:], vy[1:])
    # norms = compute_norm(vx, vy)
    # norms1 = norms[:-1] #liste des normes des vitesses de la 1ere à l'avant-derniere
    # norms2 = norms[1:] #liste des normes des vitesses de la 2eme à la derniere
    # dot_products = vx[:-1] * vx[1:] + vy[:-1] * vy[1:] #Produit scalaire entre deux vecteurs vitesse consécutifs
    # Angles (en radians)
    # cos_theta = dot_products / (norms1 * norms2)  # éviter division par zéro
    # # cos_theta = np.clip(cos_theta, -1.0, 1.0)  # éviter erreurs numériques
    # angles = np.arccos(cos_theta)  # angle en radians
    return(angles)
#angle en radians entre -pi et pi
def count_angle_above_threshold(angles,thr):
    where_above_thr = np.where(np.abs(angles)>thr,1,0)
    nb_above_thr = np.sum(where_above_thr)
    return nb_above_thr
def plot_counts_discarded_angles(angles):
    n=len(angles)
    thresholds = np.linspace(0,np.pi, n)
    counts = [count_angle_above_threshold(angles,thr) for thr in thresholds]

    plt.plot(thresholds, counts, label="Counts en fonction du seuil de l'angle", linewidth=1)
    plt.legend()
    plt.title("Trajectoire avec changements brusques")
    plt.show()


def count_bad_angles(angles, seuil):
    angles = np.array(angles)
    # Normaliser les angles dans [-π, π]
    angles = (angles + np.pi) % (2 * np.pi) - np.pi

    count_bad_angle = np.sum((angles >= (-np.pi / 2 - seuil)) & (angles <= (-np.pi / 2 + seuil)))
    return count_bad_angle


filename = r"C:\Users\thoma\Documents\ASciences\AAMU\HACKATON\Hackathon_videos\to_15\Video1_TO15"
# filename = r"C:\Users\thoma\Documents\ASciences\AAMU\HACKATON\assia_t0_15_1"
x,y,t = read_csv_traj(filename)
angles=compute_angles(x,y,t)
n=len(angles)


# plot_counts_discarded_angles_with_threshold(angles)

for diviseur in  [4,5,6,7,8]:
    seuil = np.pi / diviseur
    bad_angles = count_bad_angles(angles, np.pi/diviseur)
    plot_histo(angles, f"seuil = pi/{diviseur} : {bad_angles} / {n} ; proportion = {round(bad_angles/n*100, 1)} %", seuil)
    print(f"seuil = pi/{diviseur} : {bad_angles} / {n} ; proportion = {round(bad_angles/n*100, 1)} %")

# Couleur selon le temps
# plt.figure(figsize=(6, 6))
# plt.scatter(x, y, c=t, cmap="viridis")
# plt.colorbar(label="Temps")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Trajectoire colorée par le temps")
# plt.axis("equal")
# plt.grid(True)
# plt.show()

# print(compute_angles(x,y,t))
# print(count_angle_above_threshold(compute_angles(x,y,t), np.pi/2), "/ ", len(x) )