import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image



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

