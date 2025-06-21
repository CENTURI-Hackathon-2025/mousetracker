import numpy as np
import pandas as pd
import numpy as np

def save_traj_to_csv(x,y,t, path_and_name='test_csv'):

    df = pd.DataFrame({'x':x, 'y':y, 't':t})

    df.to_csv(f"{path_and_name}.csv")

def read_csv_traj(path_and_name):

    df = pd.read_csv(f"{path_and_name}.csv")

    x = df.x.to_numpy()
    y = df.y.to_numpy()
    t = df.t.to_numpy()    

    return x,y,t

def compute_speed(x,y,t):

    dx_arr = np.diff(x)
    dy_arr = np.diff(y)
    dt_arr = np.diff(t)

    v = np.sqrt(dx_arr**2 + dx_arr**2)/dt_arr

    return v



