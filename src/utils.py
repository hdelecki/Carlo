import pandas as pd
import numpy as np

def save_history(t, x, y, xp, yp):
    df_values = {'TimeStamp': t,
                'Position_X': x,
                'Position_Y': y,
                'Velocity_X': xp,
                'Velocity_Y': yp,
                }

    return pd.DataFrame(df_values)

def ego_position_path(initial, final, timesteps):
    mid_point = np.array([60,60])
    first_half = np.linspace(initial, mid_point, timesteps//2)
    second_half = np.linspace(mid_point, final, timesteps//2)
    return np.vstack([first_half, second_half])

def ego_angular_path(initial, final, timesteps):
    mid = np.linspace(initial, final, timesteps//5).reshape(-1,1)
    z0 = np.ones_like(mid) * initial
    z1 = np.ones_like(mid) * final
    # import ipdb; ipdb.set_trace()
    return np.vstack([z0, z0, mid, z1, z1])

def get_pos_diff(real, target):
    dx = target-real
    # return np.sign(dx) * np.linalg.norm(dx)
    # return np.linalg.norm(dx)
    return np.sum(dx)

def get_ang_diff(real, target):
    return target-real