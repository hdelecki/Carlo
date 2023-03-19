import pandas as pd
import numpy as np

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)
   
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

def get_pos_diff(real, target, idx):
    dx = target-real
    # return np.sign(dx) * np.linalg.norm(dx)
    # return np.linalg.norm(dx)
    return dx[idx]
    # return np.sum(dx)


def get_ang_diff(real, target):
    return target-real


def ref_position(dir):
    table = {'south': 1,
             'north': 1,
             'east':  0,
             'west':  0}
    return table[dir]


def init_ref_sign(dir):
    table = {'south': +1,
             'north': -1,
             'east':  -1,
             'west':  +1}
    return table[dir]


def final_ref_sign(dir):
    table = {'south': -1,
             'north': +1,
             'east':  +1,
             'west':  -1}
    return table[dir]

def get_controls(car, dt):
    ts = car.ts_now

    if ts < car.ts_total//2:
        idx=ref_position(car.init_dir)
        sgn=init_ref_sign(car.init_dir)
    else: 
        idx=ref_position(car.final_dir)
        sgn=final_ref_sign(car.final_dir)
        # import ipdb; ipdb.set_trace()

    if ts < car.pos_path.shape[0]:
        # Get throttle value.
        pos_diff = sgn * get_pos_diff(car.pos_path[ts], np.array([car.x, car.y]), idx=idx)
        u_throttle = car.pos_controller(pos_diff, dt=dt)

        # Get steering value.
        ang_diff = get_ang_diff(car.ang_path[ts], car.heading)
        u_steering = car.ang_controller(ang_diff, dt=dt)
        
    else:
        # Get throttle value.
        pos_diff = sgn * get_pos_diff(car.pos_path[-1], np.array([car.x, car.y]), idx)
        u_throttle = car.pos_controller(pos_diff, dt=dt)

        # Get steering value.
        ang_diff = get_ang_diff(car.ang_path[-1], car.heading)
        u_steering = car.ang_controller(ang_diff, dt=dt)

    car.ts_now += 1
    return pos_diff, ang_diff, u_steering, u_throttle