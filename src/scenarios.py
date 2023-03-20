from utils import *
from simple_pid import PID

from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point

def populate_rival_directions():
    dirs = ['south', 'north', 'east', 'west']
    return [(x,y) for x in dirs for y in dirs if x != y]

def init_angle(dir):
    table = {'south': 0.5*np.pi,
             'north': 1.5*np.pi,
             'east':  np.pi,
             'west':  0.0}
    return table[dir]

def init_position(dir):
    table = {'south': [63,40],
             'north': [57,80],
             'east':  [80,63],
             'west':  [40,57]}
    return np.array(table[dir])

def final_position(dir):
    table = {'south': [57,40],
             'north': [63,80],
             'east':  [80,57],
             'west':  [40,63]}
    return np.array(table[dir])

def mid_point_position(init, final):
    right_turns = [('north', 'west'),
                   ('west', 'south'),
                   ('south', 'east'),
                   ('east', 'north')]

    if (init, final) in right_turns:  # right turn: turns a tighter angle 
        table = {'south': [56,60],
                 'north': [64,60],
                 'east':  [60,56],
                 'west':  [60,64]}
    else:  # left or straight
        table = {'south': [60,60],
                 'north': [60,60],
                 'east':  [60,60],
                 'west':  [60,60]}
    return np.array(table[final])

def final_angle(dir):
    table = {'south': 1.5*np.pi,
             'north': 0.5*np.pi,
             'east':  0.0,
             'west':  np.pi}
    return table[dir]

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

def get_position_path(initial, final, timesteps, noise_std=0.0):
    mid_point = mid_point_position(initial, final)
    first_half = np.linspace(init_position(initial), mid_point, timesteps//2)
    second_half = np.linspace(mid_point, final_position(final), timesteps//2)

    result = np.vstack([first_half, second_half])
    return result + np.random.normal(scale=noise_std, size=result.shape)

def get_angular_path(initial, final, timesteps, noise_std=0.0):
    mid = np.linspace(initial, final, timesteps//3).reshape(-1,1)
    z0 = np.ones_like(mid) * initial
    z1 = np.ones_like(mid) * final

    result = np.vstack([z0, mid, z1])
    return result + np.random.normal(scale=noise_std, size=result.shape)

def get_pos_diff(real, target, idx):
    dx = target-real
    # return np.sign(dx) * np.linalg.norm(dx)
    # return np.linalg.norm(dx)
    return dx[idx]
    # return np.sum(dx)


def get_ang_diff(real, target):
    return (target-real)[0]


def get_controls(car, dt):
    ts = car.ts_now

    if ts < car.ts_total//2:
        idx=ref_position(car.init_dir)
        sgn=init_ref_sign(car.init_dir)
    else: 
        idx=ref_position(car.final_dir)
        sgn=final_ref_sign(car.final_dir)

    if ts < min(car.ang_path.shape[0], car.pos_path.shape[0]):
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


def spawn_rival(dt, timesteps, init='west', final='north', pos_path_noise=0.0, ang_path_noise=0.0):
    if (init, final) not in populate_rival_directions(): raise Exception("Invalid `init` of `final`.")

    car = Car(Point(*init_position(init)), init_angle(init), init_dir=init, final_dir=final, ts_total=timesteps)

    car.pos_path = get_position_path(initial=init, final=final, timesteps=timesteps, noise_std=pos_path_noise)
    car.ang_path = get_angular_path(initial=car.heading, final=final_angle(final), timesteps=timesteps, noise_std=ang_path_noise)

    car.pos_controller = PID(Kp=20.0, Ki=0.1, Kd=20.0, sample_time=dt, setpoint=0)
    car.ang_controller = PID(Kp=2.0, Ki=0.001, Kd=0.1, sample_time=dt, setpoint=0)


    # Special cases due to angle wrapping:
    if (init, final) == ('north', 'east'):
        car.ang_path = get_angular_path(initial=car.heading, final=2*np.pi, timesteps=timesteps)
    elif (init, final) == ('west', 'south'):
        car.ang_path = get_angular_path(initial=2*np.pi, final=final_angle(final), timesteps=timesteps)

    return car