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
    table = {'south': (63,0),
             'north': (57,120),
             'east':  (120,63),
             'west':  (0,57)}
    return table[dir]

def final_position(dir):
    table = {'south': (57,0),
             'north': (63,120),
             'east':  (120,57),
             'west':  (0,63)}
    return table[dir]

def final_angle(dir):
    table = {'south': 1.5*np.pi,
             'north': 0.5*np.pi,
             'east':  0.0,
             'west':  np.pi}
    return table[dir]


def spawn_rival(dt, timesteps, init='west', final='north'):
    if (init, final) not in populate_rival_directions(): raise Exception("Invalid `init` of `final`.")

    car = Car(Point(*init_position(init)), init_angle(init), init_dir=init, final_dir=final, ts_total=timesteps)

    car.pos_path = get_position_path(initial=np.array([*init_position(init)]), final=np.array([*final_position(final)]), timesteps=timesteps)
    car.ang_path = get_angular_path(initial=car.heading, final=final_angle(final), timesteps=timesteps)

    car.pos_controller = PID(Kp=10.0, Ki=0.1, Kd=10.0, sample_time=dt, setpoint=0)
    car.ang_controller = PID(Kp=2.0, Ki=0.001, Kd=0.01, sample_time=dt, setpoint=0)


    # Special cases due to angle wrapping:
    if (init, final) == ('north', 'east'):
        car.ang_path = get_angular_path(initial=car.heading, final=2*np.pi, timesteps=timesteps)
    elif (init, final) == ('west', 'south'):
        car.ang_path = get_angular_path(initial=2*np.pi, final=final_angle(final), timesteps=timesteps)

    return car