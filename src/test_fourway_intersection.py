from utils import *
from scenarios import *
from fourway_intersection import build_world

import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

def test(u_throttle_allowed_values, u_steering_allowed_values):

    # Build the fourway intersection world
    dt = 0.1
    w = build_world(dt)

    ts_total = 100

    for (init, final) in populate_rival_directions():

        c1 = spawn_rival(dt, timesteps=ts_total, init=init, final=final, pos_path_noise=0, ang_path_noise=0)
        c1.set_control(0, 0)
        w.add(c1)

        c2 = Car(Point(100,60), np.pi, 'blue')
        c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
        w.add(c2)

        w.render()


        for ts in range(ts_total):
            if w.collision_exists(): # we can check if there is any collision.
                print('Collision exists somewhere...')

            pos_diff, ang_diff, u_steering, u_throttle = get_controls(c1, dt)

            u_throttle = find_nearest(u_throttle_allowed_values, u_throttle)
            u_steering = find_nearest(u_steering_allowed_values, u_steering)

            c1.set_control(u_steering, u_throttle)
            
            w.tick() # This ticks the world for one time step (dt second)
            w.render()
            time.sleep(dt/50) # Let's watch it 4x

            print(f"Timestep: {ts}, Pos_Diff: {pos_diff} and u_th: {u_throttle}  |  Ang_Diff: {ang_diff} and u_st: {u_steering}")

        c1.set_control(0, -np.Inf)
        print((init, final))

        import ipdb; ipdb.set_trace()

if __name__ == "__main__":

    ### Params ###
    u_throttle_allowed_values = np.linspace(-2, +2, 3)
    u_steering_allowed_values = np.linspace(-1, +1, 20)
    ##############

    test(u_throttle_allowed_values, u_steering_allowed_values)