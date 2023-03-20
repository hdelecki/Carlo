from utils import *
from scenarios import *
from fourway_intersection import build_world

import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

# Build the fourway intersection world
dt = 0.1
w = build_world(dt)

# c1 = Car(Point(0,57), np.pi*0)
ts_total = 200

(init, final) = ('west', 'south')


c1 = spawn_rival(dt, timesteps=ts_total, init=init, final=final)
c1.set_control(0, 0)
w.add(c1)

c2 = Car(Point(100,60), np.pi, 'blue')
c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
w.add(c2)

w.render()



for ts in range(400):
    # if w.collision_exists(): # we can check if there is any collision.
    #     print('Collision exists somewhere...')

    pos_diff, ang_diff, u_steering, u_throttle = get_controls(c1, dt)
    c1.set_control(u_steering, u_throttle)
    
    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt/15) # Let's watch it 4x

    print(f"Timestep: {ts}, Pos_Diff: {pos_diff} and u_th: {u_throttle}  |  Ang_Diff: {ang_diff} and u_st: {u_steering}")

print((init, final))

