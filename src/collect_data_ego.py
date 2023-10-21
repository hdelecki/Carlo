import sys
sys.path.insert(1, 'carlo')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import argparse

from carlo.utils import *
from carlo.scenarios import *
from carlo.fourway_intersection import build_world

from random import choice
import numpy as np
import pandas as pd

from carlo.world import World
from carlo.agents import Car, RectangleBuilding, Pedestrian, Painting
from carlo.geometry import Point
import time

from tqdm import tqdm
from multiprocessing import cpu_count, Pool
# from joblib import Parallel, delayed
import carlo.istarmap  # import to apply patch


def get_inputs(results_dir, id, ego_dirs, ts_total_min, ts_total_max):
    init_dir, final_dir = choice(ego_dirs)
    ts_total  = np.random.randint(ts_total_min, ts_total_max)

    # print((init_dir, final_dir, pos_noise, ang_noise, ts_total))
    return (results_dir, id, init_dir, final_dir, ts_total)


def loop(results_dir, id, init_dir, final_dir, ts_total):
    # Build the fourway intersection world
    dt = 0.1
    w = build_world(dt)

    c1 = spawn_car(dt, ts_total, init_dir, final_dir, pos_path_noise=0.01)
    c1.set_control(0, 0)
    w.add(c1)

    Rows = []

    for ts in range(ts_total):
        if w.collision_exists(c1): return w.close()  # return without recording

        pos_diff, ang_diff, u_steering, u_throttle = get_controls(c1, dt)

        u_throttle, _ = find_nearest(u_throttle_allowed_values, u_throttle)
        u_steering, _ = find_nearest(u_steering_allowed_values, u_steering)

        c1.set_control(u_steering, u_throttle)
    
        w.tick()
        if render: w.render(); time.sleep(dt/20)

        Rows.append([ts, c1.x, c1.y, c1.xp, c1.yp, c1.heading, init_dir, final_dir, u_steering, u_throttle])


    if close_to(final_position(final_dir), [c1.x, c1.y]):
        Data = empty_df()
        #Data = Data._append(pd.DataFrame(Rows, columns=Data.columns), ignore_index=True)
        #Data.loc[len(Data)] = pd.DataFrame(Rows, columns=Data.columns)
        Data = pd.concat([Data, pd.DataFrame(Rows, columns=Data.columns)])
        dump_csv(results_dir, Data, id, cartype="ego")

    return w.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Disable parallel processing')
    parser.add_argument('--runs', type=int, default=1000, help='Number of runs')
    parser.add_argument('--ts-total-min', type=int, default=150, help='Minimum ts_total value')
    parser.add_argument('--ts-total-max', type=int, default=151, help='Maximum ts_total value')
    parser.add_argument('--ego-dirs', nargs='+', type=str, default=['south', 'west'], help='List of ego directions')
    parser.add_argument('--results-dir', type=str, default='../csvfiles', help='Directory to save results')
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    ### Params ###
    # render = False
    # parallel = True
    u_throttle_allowed_values = np.linspace(-25, +25, 3)
    u_steering_allowed_values = np.linspace(-5, +5, 200)
    # num_of_runs = 1000
    # ts_total_min = 150
    # ts_total_max = 151
    # ego_dirs = [("south", "west")]
    # results_dir = "../csvfiles_ego"
    render = args.render
    parallel = args.parallel
    num_of_runs = args.runs
    ts_total_min = args.ts_total_min
    ts_total_max = args.ts_total_max
    ego_dirs = [tuple(args.ego_dirs[i:i+2]) for i in range(0, len(args.ego_dirs), 2)]
    results_dir = args.results_dir
    ##############

    Path(results_dir).mkdir(parents=True, exist_ok=True)  # creates new folder


    
    iterable = [get_inputs(results_dir, id, ego_dirs, ts_total_min, ts_total_max) for id in tqdm(range(num_of_runs), desc="Building scenarios")]

    if parallel:
        if render: raise Exception("Cannot render graphics while parallelized.")

        num_cores = cpu_count()-1
        with Pool(num_cores) as pool:
            for _ in tqdm(pool.istarmap(loop, iterable), total=len(iterable), desc="Playing scenarios"):
                pass

    else:
        for items in tqdm(iterable, desc="Playing scenarios"):
            loop(*items)