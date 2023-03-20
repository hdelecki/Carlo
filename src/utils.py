import pandas as pd
import numpy as np
import time

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)
   
def empty_df():
    df_values = {'Timestep' : [],
                'X_Position': [],
                'Y_Position': [],
                'X_Velocity': [],
                'Y_Velocity': [],
                'Heading'   : [],
                'U_Steering': [],
                'U_Throttle': [],
                }

    return pd.DataFrame(df_values)

def dump_csv(obj, id):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    savename = f"../csvfiles/{timestamp}_{id}.csv"
    return obj.to_csv(savename, index=False)