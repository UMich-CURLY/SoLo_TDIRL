import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

import sys
import os

# from script.irl.deep_maxent_irl import deep_maxent_irl_fetch

sys.path.append(os.path.abspath('./irl/'))

import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from maxent_irl import *
from utils import *
from lp_irl import *

Step = namedtuple('Step','cur_state next_state')

def read_csv(self):
    # Get the feature map and trajectory.
    traj = []
    for filename in os.listdir("../dataset/trajs_test/"):
        # print(filename)
        number_str = ""
        for m in filename:
            if m.isdigit():
                number_str = number_str + m

        with np.load(os.path.join("../dataset/trajs_test", filename)) as data:
            file_fm_name = "fm" + number_str + ".npz"
            with np.load(os.path.join("../dataset/fm_test", file_fm_name)) as data2:
                file_percent_change_name = "percent_change" + number_str + ".npz"
                with np.load(os.path.join("../dataset/percent_change_test", file_percent_change_name)) as data3:
                    print("data1:", len(data.files))
                    print("data2:", len(data2.files))
                    print("data3:", len(data3.files))
                    for i in range(len(data.files)):
                        traj_name = 'arr_{}'.format(i)
                        cur_traj_len = len(data[traj_name])
                        if(cur_traj_len > 1):
                            for j in range(len(data[traj_name]) - 1):
                                traj.append(Step(cur_state=int(data[traj_name][j]), next_state=int(data[traj_name][j+1])))
                            self.trajs.append(traj)
                            traj = []
                    # for j in range(len(data2.files)):
                            fm_name = 'arr_{}'.format(i)
                            self.fms.append(data2[fm_name])

                            percent_change_name = 'arr_{}'.format(i)
                            self.percent_change.append(data3[percent_change_name])

                

    # print(len(self.fms))
    # print(len(self.trajs))
    # print(self.fms)
    '''
    trajs = [[Step(cur_state=7.0, next_state=4.0), Step(cur_state=4.0, next_state=1.0)], 
                [Step(cur_state=7.0, next_state=4.0), Step(cur_state=4.0, next_state=5.0)]]
    fms = [array([[0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0]]), 
        
        array([[0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]])]
    '''

def accuracy_rate(feature_map1, traj1, feature_map2, traj2):
    # 
    nn_r = DeepIRLFC(self.NUM_FEATURE, 0.01, 3, 3)
        # print("before load weight")
    nn_r.load_weights()
    pass