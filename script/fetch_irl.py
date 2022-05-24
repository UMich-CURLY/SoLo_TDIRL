import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

import sys
import os
sys.path.append(os.path.abspath('./irl/'))

import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from maxent_irl import *
from utils import *
from lp_irl import *

# N_gridworld = 1
Step = namedtuple('Step','cur_state next_state')

class IRL_Agent():
    def __init__(self):
        self.trajs = []
        self.fms  =[]
        self.read_csv()
        self.N_STATE = self.fms[0][0].shape[0]
        # Assume the grid is square.
        self.H = int(np.sqrt(self.N_STATE))
        self.W = int(np.sqrt(self.N_STATE))
        self.ACT_RAND = 0.3
        self.GAMMA = 0.9
        self.LEARNING_RATE = 0.02
        self.N_ITERS = 20

        

    def read_csv(self):
        # Get the feature map and trajectory.
        traj = []
        with np.load('trajs.npz') as data:
            for i in range(len(data.files)):
                traj_name = 'arr_{}'.format(i)
                for j in range(len(data[traj_name]) - 1):
                    traj.append(Step(cur_state=int(data[traj_name][j]), next_state=int(data[traj_name][j+1])))
                self.trajs.append(traj)
                traj = []

        with np.load('fm.npz') as data:
            for i in range(len(data.files)):
                fm_name = 'arr_{}'.format(i)
                self.fms.append(data[fm_name])

        # print(self.trajs)
        print(len(self.fms))
        print(len(self.trajs))
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
    def deep_irl(self):
        # feed the feature maps and traj into network and train.
        rmap_gt = np.zeros([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        rewards = deep_maxent_irl_fetch(self.fms, P_a, self.GAMMA, self.trajs, self.LEARNING_RATE, self.N_ITERS)

    def save_weight(self):
        pass

    def eval(self):
        pass


if __name__=="__main__":
    irl_agent = IRL_Agent()
    # irl_agent.deep_irl()
    # irl_agent.read_csv()