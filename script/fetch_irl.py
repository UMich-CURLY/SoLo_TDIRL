from pyexpat import features
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
import random

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

# N_gridworld = 1
Step = namedtuple('Step','cur_state next_state')

class IRL_Agent():
    def __init__(self):
        self.trajs = []
        self.fms  =[]
        self.percent_change = []
        self.read_csv()
        self.N_STATE = self.fms[0][0].shape[0]
        # Assume the grid is square.
        self.H = int(np.sqrt(self.N_STATE))
        self.W = int(np.sqrt(self.N_STATE))
        self.ACT_RAND = 0.3
        self.GAMMA = 0.9
        self.LEARNING_RATE = 0.001
        self.N_ITERS = 50


    def read_csv(self):
        # Get the feature map and trajectory.
        # test_1 contain the trajectory loss
        traj = []
        for filename in os.listdir("../dataset/trajs_test_1/"):
            # print(filename)
            number_str = ""
            for m in filename:
                if m.isdigit():
                    number_str = number_str + m

            with np.load(os.path.join("../dataset/trajs_test_2", filename)) as data:
                file_fm_name = "fm" + number_str + ".npz"
                with np.load(os.path.join("../dataset/fm_test_2", file_fm_name)) as data2:
                    file_percent_change_name = "percent_change" + number_str + ".npz"
                    with np.load(os.path.join("../dataset/percent_change_test_2", file_percent_change_name)) as data3:
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
    def train(self):
        # feed the feature maps and traj into network and train.
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        # deep_maxent_irl_traj_loss(feat_maps, P_a, gamma, trajs,percent_change,  lr, n_iters)
        rewards = deep_maxent_irl_traj_loss(self.fms, P_a, self.GAMMA, self.trajs,self.percent_change, self.LEARNING_RATE, self.N_ITERS)
        img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward Map - Deep Maxent', block=False)
        plt.show()


    def train_without_trajloss(self):
        # feed the feature maps and traj into network and train.
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        # deep_maxent_irl_traj_loss(feat_maps, P_a, gamma, trajs,percent_change,  lr, n_iters)
        rewards = deep_maxent_irl_no_traj_loss(self.fms, P_a, self.GAMMA, self.trajs,self.percent_change, self.LEARNING_RATE, self.N_ITERS)
        img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward Map - Deep Maxent', block=False)
        plt.show()

    def test(self):
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        rewards, policy = get_irl_reward_policy(self.fms[0], P_a)
        print(self.fms[0].shape)
        img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward Map - Deep Maxent', block=False)
        plt.show()

    def eval(self):
        nn_r = DeepIRLFC(self.fms[0].shape[0], self.LEARNING_RATE, 3, 3)
        nn_r.load_weights()
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        correct_pairs = 0
        total_pairs = 500
        for i in range(total_pairs):
            num1 = random.randint(0, len(self.percent_change)-1)
            num2 = random.randint(0, len(self.percent_change)-1)
            while(num1 == num2):
                num2 = random.randint(0, len(self.percent_change)-1)
            rewards1, policy1 = get_irl_reward_policy(nn_r, self.fms[num1], P_a)
            rewards2, policy2 = get_irl_reward_policy(nn_r, self.fms[num2], P_a)

            total_reward1 = get_reward_sum_from_policy(rewards1, policy1, (self.H, self.W))
            total_reward2 = get_reward_sum_from_policy(rewards2, policy2, (self.H, self.W))

            if((total_reward1 >= total_reward2 and self.percent_change[num1] >= self.percent_change[num2]) or \
                (total_reward1 <= total_reward2 and self.percent_change[num1] <= self.percent_change[num2])):
                if(self.percent_change[num1] == self.percent_change[num2]):
                    total_pairs -= 1.0
                else:
                    correct_pairs += 1.0
                
        print("The accuracy is: ", correct_pairs / total_pairs * 100)
        return correct_pairs / total_pairs * 100
        



if __name__=="__main__":
    irl_agent = IRL_Agent()
    irl_agent.eval()
    # irl_agent.read_csv()