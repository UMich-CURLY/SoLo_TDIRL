from cmath import sqrt
from pyexpat import features
from turtle import shape

# from matplotlib.colors import same_color
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
import random

import sys
import os

from IPython import embed
# from script.irl.deep_maxent_irl import deep_maxent_irl_fetch

sys.path.append(os.path.abspath('/root/catkin_ws/src/SoLo_TDIRL/script/irl/'))

import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl_ori import *
# from maxent_irl import *
from utils import *
# from lp_irl import *
import yaml
# N_gridworld = 1
Step = namedtuple('Step','cur_state next_state')

class IRL_Agent():
    def __init__(self):
        self.trajs = []
        self.fms  =[]
        self.percent_change = []
        self.ACT_RAND = 0.3
        self.GAMMA = 0.9
        self.LEARNING_RATE = 0.001
        self.N_ITERS = 1
        self.good_percent = 0.50
        self.data_path = "../dataset_2"
        self.weights_folder = "weights"
        if self.weights_folder not in os.listdir(self.data_path):
            __ = os.system("mkdir " + self.data_path+"/"+self.weights_folder)
        else:
            print("Do you want to re train?")
            embed()
        self.weight_path = self.data_path+"/"+self.weights_folder+"/saved_weights"
        
    def calculate_good_percent(self):
        total = len(self.percent_change)
        bad_num = np.count_nonzero(self.percent_change)
        return 1 - float(bad_num) / total


    def randomly_remove(self):
        print("good percent is: ", self.calculate_good_percent())
        if(self.calculate_good_percent() > self.good_percent):
            while(self.calculate_good_percent() > self.good_percent):
                good_demo_index = np.where(np.array(self.percent_change) == 0.0)[0]
                remove_index = np.random.choice(good_demo_index, size=1)[0]
                self.percent_change.pop(remove_index)
                self.fms.pop(remove_index)
                self.trajs.pop(remove_index)
        elif(self.calculate_good_percent() < self.good_percent):
            while(self.calculate_good_percent() < self.good_percent):
                good_demo_index = np.where(np.array(self.percent_change) < 0)[0]
                remove_index = np.random.choice(good_demo_index, size=1)[0]
                self.percent_change.pop(remove_index)
                self.fms.pop(remove_index)
                self.trajs.pop(remove_index)

        print("good percent is: ", self.calculate_good_percent())

    def read_csv_test(self):
        # Get the feature map and trajectory.
        # test_1 contain the trajectory loss
        traj = []
        for filename in os.listdir("../dataset/trajs_test_2/"):
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
                                self.percent_change.append(float(data3[percent_change_name]))

        self.N_STATE = self.fms[0][0].shape[0]
        # Assume the grid is square.
        self.H = int(np.sqrt(self.N_STATE))
        self.W = int(np.sqrt(self.N_STATE))
        
        self.randomly_remove()

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

    def read_csv_train(self):
        # Get the feature map and trajectory.
        # test_1 contain the trajectory loss
        traj = []
        for filename in os.listdir("../dataset/trajs_2/"):
            # print(filename)
            number_str = ""
            for m in filename:
                if m.isdigit():
                    number_str = number_str + m

            with np.load(os.path.join("../dataset/trajs_4", filename)) as data:
                file_fm_name = "fm" + number_str + ".npz"
                with np.load(os.path.join("../dataset/fm_4", file_fm_name)) as data2:
                    file_percent_change_name = "percent_change" + number_str + ".npz"
                    with np.load(os.path.join("../dataset/percent_change_1", file_percent_change_name)) as data3:
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
                               
                                self.percent_change.append(float(data3[percent_change_name]))

        # print(self.percent_change)
        self.N_STATE = self.fms[0][0].shape[0]
        # Assume the grid is square.
        self.H = int(np.sqrt(self.N_STATE))
        self.W = int(np.sqrt(self.N_STATE))
        # self.randomly_remove()
        # self.randomly_remove()
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

    def read_csv_train_no_loss(self):
        # Get the feature map and trajectory.
        # test_1 contain the trajectory loss
        traj = []
        for foldername in os.listdir(self.data_path):
            number_str = "_"
            valid = False
            for m in foldername:
                if m.isdigit():
                    valid = True
                    number_str = number_str + m
                else:
                    valid = False
            if (not valid):
                continue
            folder = self.data_path+"/demo"+number_str
            for filename in os.listdir(folder+"/trajs/"):
                # print(filename)
                number_str = "_"
                for m in filename:
                    if m.isdigit():
                        number_str = number_str + m
                
                with np.load(os.path.join(folder+"/trajs", filename)) as data:
                    file_fm_name = "fm" + number_str + ".npz"
                    with np.load(os.path.join(folder+"/fm", file_fm_name)) as data2:
                        print(file_fm_name)
                        for i in range(len(data.files)):
                            traj_name = 'arr_{}'.format(i)
                            cur_traj_len = len(data[traj_name])
                            if(cur_traj_len > 1):
                                for j in range(len(data[traj_name]) - 1):
                                    traj.append(Step(cur_state=int(data[traj_name][j]), next_state=int(data[traj_name][j+1])))
                            else:
                                traj.append(Step(cur_state=int(data[traj_name][0]), next_state=int(data[traj_name][0])))
                                # continue
                            self.trajs.append(traj)
                            traj = []
                            temp_fm = []
                            for j in range(len(data2.files)):
                                fm_name = 'arr_{}'.format(j)
                                temp_fm.append(data2[fm_name])
                            temp_fm[4] = temp_fm[4]/70
                            self.fms.append(np.array(temp_fm[0:-1]))
        # print(self.percent_change)
        self.N_STATE = self.fms[0][0].shape[0]
        # Assume the grid is square.
        self.H = int(np.sqrt(self.N_STATE))
        self.W = int(np.sqrt(self.N_STATE))

    def read_csv_train_multi_trajs(self):
        # Get the feature map and trajectory.
        # test_1 contain the trajectory loss
        traj = []
        trajs_set = []
        for filename in os.listdir("../dataset/dataset/trajs/"):
            # print(filename)
            number_str = ""
            for m in filename:
                if m.isdigit():
                    number_str = number_str + m
            with np.load(os.path.join("../dataset/dataset/trajs", filename)) as data:
                file_fm_name = "fm" + number_str + ".npz"
                with np.load(os.path.join("../dataset/dataset/fm", file_fm_name)) as data2:
                    for i in range(len(data.files)):
                        traj_name = 'arr_{}'.format(i)
                        cur_traj_len = len(data[traj_name])
                        if(cur_traj_len > 1):
                            for j in range(len(data[traj_name]) - 1):
                                traj.append(Step(cur_state=int(data[traj_name][j]), next_state=int(data[traj_name][j+1])))
                            trajs_set.append(traj)
                            traj = []
                    temp_fm = data2["arr_0"].T
                    # print(temp_fm.shape)
                    fm = np.reshape(temp_fm, [int(np.sqrt(temp_fm.shape[0])), int(np.sqrt(temp_fm.shape[0])), temp_fm.shape[1]])
                    fm = np.array([fm])
                    # print(fm.shape)
                    self.fms.append(fm)
                    self.trajs.append(trajs_set)
                    trajs_set = []
        # print(len(self.trajs))
        # print(len(self.fms))
        # print(self.fms.shape)
        self.N_STATE = fm.shape[1]*fm.shape[2]
        self.H = fm.shape[2]
        self.W = fm.shape[1]
    
    def train(self):
        # feed the feature maps and traj into network and train.
        self.read_csv_train()
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        # deep_maxent_irl_traj_loss(feat_maps, P_a, gamma, trajs,percent_change,  lr, n_iters)
        print("Starting training process")
        rewards = deep_maxent_irl_traj_loss(self.fms, P_a, self.GAMMA, self.trajs,self.percent_change, self.LEARNING_RATE, self.N_ITERS)
        img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward Map - Deep Maxent', block=False)
        plt.show()

    def flip_distance_feature(self):
        for i in range(len(self.fms)):
            self.fms[i][0] = np.ones(self.fms[i][0].shape) - self.fms[i][0]
        return True 
    
    def train_without_trajloss(self):
        # feed the feature maps and traj into network and train.
        self.read_csv_train_no_loss()
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        # deep_maxent_irl_traj_loss(feat_maps, P_a, gamma, trajs,percent_change,  lr, n_iters)
        rewards, nn_r = deep_maxent_irl_no_traj_loss_tribhi(self.fms, P_a, self.GAMMA, self.trajs, self.LEARNING_RATE, self.N_ITERS, self.weight_path)
        print ("The reward is ", rewards)

        config_vals = {"name": nn_r.name, "lr": nn_r.lr, "n_h1": nn_r.n_h1, "n_h2": nn_r.n_h2, "n_iters": self.N_ITERS, "gamma": self.GAMMA}
        with open(self.data_path+"/"+self.weights_folder+"/config.yml", 'w') as file:
            yaml.dump(config_vals, file)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        plt.subplot(2, 2, 1)
        ax1 = img_utils.heatmap2d(np.reshape(self.fms[3][0], (self.H,self.W)), 'Distance Feature', block=False)
        plt.subplot(2, 2, 2)
        if (self.fms[3].shape[1] == 2):
            ax2 = img_utils.heatmap2d(np.reshape(self.fms[3][1], (self.H,self.W)), 'obs Feature', block=False)
        plt.subplot(2, 2, 3)
        rewards, policy = get_irl_reward_policy(nn_r,self.fms[3], P_a)
        ax3 = img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward', block=False)
        # plt.subplot(2, 2, 4)
        plt.show()
        

    def train_multi_trajs(self):
        # feed the feature maps and traj into network and train.
        self.read_csv_train_multi_trajs()
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        rewards = deep_maxent_irl_mul_traj(self.fms, P_a, self.GAMMA, self.trajs, self.LEARNING_RATE, self.N_ITERS)
        img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward Map - Deep Maxent', block=False)
        plt.show()

    def test(self):
        self.read_csv_test()
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        rewards, policy = get_irl_reward_policy(self.fms[0], P_a)
        # print(self.fms[0].shape)
        img_utils.heatmap2d(np.reshape(rewards, (self.H,self.W)), 'Reward Map - Deep Maxent', block=False)
        plt.show()

    def eval(self):
        self.read_csv_test()
        nn_r = DeepIRLFC(self.fms[0].shape[0], self.LEARNING_RATE, 3, 3)
        nn_r.load_weights()
        rmap_gt = np.ones([self.H, self.W])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - self.ACT_RAND)
        P_a = gw.get_transition_mat()
        correct_pairs = 0
        total_pairs = 500
        batch_size = 100.0
        same_pair = 0
        for i in range(1, total_pairs+1):
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
                        correct_pairs += 1.0
                
            if(i % batch_size == 0):
                print("The accuracy is: ", correct_pairs / (batch_size) * 100)
                correct_pairs = 0
        return correct_pairs / total_pairs * 100
        



if __name__=="__main__":
    irl_agent = IRL_Agent()
    # irl_agent.read_csv_train_multi_trajs()

    # irl_agent.train_multi_trajs()
    # irl_agent.train()
    # irl_agent.eval()
    irl_agent.train_without_trajloss()

    # irl_agent.read_csv_test()
    # irl_agent.read_csv()