import numpy as np
import tensorflow as tf

import os
import pickle
import argparse
import ipdb
import sys

sys.path.append(os.path.abspath('./social_lstm_tf/'))

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask
import rospy
from pedsim_msgs.msg import AgentStates

class Traj_Predictor():

    def __init__(self):
        parser = argparse.ArgumentParser()
        # Observed length of the trajectory parameter
        parser.add_argument('--obs_length', type=int, default=5,
                            help='Observed length of the trajectory')
        # Predicted length of the trajectory parameter
        parser.add_argument('--pred_length', type=int, default=3,
                            help='Predicted length of the trajectory')
        # Test dataset
        parser.add_argument('--test_dataset', type=int, default=0,
                            help='Dataset to be tested on')
        # Parse the parameters
        self.sample_args = parser.parse_args()
        # Define the path for the config file for saved args
        with open(os.path.join('social_lstm_tf/save', 'social_config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)
        # Create a SocialModel object with the saved_args and infer set to true
        self.model = SocialModel(self.saved_args, True)
        
        # self.graph = tf.Graph()
        # Initialize a TensorFlow session
        self.sess = tf.InteractiveSession()
        # Initialize a saver
        self.saver = tf.train.Saver()
        # Get the checkpoint state for the model
        self.ckpt = tf.train.get_checkpoint_state('social_lstm_tf/save')

        self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)

        self.pose_sub = rospy.Subscriber("/pedsim_simulator/simulated_agents", AgentStates, self.pose_callback, queue_size=100)

        self.obs_traj = np.empty((0,30 ,3), float) # 5 x 30 x 3

        self.delta_T = 0.4

    def pose_callback(self, states):
        # print("Into callback")
        agent_pose = np.empty((0,3), float)
        agent_pose = np.zeros((30,3), float)
        # print(states.header.frame_id)
        # print(len(states.agent_states))
        for i in range(len(states.agent_states)):
            agent_pose[i] = np.array([states.agent_states[i].id - 1, states.agent_states[i].pose.position.x, states.agent_states[i].pose.position.y])
        # for state in states.agent_states:
        #     pose = np.array([[state.id - 1, state.pose.position.x, state.pose.position.y]])
        #     agent_pose = np.append(agent_pose, pose, axis=0)
        # print(agent_pose.shape)

        if self.obs_traj.shape[0] == self.sample_args.obs_length:
            self.obs_traj = self.obs_traj[1:]
            self.obs_traj = np.append(self.obs_traj, np.array([agent_pose]), axis=0)

        elif self.obs_traj.shape[0] < self.sample_args.obs_length:
            self.obs_traj = np.append(self.obs_traj, np.array([agent_pose]), axis=0)

        rospy.sleep(self.delta_T)


    def get_predicted_trajs(self):

        if(self.obs_traj.shape[0] == self.sample_args.obs_length):
            # print(self.obs_traj)
            x_batch = self.obs_traj / 29.0

            d_batch = 0

            if d_batch == 0 :
                dimensions = [29, 29]
            else:
                dimensions = [720, 576]

            grid_batch = getSequenceGridMask(x_batch, dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

            # add three columns zeros to x_batch

            x_batch = np.vstack([x_batch, np.ones((3,30,3))])

            # print(x_batch.shape)            
            obs_traj = x_batch[:self.sample_args.obs_length]
            obs_grid = grid_batch[:self.sample_args.obs_length]

            # obs_traj is an array of shape obs_length x maxNumPeds x 3


            complete_traj = self.model.sample(self.sess, obs_traj, obs_grid, dimensions, x_batch, self.sample_args.pred_length)
            # print( complete_traj)
            return complete_traj * 29.0
            # print(complete_traj[5:].shape)
        return None

    def test(self):

            # print(self.obs_traj)
            x_batch = np.array([[[0.     , 0.     , 0.     ],
            [1.     , 0.43125, 0.68125],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

        [[0.     , 0.     , 0.     ],
            [1.     , 0.43438, 0.71458],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

        [[0.     , 0.     , 0.     ],
            [1.     , 0.44219, 0.74792],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

        [[0.     , 0.     , 0.     ],
            [1.     , 0.44688, 0.78333],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

        [[0.     , 0.     , 0.     ],
            [1.     , 0.45156, 0.81458],
            [2.     , 0.52344, 0.91875],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

            [[0.     , 0.     , 0.     ],
            [1.     , 0.45156, 0.81458],
            [2.     , 0.52344, 0.91875],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

            [[0.     , 0.     , 0.     ],
            [1.     , 0.45156, 0.81458],
            [2.     , 0.52344, 0.91875],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]],

            [[0.     , 0.     , 0.     ],
            [1.     , 0.45156, 0.81458],
            [2.     , 0.52344, 0.91875],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ],
            [0.     , 0.     , 0.     ]]

            ])

            # x_batch = x_batch * 10

            d_batch = 0

            if d_batch == 0 :
                dimensions = [640, 480]
            else:
                dimensions = [720, 576]

            grid_batch = getSequenceGridMask(x_batch, dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

            obs_traj = x_batch[:self.sample_args.obs_length]
            obs_grid = grid_batch[:self.sample_args.obs_length]

            # obs_traj is an array of shape obs_length x maxNumPeds x 3
            

            complete_traj = self.model.sample(self.sess, obs_traj, obs_grid, dimensions, x_batch, self.sample_args.pred_length)

            print(complete_traj)

if __name__=="__main__":
    traj_pred = Traj_Predictor()
    rospy.init_node("Traj_pred")
    while(not rospy.is_shutdown()):
        traj_pred.get_predicted_trajs()
        # traj_pred.test()
        rospy.sleep(1)