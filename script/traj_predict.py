import sys
import os

sys.path.append(os.path.abspath('./irl/'))

from cmath import e
from unittest import result
import rospy
import tf
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt
import img_utils
from Traj_Predictor import Traj_Predictor

class TrajPred():

    def __init__(self, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution
        self.discount_factor = 1 / e
        self.listener = tf.TransformListener()
        self.TrajPredictor = Traj_Predictor()


    def get_feature_matrix(self):
        '''
        traj = [[[0.      0.      0.     ]
                [1.      0.43125 0.68125]]

                [[0.      0.      0.     ]
                [1.      0.43438 0.71458]]
                        ....
                [[0.      0.      0.     ]
                [1.      0.44219 0.74792]]]
        '''
        traj_matrix = self.TrajPredictor.get_predicted_trajs()
        while(traj_matrix is None):
            traj_matrix = self.TrajPredictor.get_predicted_trajs()

        # get_transform = False
        result = np.array([0.0 for i in range(self.gridsize[0] * self.gridsize[1])])
        # print(traj_matrix)
        for i in range(traj_matrix.shape[0]):
            # print("i",i)
            global_poses = [PoseStamped() for k in range(traj_matrix.shape[1])]
            local_poses = [PoseStamped() for k in range(traj_matrix.shape[1])]
            for j in range(len(global_poses)):
                # print("j",j)
            #    while(not get_transform): 
                global_poses[j].pose.position.x = traj_matrix[i][j][1]
                global_poses[j].pose.position.y = traj_matrix[i][j][2]
                # print(global_poses[j].pose.position.x, global_poses[j].pose.position.y)
                global_poses[j].pose.position.z = 0
                global_poses[j].header.frame_id = "/map"
                try:
                    # while(not self.listener.frameExists("/base_link") or not self.listener.frameExists("/map")):
                    self.listener.waitForTransform("/map", "/base_link", rospy.Time.now(), rospy.Duration(4.0))
                    # print("Get the transform")
                    local_poses[j] = self.listener.transformPose("/base_link", global_poses[j])

                    local_index = self.get_grid_index([-local_poses[j].pose.position.y, local_poses[j].pose.position.x])
                    # print(local_poses[j].pose.position.x, local_poses[j].pose.position.y)
                    # print(global_poses[j].pose.position.x, global_poses[j].pose.position.y)
                    if(self.inside_grid(local_index[0], local_index[1])):
                        result[int(local_index[1]*self.gridsize[1] + local_index[0])] += 1 * self.discount_factor**(i)
                except:
                    print("Do not get transform!")
        
        # max_distance = max(result)
        # min_distance = min(result)
        # if max_distance - min_distance != 0:
        #     result = [[(result[i]-min_distance) / (max_distance - min_distance)] for k in range(len(result))]
        # else:
        #     result = [[0] for k in range(len(result))]
        return result

    def get_grid_index(self, position):
        # first calculate index in y direction
        # x, y = y, -x

        y = (self.gridsize[1]*self.resolution - position[1]) // self.resolution

        x = (position[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
        return (x,y)    

    def inside_grid(self, x, y):
        if(x>=0 and x<self.gridsize[0] and y>=0 and y<self.gridsize[1]):
            return True
        else:
            return False


if __name__ == "__main__":
    rospy.init_node("traj_pred")

    traj_pred = TrajPred()
    traj =  np.array([[[0.  ,    5.   ,   -4.],
                [1.  ,    2.5,  -1]],

                [[0.  ,    5.   ,   -5.     ],
                [1.    ,  2.5 , 0]],

                [[0.   ,   5.   ,   -6.     ],
                [1.   ,   1.5,  0]]])
    # while(not rospy.is_shutdown()):

    # rospy.sleep(1)
    while(not rospy.is_shutdown()):
        result = traj_pred.get_feature_matrix()
        print(np.reshape(result, traj_pred.gridsize))
        img_utils.heatmap2d(np.reshape(result, traj_pred.gridsize), 'Traj Cost', block=False)
        plt.show()
        # print(np.reshape(result, traj_pred.gridsize))
    
    '''
    x x x x x g
    x x x x x x
    x x x x x x 
    o x x x x x 
    o o x x x x
    x x x x x x
    r x x x x x
    
    '''

