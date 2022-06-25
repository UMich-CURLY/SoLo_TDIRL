import sys
import os

sys.path.append(os.path.abspath('./irl/'))

from cmath import e
from unittest import result
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import matplotlib.pyplot as plt
import img_utils
from Traj_Predictor import Traj_Predictor
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
# from std_msgs.msg import Float64MultiArray


class TrajPred():

    def __init__(self, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution
        self.discount_factor = 1 / e
        self.listener = tf.TransformListener()
        self.path_pub = rospy.Publisher("test_path", Path, queue_size=1000)
        self.TrajPredictor = Traj_Predictor()
        self.feature_pub = rospy.Publisher("traj_matrix", numpy_msg(Floats), queue_size=100)


    def publish_feature_matrix(self):
        '''
        traj = [[[0.      0.      0.     ]
                [1.      0.43125 0.68125]]

                [[0.      0.      0.     ]
                [1.      0.43438 0.71458]]
                        ....
                [[0.      0.      0.     ]
                [1.      0.44219 0.74792]]]
        '''
        traj_matrix_complete = self.TrajPredictor.get_predicted_trajs()
        while(traj_matrix_complete is None):
            traj_matrix_complete = self.TrajPredictor.get_predicted_trajs()
        traj_matrix = traj_matrix_complete[5:]
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
        
        max_distance = max(result)
        min_distance = min(result)
        if max_distance - min_distance != 0:
            result = np.array([[(result[k]-min_distance) / (max_distance - min_distance)] for k in range(len(result))])
        else:
            result = np.array([[0.0] for k in range(len(result))])
        # result = np.array([[result[k]] for k in range(len(result))])
        # data = Float64MultiArray()
        # data.data = result
        # print(data.data)
        self.feature_pub.publish(result)
        return result, traj_matrix_complete


    def publish_test_traj(self, result):
        # result = result[:5]
        path = Path()
        path.header.frame_id = "map"
        for arr in result:
            position = PoseStamped()
            position.header.frame_id = "map"
            position.pose.position.x = arr[0][1]
            position.pose.position.y = arr[0][2]
            # print(arr[0])
            path.poses.append(position)
        self.path_pub.publish(path)


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
        result, traj_matrix = traj_pred.publish_feature_matrix()
        # traj_pred.publish_test_traj(traj_matrix)
        # rospy.sleep(0.1)
        # print(np.reshape(result, traj_pred.gridsize))
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

