from cmath import exp
from numpy import average
import tf
from math import sqrt, pow
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

class Distance2goal():

    def __init__(self, gridsize=(3,3), resolution=1):
        # gridsize: a tuple describe the size of grid, default (3,3)
        self.gridsize = gridsize
        self.resolution = resolution
        self.listener = tf.TransformListener()

    
    def get_feature_matrix(self, goal):
        # Goal should be in PoseStamped form
        result = [0 for i in range(self.gridsize[0] * self.gridsize[1])]
        try:
            self.listener.waitForTransform("/map", "/base_link", rospy.Time(), rospy.Duration(4.0))
            goal_in_base = self.listener.transformPose("/base_link", goal)
            for x in range(self.gridsize[0]):
                for y in range(self.gridsize[1]):
                    grid_center_x, grid_center_y = self.get_grid_center_position([x , y])
                    distance = sqrt(pow(grid_center_x - goal_in_base.pose.position.x, 2) + pow(grid_center_y - goal_in_base.pose.position.y, 2))
                    # distance = exp(abs(distance))
                    result[y * self.gridsize[1] + x] = distance
        except:
            print("Do not get transform!")

        max_distance = max(result)
        min_distance = min(result)

        result = [[(result[i]-min_distance) / (max_distance - min_distance)] for i in range(len(result))]
        # result = [[result[i]] for i in range(len(result))]

        # ave_dis = sum(result)/(self.gridsize[0]*self.gridsize[1])

        # std_dev = np.std(result)

        # for i in range(len(result)):
        #     if(result[i] < ave_dis + std_dev and result[i] > ave_dis - std_dev):
        #         result[i] = [0,1,0]
        #     elif(result[i] > ave_dis + std_dev):
        #         result[i] = [0,0,1]
        #     else:
        #         result[i] = [1,0,0]
        return result

    def get_grid_center_position(self, index):
        center_x = self.resolution / 2.0 + self.resolution * index[0] - self.resolution*(self.gridsize[0] / 2.0)
        center_y = self.resolution * (self.gridsize[0] - index[1] - 1) + self.resolution / 2.0

        center_x, center_y = center_y, -center_x
        return (center_x, center_y)

if __name__ == "__main__":
    rospy.init_node("distance2goal",anonymous=False)
    distance2goal = Distance2goal(gridsize=(3,3), resolution=0.5)
    rospy.sleep(1)
    while not rospy.is_shutdown():
        data = PoseStamped()
        data.pose.position.x = 5
        data.pose.position.y = -5
        data.pose.position.z = 0
        data.header.frame_id = "/map"
        result = distance2goal.get_feature_matrix(data)
        print(result[:3])
        print(result[3:6])
        print(result[6:9])
        # print(result[12:16])
        # print(result)
        print("------------------------")





        