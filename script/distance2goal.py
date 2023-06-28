from cmath import exp
from numpy import average
import tf
from math import sqrt, pow
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from IPython import embed
import tf2_ros
import tf2_geometry_msgs
class Distance2goal():

    def __init__(self, gridsize=(3,3), resolution=1):
        # gridsize: a tuple describe the size of grid, default (3,3)
        self.gridsize = gridsize
        self.resolution = resolution
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer) 

    def transform_pose(self, input_pose, from_frame, to_frame):

        # **Assuming /tf2 topic is being broadcasted

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now() - rospy.Duration(1)
        print("Time in distance is ", pose_stamped.header.stamp)
        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, to_frame, timeout = rospy.Duration(1))
            return output_pose_stamped

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("No Transform found?")
            raise
            return None 
    def get_feature_matrix(self, goal):
        # Goal should be in PoseStamped form
        result = [0 for i in range(self.gridsize[0] * self.gridsize[1])]
        frame_id = goal.header.frame_id
        
        goal_in_base = self.transform_pose(goal.pose, frame_id, "base_link")
        if (not goal_in_base):
            return result
        # print("Goal in base link ", goal_in_base)  
        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                grid_center_x, grid_center_y = self.get_grid_center_position([x , y])
                distance = sqrt(pow(grid_center_x - goal_in_base.pose.position.x, 2) + pow(grid_center_y - goal_in_base.pose.position.y, 2))
                # distance = exp(abs(distance))
                result[y * self.gridsize[1] + x] = distance
        
        #### Normalizing the goal distance #####
        # max_distance = max(result)
        # min_distance = min(result)
        # result = [[(result[i]-min_distance) / (max_distance - min_distance)] for i in range(len(result))]

        ##########################################
        result = [[result[i]] for i in range(len(result))]

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





        