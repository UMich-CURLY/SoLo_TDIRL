from cmath import e
from pyexpat import features
from distance2goal import Distance2goal
from laser2density import Laser2density
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, markers

class FeatureExpect():
    def __init__(self, goal, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution

        self.Distance2goal = Distance2goal(gridsize=gridsize, resolution=resolution)
        self.goal = goal
        self.Laser2density = Laser2density(gridsize=gridsize, resolution=resolution)
        self.robot_pose = (0.0, 0.0)
        self.robot_pose_rb = (0.0, 0.0)
        self.position_offset = (0.0,0.0)
        self.trajectory = []

        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.get_robot_pose, queue_size=1)
    
    def get_robot_pose(self, data):
        self.robot_pose = (data.pose.pose.position.x, data.pose.pose.position.y)

    def in_which_cell(self):
        if self.robot_pose_rb[0] <= self.gridsize[0]*self.resolution / 2.0 and self.robot_pose_rb[0] >= -self.gridsize[0]*self.resolution / 2.0 \
            and self.robot_pose_rb[1] >= 0 and self.robot_pose_rb <=  self.gridsize[1]*self.resolution:
            y = (self.gridsize[1]*self.resolution - self.robot_pose_rb[1]) // self.resolution

            x = (self.robot_pose_rb[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution

            return (x,y)
        else:
            # The robot is out of the cell
            return None

    def get_current_feature(self):
        self.distance_feature = self.Distance2goal.get_feature_matrix(self.goal)
        self.localcost_feature = self.Laser2density.temp_result
        print(self.distance_feature[0], self.localcost_feature[0])
        self.current_feature = np.array([self.distance_feature[i] + self.localcost_feature[i] for i in range(len(self.distance_feature))])

    def get_expect(self):
        self.position_offset = self.robot_pose
        self.get_current_feature()
        self.feature_expect = np.array([0 for i in range(len(self.current_feature[0]))])

        self.robot_pose_rb = (0.0,0.0)
        while(self.in_which_cell()):
            self.robot_pose_rb = (self.robot_pose[0] - self.position_offset[0], self.robot_pose[1] - self.position_offset[1])
            index_x, index_y = self.in_which_cell()
            if(not [index_x, index_y] in self.trajectory):
                self.trajectory.append([index_x, index_y])
            rospy.sleep(0.1)
        discount = [(1/e)**i for i in range(len(self.trajectory))]
        for i in range(len(discount)):
            self.feature_expect += np.multiply(self.current_feature[self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0]], discount[i])
            
        print(self.feature_expect)

if __name__ == "__main__":
        rospy.init_node("Feature_expect",anonymous=False)
        data = PoseStamped()
        data.pose.position.x = 9
        data.pose.position.y = 9
        data.header.frame_id = "/map"
        feature = FeatureExpect(goal=data)

        while(not rospy.is_shutdown()):
            feature.get_expect()
            plt.ion() # enable real-time plotting
            plt.figure(1) # create a plot
            plt.plot(125,250, markersize=15, marker=10, color="red")
            plt.imshow(1.0 - 1./(1.+np.exp(feature.Laser2density.map_logs)), 'Greys')
            plt.pause(0.005)

        




    


        