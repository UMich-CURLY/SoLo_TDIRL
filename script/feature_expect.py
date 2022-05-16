from cmath import e
from distance2goal import Distance2goal
from laser2density import Laser2density
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import colors, markers
import tf
from tf.transformations import quaternion_matrix

class FeatureExpect():
    def __init__(self, goal, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution

        self.Distance2goal = Distance2goal(gridsize=gridsize, resolution=resolution)
        self.goal = goal
        self.Laser2density = Laser2density(gridsize=gridsize, resolution=resolution)
        self.robot_pose = [0.0, 0.0]
        self.robot_pose_rb = [0.0, 0.0]
        self.position_offset = [0.0,0.0]
        self.trajectory = []
        self.tf_listener =  tf.TransformListener()

        # self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.get_robot_pose, queue_size=1)

    def get_robot_pose(self):
        self.tf_listener.waitForTransform("/map", "/base_link", rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        self.robot_pose = [trans[0], trans[1]]
        tf_matrix = quaternion_matrix(rot)
        tf_matrix[0][3] = trans[0]
        tf_matrix[1][3] = trans[1]
        tf_matrix[2][3] = trans[2]
        # print(tf_matrix)
        return tf_matrix

    def in_which_cell(self):
        self.robot_pose_rb = [-self.robot_pose_rb[1], self.robot_pose_rb[0]]
        # print("Current robot pose in fixed robot frame:",self.robot_pose_rb)
        # print(self.robot_pose_rb) # (0,0) -> (0,0.5) -> ()
        if self.robot_pose_rb[0] <= self.gridsize[0]*self.resolution / 2.0 and self.robot_pose_rb[0] >= -self.gridsize[0]*self.resolution / 2.0 \
            and self.robot_pose_rb[1] >= -0.1 and self.robot_pose_rb[1] <=  self.gridsize[1]*self.resolution:

            self.robot_pose_rb[1] = max(0,self.robot_pose_rb[1])
            
            y = min((self.gridsize[1]*self.resolution - self.robot_pose_rb[1]) // self.resolution, 2)

            x = (self.robot_pose_rb[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
            # print([x, y]) # (1,2) -> (1,1) -> (0,1)
            return [x, y]
        else:
            # The robot is out of the cell

            return None

    def get_current_feature(self):
        self.distance_feature = self.Distance2goal.get_feature_matrix(self.goal)
        self.localcost_feature = self.Laser2density.temp_result
        # print(self.distance_feature[0], self.localcost_feature[0])
        self.current_feature = np.array([self.distance_feature[i] + self.localcost_feature[i] for i in range(len(self.distance_feature))])

    def get_expect(self):
        R1 = self.get_robot_pose()

        # self.position_offset = self.robot_pose
        self.get_current_feature()
        self.feature_expect = np.array([0 for i in range(len(self.current_feature[0]))], dtype=np.float64)

        self.robot_pose_rb = [0.0,0.0]
        
        index = self.in_which_cell()
        while(index):

            R2 = self.get_robot_pose()
            

            # pose_vector = np.array([[self.robot_pose[0]],
            #                         [self.robot_pose[1]],
            #                         [0],
            #                         [1]])

            R = np.dot(np.linalg.inv(R1), R2)

            # ang = self.rot2eul(R)

            # R2d = np.array([[cos(ang), sin(ang)],
            #                 [-sin(ang), cos(ang)]])

            # self.robot_pose_rb = np.dot(R2d, np.array([[R[0][3]], 
            #                                            [R[1][3]]]))
            self.robot_pose_rb = np.dot(R, np.array([[0, 0, 0, 1]]).T)

            self.robot_pose_rb = [self.robot_pose_rb[0][0], self.robot_pose_rb[1][0]]
            
            # print("Current robot pose in fixed robot frame:",self.robot_pose_rb)
            # print("Current robot pose in map frame:", self.robot_pose)
            # print("Fixed frame in map:", [R1[0][3], R1[1][3]])

            # self.robot_pose_rb = [self.robot_pose[0] - self.position_offset[0], self.robot_pose[1] - self.position_offset[1]]
            index = self.in_which_cell()
            if(not index in self.trajectory and index):
                self.trajectory.append(index)
            rospy.sleep(0.1)
        discount = [(1/e)**i for i in range(len(self.trajectory))]
        for i in range(len(discount)):
            print("Feature value:", self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])])
            self.feature_expect += np.dot(self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])], discount[i])
        
        self.trajectory = []

        print(self.feature_expect)

    def rot2eul(self, R) :

        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular :
            z = np.arctan2(R[1,0], R[0,0])
        else :
            z = 0

        return z

if __name__ == "__main__":
        rospy.init_node("Feature_expect",anonymous=False)
        data = PoseStamped()
        data.pose.position.x = 6
        data.pose.position.y = -6
        data.header.frame_id = "/map"
        feature = FeatureExpect(goal=data)
        
        while(not rospy.is_shutdown()):
            feature.get_expect()
            plt.ion() # enable real-time plotting
            plt.figure(1) # create a plot
            plt.plot(125,250, markersize=15, marker=10, color="red")
            plt.imshow(1.0 - 1./(1.+np.exp(feature.Laser2density.map_logs)), 'Greys')
            plt.pause(0.005)
