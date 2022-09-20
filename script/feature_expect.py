from cmath import e

from pyparsing import empty
from distance2goal import Distance2goal
from laser2density import Laser2density
from social_distance import SocialDistance
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from pedsim_msgs.msg import TrackedPersons, AgentStates
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import colors, markers
import tf
from tf.transformations import quaternion_matrix

from collections import namedtuple
from threading import Thread
from traj_predict import TrajPred

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, goal, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution

        self.Distance2goal = Distance2goal(gridsize=gridsize, resolution=resolution)
        self.goal = goal
        self.Laser2density = Laser2density(gridsize=gridsize, resolution=resolution)
        self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)
        self.SocialDistance = SocialDistance(gridsize=gridsize, resolution=resolution)
        self.sub_people = rospy.Subscriber("/pedsim_simulator/simulated_agents", AgentStates,self.people_callback, queue_size=100)
        self.robot_pose = [0.0, 0.0]
        self.previous_robot_pose = []
        self.robot_pose_rb = [0.0, 0.0]
        self.robot_distance = 0.0
        self.position_offset = [0.0,0.0]
        self.trajectory = []
        self.traj_feature = [[0.0] for i in range(gridsize[0] * gridsize[1])]
        self.tf_listener =  tf.TransformListener()
        self.feature_maps = []
        self.trajs = []
        self.percent_reward = []
        self.fm_dict = {}
        self.traj_dict = {}
        self.pose_people = []
        self.velocity_people = []
        self.previous_velocity = []
        self.velocity_people_record = []
        self.orientation_people = []
        self.previous_orientation = []
        self.orientation_people_record = []
        self.delta_t = 0.0
        self.pose_people_tf = np.empty((0,4 ,4), float)
        
    def get_robot_pose(self):
        self.tf_listener.waitForTransform("/map", "/base_link", rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        self.robot_pose = [trans[0], trans[1]]
        if(len(self.previous_robot_pose) == 0):
            self.previous_robot_pose = self.robot_pose
        else:
            self.robot_distance += np.sqrt((self.robot_pose[0] - self.previous_robot_pose[0])**2 + (self.robot_pose[1] - self.previous_robot_pose[1])**2)
            self.previous_robot_pose = self.robot_pose
        tf_matrix = quaternion_matrix(rot)
        tf_matrix[0][3] = trans[0]
        tf_matrix[1][3] = trans[1]
        tf_matrix[2][3] = trans[2]
        # print(tf_matrix)
        return tf_matrix

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,data):
            # print(percent_change)
        self.pose_people = np.array([[people.pose.position.x,people.pose.position.y, people.pose.position.z, people.pose.orientation.x, people.pose.orientation.y, people.pose.orientation.z, people.pose.orientation.w] for people in data.agent_states])
        self.pose_people_tf = np.empty((0,4 ,4), float)
        for people_pose in self.pose_people:
            rot = people_pose[3:]
            pose_people_tf = quaternion_matrix(rot)
            pose_people_tf[0][3] = people_pose[0]
            pose_people_tf[1][3] = people_pose[1]
            pose_people_tf[2][3] = people_pose[2]
            self.pose_people_tf = np.append(self.pose_people_tf, np.array([pose_people_tf]), axis=0)

    # def people_callback(self,data):

    #     if len(self.previous_velocity) == 0:
    #         self.previous_velocity = np.array([np.sqrt(people.twist.twist.linear.x**2 + people.twist.twist.linear.y**2) for people in data.tracks])
    #         self.previous_orientation = np.array([[people.twist.twist.linear.x , people.twist.twist.linear.y] for people in data.tracks])
    #     else:
    #         self.velocity_people = np.array([np.sqrt(people.twist.twist.linear.x**2 + people.twist.twist.linear.y**2) for people in data.tracks])
    #         self.orientation_people = np.array([[people.twist.twist.linear.x , people.twist.twist.linear.y] for people in data.tracks])
    #         self.delta_v = self.velocity_people - self.previous_velocity
    #         self.delta_ori = self.orientation_people - self.previous_orientation
    #         self.percent_change_ori = self.delta_ori / self.previous_orientation * 100
    #         self.percent_change_ori = np.array([max(percent_ori) for percent_ori in self.percent_change_ori])
    #         self.percent_change = self.delta_v / self.previous_velocity * 100
    #         self.previous_orientation = self.orientation_people
    #         self.previous_velocity = self.velocity_people
    #         self.velocity_people_record.append(self.percent_change.tolist())
    #         self.orientation_people_record.append(self.percent_change_ori.tolist())
    #         # print(percent_change)
    #     self.pose_people = np.array([[people.pose.pose.position.x,people.pose.pose.position.y, people.pose.pose.position.z, people.pose.pose.orientation.x, people.pose.pose.orientation.y, people.pose.pose.orientation.z, people.pose.pose.orientation.w] for people in data.tracks])
    #     self.pose_people_tf = np.empty((0,4 ,4), float)
    #     for people_pose in self.pose_people:
    #         rot = people_pose[3:]
    #         pose_people_tf = quaternion_matrix(rot)
    #         pose_people_tf[0][3] = people_pose[0]
    #         pose_people_tf[1][3] = people_pose[1]
    #         pose_people_tf[2][3] = people_pose[2]
    #         self.pose_people_tf = np.append(self.pose_people_tf, np.array([pose_people_tf]), axis=0)

    # def people_callback(self,data):

    #     if len(self.previous_velocity) == 0:
    #         self.previous_velocity = np.array([np.sqrt(people.twist.twist.linear.x**2 + people.twist.twist.linear.y**2) for people in data.tracks])
    #     else:
    #         self.velocity_people = np.array([np.sqrt(people.twist.twist.linear.x**2 + people.twist.twist.linear.y**2) for people in data.tracks])
    #         self.delta_v = self.velocity_people - self.previous_velocity
    #         self.percent_change = self.delta_v / self.previous_velocity * 100
    #         self.previous_velocity = self.velocity_people
    #         self.velocity_people_record.append(self.percent_change.tolist())
    #         # print(percent_change)
    #     self.pose_people = np.array([[people.pose.pose.position.x,people.pose.pose.position.y, people.pose.pose.position.z, people.pose.pose.orientation.x, people.pose.pose.orientation.y, people.pose.pose.orientation.z, people.pose.pose.orientation.w] for people in data.tracks])
    #     self.pose_people_tf = np.empty((0,4 ,4), float)
    #     for people_pose in self.pose_people:
    #         rot = people_pose[3:]
    #         pose_people_tf = quaternion_matrix(rot)
    #         pose_people_tf[0][3] = people_pose[0]
    #         pose_people_tf[1][3] = people_pose[1]
    #         pose_people_tf[2][3] = people_pose[2]
    #         self.pose_people_tf = np.append(self.pose_people_tf, np.array([pose_people_tf]), axis=0)

        # print(self.pose_people_tf)
        # print(tf_matrix)
        # return tf_matrix
        # rospy.sleep(self.delta_t)

        # print("delta_v: ",self.delta_v)
        
    def in_which_cell(self, pose):
        pose = [-pose[1], pose[0]]

        if pose[0] <= self.gridsize[0]*self.resolution / 2.0 and pose[0] >= -self.gridsize[0]*self.resolution / 2.0 \
            and pose[1] >= -0.1 and pose[1] <=  self.gridsize[1]*self.resolution:

            pose[1] = max(0,pose[1])
            
            y = min((self.gridsize[1]*self.resolution - pose[1]) // self.resolution, 2)

            x = (pose[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
            # print([x, y]) # (1,2) -> (1,1) -> (0,1)
            return [x, y]
        else:

            return None

    def get_current_feature(self):
        self.distance_feature = self.Distance2goal.get_feature_matrix(self.goal)
        self.localcost_feature = self.Laser2density.temp_result
        self.social_distance_feature = np.ndarray.tolist(self.SocialDistance.get_features())
        self.current_feature = np.array([self.distance_feature[i] + self.localcost_feature[i] + self.traj_feature[i] + [0.0] for i in range(len(self.distance_feature))])


    def get_expect(self):
        R1 = self.get_robot_pose()

        self.get_current_feature()

        self.feature_maps.append(np.array(self.current_feature).T)

        self.feature_expect = np.array([0 for i in range(len(self.current_feature[0]))], dtype=np.float64)

        self.robot_pose_rb = [0.0,0.0]
        
        index = self.in_which_cell(self.robot_pose_rb)
        percent_temp = 0
        while(index):
            
            # Robot pose
            R2 = self.get_robot_pose()
            
            R = np.dot(np.linalg.inv(R1), R2)

            self.robot_pose_rb = np.dot(R, np.array([[0, 0, 0, 1]]).T)

            self.robot_pose_rb = [self.robot_pose_rb[0][0], self.robot_pose_rb[1][0]]

            index = self.in_which_cell(self.robot_pose_rb)
            if(not index in self.trajectory and index):
                self.trajectory.append(index)
            
            step_list = []
            rospy.sleep(0.1)

        trajs = [self.trajectory[i][1]*self.gridsize[1]+self.trajectory[i][0] for i in range(len(self.trajectory))]

        self.trajs.append(np.array(trajs))
        
        discount = [(1/e)**i for i in range(len(self.trajectory))]
        for i in range(len(discount)):

            self.feature_expect += np.dot(self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])], discount[i])
        
        self.trajectory = []

        num_changes = abs(sum(self.percent_reward)) / self.robot_distance

        print("Normalized sudden change: ", num_changes)
	
        # print(self.feature_maps)

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
        data.pose.position.x = 0
        data.pose.position.y = 4
        data.header.frame_id = "/map"
        feature = FeatureExpect(goal=data, resolution=1)


        fm_file = "../dataset/fm_4/fm4.npz"
        traj_file = "../dataset/trajs_4/trajs4.npz"

        while(not rospy.is_shutdown()):
            feature.get_expect()
            
            np.savez(fm_file, *feature.feature_maps)
            np.savez(traj_file, *feature.trajs)
        

        

