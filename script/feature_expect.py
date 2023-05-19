#!/usr/bin/env python
# note need to run viewer with python2!!!

from cmath import e
import os
import sys
sys.path.append(os.path.abspath('/root/catkin_ws/src/SoLo_TDIRL/script/irl/'))
sys.path.append(os.path.abspath('/root/catkin_ws/src/SoLo_TDIRL/script/'))
from pyparsing import empty
from distance2goal import Distance2goal
from laser2density import Laser2density
from social_distance import SocialDistance
from traj_predict import TrajPred
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import colors, markers
from nav_msgs.msg import Odometry, OccupancyGrid
# import img_utils
from utils import *
import tf
from tf.transformations import quaternion_matrix
import tf2_ros
import tf2_geometry_msgs
from collections import namedtuple
from threading import Thread
from IPython import embed
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from collections import deque
def transform_pose(input_pose, from_frame, to_frame):

    # **Assuming /tf2 topic is being broadcasted
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = input_pose
    pose_stamped.header.frame_id = from_frame
    pose_stamped.header.stamp = rospy.Time.now()

    try:
        # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(4.0))
        return output_pose_stamped

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise

Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution

        self.Distance2goal = Distance2goal(gridsize=gridsize, resolution=resolution)
        self.goal = PoseStamped()
        self.received_goal = False
        self.Laser2density = Laser2density(gridsize=gridsize, resolution=resolution)
        self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)
        self.SocialDistance = SocialDistance(gridsize=gridsize, resolution=resolution)

        ### Replace with esfm
        self.sub_people = rospy.Subscriber("sim/agent_poses", PoseArray, self.people_callback, queue_size=100)
        self.sub_goal = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=100)
        self.robot_pose = [0.0, 0.0]
        self.previous_robot_pose = []
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
        self.reward_pub = rospy.Publisher("reward_map", OccupancyGrid, queue_size=1000)
        self.discrete_step_counter = np.zeros(int(self.gridsize[0]*0.5/self.resolution))
        self.counter = 0
        self.robot_poses = deque()
        self.trajs = deque()
        self.bad_feature = False
        # self.initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        # self.initpose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initpose_callback, queue_size=1)
        # self.initpose = PoseWithCovarianceStamped()
        # self.initpose_get = False
    
    # def initpose_callback(self,data):
    #     self.initpose = data
    #     self.initpose_get = True

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
        print("inside traj callback")
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,data):
            # print(percent_change)
        agent_poses = data.poses
        # people_stamped = np.array([transform_pose(people, "my_map_frame", "map") for people in agent_poses])
        self.pose_people = np.array([[people.position.x,people.position.y, people.position.z, people.orientation.x, people.orientation.y, people.orientation.z, people.orientation.w] for people in agent_poses])
        self.pose_people_tf = np.empty((0,4 ,4), float)
        for people_pose in self.pose_people:
            rot = people_pose[3:]
            pose_people_tf = quaternion_matrix(rot)
            pose_people_tf[0][3] = people_pose[0]
            pose_people_tf[1][3] = people_pose[1]
            pose_people_tf[2][3] = people_pose[2]
            self.pose_people_tf = np.append(self.pose_people_tf, np.array([pose_people_tf]), axis=0)
    def goal_callback(self,data):
        self.goal = data
        self.received_goal = True
        # self.robot_poses = deque()
        # self.trajs = deque()
        # self.bad_feature = False
        print("Goal Received")

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
        # pose = [-pose[1], pose[0]]

        if pose[0] < self.gridsize[1]*self.resolution and pose[0] > -0.5*self.resolution \
            and pose[1] > -0.5*self.gridsize[0] and pose[1] < 0.5*self.gridsize[0]:

            # pose[1] = max(0,pose[1])
            
            # y = min(((self.gridsize[1])*self.resolution - pose[1]) // self.resolution, 2)
            y = ((self.gridsize[1]-0.5)*self.resolution - pose[0]) // self.resolution

            x = (-pose[1] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
            # print([x, y]) # (1,2) -> (1,1) -> (0,1)
            if (x<0 or y<0):
                return None
            return [x, y]
        else:
            return None
        
    def get_local_goal(self):
        if(not self.received_goal):
            return None 
        else:
            pass

    def get_current_feature(self):
        # 
        self.localcost_feature = self.Laser2density.temp_result
        # print("Local cost feature is ", self.localcost_feature)
        self.social_distance_feature = np.ndarray.tolist(self.SocialDistance.get_features())
        # feature_list = [self.social_distance_feature]
        # self.current_feature = np.array([self.distance_feature[i] + self.localcost_feature[i] + self.traj_feature[i] + [0.0] for i in range(len(self.distance_feature))])
        self.distance_feature = [0 for i in range(self.gridsize[0] * self.gridsize[1])]
        # print("Current feature is ", self.current_feature)
        if (self.received_goal):
            self.distance_feature = self.Distance2goal.get_feature_matrix(self.goal)
            if (not self.distance_feature):
                self.bad_feature = True
                return
            else:
                self.bad_feature = False
            if (np.linalg.norm(self.distance_feature-np.zeros(len(self.distance_feature)))<0.1):
                self.recived_goal = False
                print(self.distance_feature)
                print("Finished this goal, need new one")
                exit(0)
        self.current_feature = np.array([[self.distance_feature[i]] for i in range(len(self.distance_feature))])
        self.feature_maps.append(np.array(self.current_feature).T)
        reward_map = OccupancyGrid()
        reward_map.header.stamp = rospy.Time.now()
        reward_map.header.frame_id = "base_link"
        reward_map.info.resolution = self.resolution
        reward_map.info.width = self.gridsize[0]
        reward_map.info.height = self.gridsize[1]
        reward_map.info.origin.position.x = 0
        reward_map.info.origin.position.y = - (reward_map.info.width / 2.0) * reward_map.info.resolution
        reward_map.data = [int(cell) for cell in self.current_feature]
        self.reward_pub.publish(reward_map)
        single_feature = np.array(self.current_feature).T
        if (self.received_goal):
            fm_file = "../dataset/fm/fm_"+str(self.counter)+".npz"
            np.savez(fm_file, *single_feature)
            self.counter +=1

    def get_index_in_robot_frame(self, origin_pose, current_pose):
        robot_pose_rb = [0.0,0.0]
        R = np.dot(np.linalg.inv(origin_pose), current_pose)
        robot_pose_rb = np.dot(R, np.array([[0, 0, 0, 1]]).T)
        robot_pose_rb = [robot_pose_rb[0][0], robot_pose_rb[1][0]]
        index = self.in_which_cell(robot_pose_rb)
        return index, robot_pose_rb
        
    def get_expect(self):
        if (not self.received_goal):
            return
        R1 = self.get_robot_pose()
        try:
            if ((R1 == self.robot_poses[-1]).all()):
                return 
        except:
            pass
        self.robot_poses.append(R1)
        index, pose = self.get_index_in_robot_frame(R1,R1)
        unraveled_index = index[1]*self.gridsize[1]+index[0]
        self.trajs.append([unraveled_index])
        self.get_current_feature()
        if (self.bad_feature):
            return
        
        self.feature_expect = np.array([0 for i in range(len(self.current_feature[0]))], dtype=np.float64)
        percent_temp = 0        
        traj_files = []
        print("Len of robot_poses and current counter and trajs", len(self.robot_poses), self.counter, len(self.trajs))
        remove_indices = []
        distance = np.sqrt((self.robot_pose[0] - self.goal.pose.position.x)**2+(self.robot_pose[1] - self.goal.pose.position.y)**2)
        for i in range(len(self.robot_poses)):
            # Robot pose
            index, robot_pose_rb = self.get_index_in_robot_frame(self.robot_poses[i], R1)
            if(distance < 0.2 or not index):
                remove_indices.append(i)
                traj_counter = int(i+(self.counter - len(self.robot_poses)))
                
                traj_files.append(["../dataset/trajs/trajs_"+str(traj_counter)+".npz"])
                print("wanting to remove indices ", i, "Traj counter is ", traj_counter, len(traj_files))
                if (distance <0.1):
                    print("Finished a goal! ")
                    self.received_goal = False
                #### Save the traj queue here #### 
                continue
            unraveled_index = index[1]*self.gridsize[1]+index[0]
            print("Relative robot_pose is ", robot_pose_rb, "Index is ", index)
            if(not unraveled_index in self.trajs[i]):
                self.trajs[i].append(unraveled_index)
            # print("trajs array is ", self.trajs[i])
        print("distance ", distance)
            # Whether the robot reaches the goal
        
        for j in range(len(remove_indices)):
            index = int(remove_indices[j])
            np.savez(traj_files[j][0], self.trajs[index])
            del self.robot_poses[index]
            # np.savez(traj_files[index][0], self.trajs[index])
            del self.trajs[index]
            remove_indices = remove_indices-np.ones(len(remove_indices))
            print("writing the traj file and removing index ", index, remove_indices)
            # print("distance: ", distance)
        # self.traj = [self.trajectory[i][1]*self.gridsize[1]+self.trajectory[i][0] for i in range(len(self.trajectory))]

        # if(len(self.traj) > 1):
        #     self.trajs.append(np.array(self.traj))
        
        # discount = [(1/e)**i for i in range(len(self.trajectory))]
        # # for i in range(len(discount)):

        # #     self.feature_expect += np.dot(self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])], discount[i])
        
        # self.trajectory = []

        # num_changes = abs(sum(self.percent_reward)) / self.robot_distance

        # print("Normalized sudden change: ", num_changes)
	
        # print(self.feature_maps)

    def rot2eul(self, R) :

        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular :
            z = np.arctan2(R[1,0], R[0,0])
        else :
            z = 0

        return z

    # def reset_robot(self):
    #     self.initpose_pub.publish(self.initpose)
        # print("Publish successfully")

        



if __name__ == "__main__":
        rospy.init_node("Feature_expect",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = FeatureExpect(resolution=0.5, gridsize=(3,3))

        fm_file = "../dataset/fm/fm.npz"
        traj_file = "../dataset/trajs/trajs.npz"
        # while(not feature.initpose_get):
        #     rospy.sleep(0.1)
        # feature.reset_robot()
        rospy.sleep(1)
        # while(not feature.received_goal):
        #     rospy.sleep(0.1)
        # np.savez(fm_file, *feature.feature_maps)
        print("Feature map is ", feature.feature_maps)
        print("Rospy shutdown", rospy.is_shutdown())
        while(not rospy.is_shutdown()):
            feature.get_expect()
            rospy.sleep(0.5)