from cmath import e

from pyparsing import empty
from distance2goal import Distance2goal
from laser2density import Laser2density
from social_distance import SocialDistance
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from pedsim_msgs.msg import TrackedPersons
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import colors, markers
import tf
from tf.transformations import quaternion_matrix

from collections import namedtuple
from threading import Thread
from traj_predict import TrajPred


# Step = namedtuple('Step','cur_state action next_state')
Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, goal, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution

        self.Distance2goal = Distance2goal(gridsize=gridsize, resolution=resolution)
        self.goal = goal
        self.Laser2density = Laser2density(gridsize=gridsize, resolution=resolution)
        self.TrajPred = TrajPred(gridsize=gridsize, resolution=resolution)
        self.SocialDistance = SocialDistance(gridsize=gridsize, resolution=resolution)
        self.sub_people = rospy.Subscriber("/pedsim_visualizer/tracked_persons", TrackedPersons,self.people_callback, queue_size=100)
        self.robot_pose = [0.0, 0.0]
        self.robot_pose_rb = [0.0, 0.0]
        self.position_offset = [0.0,0.0]
        self.trajectory = []
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
        self.delta_t = 0.0
        self.pose_people_tf = np.empty((0,4 ,4), float)
        

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

    def people_callback(self,data):

        if len(self.previous_velocity) == 0:
            self.previous_velocity = np.array([np.sqrt(people.twist.twist.linear.x**2 + people.twist.twist.linear.y**2) for people in data.tracks])
        else:
            self.velocity_people = np.array([np.sqrt(people.twist.twist.linear.x**2 + people.twist.twist.linear.y**2) for people in data.tracks])
            self.delta_v = self.velocity_people - self.previous_velocity
            self.percent_change = self.delta_v / self.previous_velocity * 100
            self.previous_velocity = self.velocity_people
            self.velocity_people_record.append(self.percent_change.tolist())
            # print(percent_change)
        self.pose_people = np.array([[people.pose.pose.position.x,people.pose.pose.position.y, people.pose.pose.position.z, people.pose.pose.orientation.x, people.pose.pose.orientation.y, people.pose.pose.orientation.z, people.pose.pose.orientation.w] for people in data.tracks])
        self.pose_people_tf = np.empty((0,4 ,4), float)
        for people_pose in self.pose_people:
            rot = people_pose[3:]
            pose_people_tf = quaternion_matrix(rot)
            pose_people_tf[0][3] = people_pose[0]
            pose_people_tf[1][3] = people_pose[1]
            pose_people_tf[2][3] = people_pose[2]
            self.pose_people_tf = np.append(self.pose_people_tf, np.array([pose_people_tf]), axis=0)

        # print(self.pose_people_tf)
        # print(tf_matrix)
        # return tf_matrix
        # rospy.sleep(self.delta_t)

        # print("delta_v: ",self.delta_v)
        
    def in_which_cell(self, pose):
        pose = [-pose[1], pose[0]]
        # print("Current robot pose in fixed robot frame:",pose)
        # print(pose) # (0,0) -> (0,0.5) -> ()
        if pose[0] <= self.gridsize[0]*self.resolution / 2.0 and pose[0] >= -self.gridsize[0]*self.resolution / 2.0 \
            and pose[1] >= -0.1 and pose[1] <=  self.gridsize[1]*self.resolution:

            pose[1] = max(0,pose[1])
            
            y = min((self.gridsize[1]*self.resolution - pose[1]) // self.resolution, 2)

            x = (pose[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
            # print([x, y]) # (1,2) -> (1,1) -> (0,1)
            return [x, y]
        else:
            # The robot is out of the cell

            return None

    def get_current_feature(self):
        self.distance_feature = self.Distance2goal.get_feature_matrix(self.goal)
        self.localcost_feature = self.Laser2density.temp_result
        self.traj_feature, _ = self.TrajPred.publish_feature_matrix()
        self.traj_feature = np.ndarray.tolist(self.traj_feature)
        self.social_distance_feature = np.ndarray.tolist(self.SocialDistance.get_features())
        # print(self.traj_feature)
        # print(self.distance_feature[0], self.localcost_feature[0])
        self.current_feature = np.array([self.distance_feature[i] + self.localcost_feature[i] + self.traj_feature[i] + self.social_distance_feature[i] for i in range(len(self.distance_feature))])
        # print(self.current_feature)

    def get_expect(self):
        R1 = self.get_robot_pose()

        # self.position_offset = self.robot_pose
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

            # people pose
            pose_temp = self.pose_people_tf
            percent_change = self.percent_change

            
            for i in range(len(pose_temp)):
                R3 = pose_temp[i]
                R_temp = np.dot(np.linalg.inv(R1), R3)
                pose_people = np.dot(R_temp, np.array([[0, 0, 0, 1]]).T)
                pose_people = [pose_people[0][0], pose_people[1][0]]
                index = self.in_which_cell(pose_people)
                # if(index and self.percent_change[i] > 20):
                if(percent_change[i] > 20 and index):
                    percent_temp -= 1
                    print("ID: ",i,",    -1")


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
            index = self.in_which_cell(self.robot_pose_rb)
            if(not index in self.trajectory and index):
                self.trajectory.append(index)
            
            step_list = []
            rospy.sleep(0.1)
        self.percent_reward = np.append(self.percent_reward,percent_temp)
        # for i in range(len(self.trajectory) - 1):
        #     step_list.append(Step(cur_state=self.trajectory[i], next_state=self.trajectory[i+1]))
        
        trajs = [self.trajectory[i][1]*self.gridsize[1]+self.trajectory[i][0] for i in range(len(self.trajectory))]

        self.trajs.append(np.array(trajs))
        
        discount = [(1/e)**i for i in range(len(self.trajectory))]
        for i in range(len(discount)):
            # print("Feature value:", self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])])
            self.feature_expect += np.dot(self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])], discount[i])
        
        self.trajectory = []
	
        print(self.feature_maps)

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
        data.pose.position.y = 0
        data.header.frame_id = "/map"
        feature = FeatureExpect(goal=data, resolution=0.5)

        # fm_file = TemporaryFile()
        fm_file = "../dataset/fm_test/fm0.npz"
        traj_file = "../dataset/trajs_test/trajs0.npz"
        percent_change_file = "../dataset/percent_change_test/percent_change0.npz"
        # feature.get_expect()
        while(not rospy.is_shutdown()):
            feature.get_expect()
            # try:
            #     plt.plot([a[0] for a in feature.velocity_people_record])
            #     plt.pause(0.005)
            # except:
            #     pass
            # feature.get_expect(fm_file)
            # np.savez(fm_file, *feature.feature_maps)
            # np.savez(traj_file, *feature.trajs)
            # threads = []
            # for n in range(1, 11):
            #     t = Thread(target=task, args=(n,))
            #     threads.append(t)
            #     t.start()
            #     rospy.sleep(0.3)

            # for t in threads:
            #     t.join()
            
            np.savez(fm_file, *feature.feature_maps)
            np.savez(traj_file, *feature.trajs)
            print(feature.percent_reward)
            np.savez(percent_change_file, *np.array(feature.percent_reward))

            # plt.ion() # enable real-time plotting
            # plt.figure(1) # create a plot
            # plt.plot(125,250, markersize=15, marker=10, color="red")
            # plt.imshow(1.0 - 1./(1.+np.exp(feature.Laser2density.map_logs)), 'Greys')
            # plt.pause(0.005)
        

        

