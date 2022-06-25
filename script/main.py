import sys
import os

sys.path.append(os.path.abspath('./irl/'))
from distance2goal import Distance2goal
from laser2density import Laser2density
import numpy as np
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from controller import PathPublisher
import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from traj_predict import TrajPred
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

class Agent():
    def __init__(self, goal, gridsize,resolution):
        
        rospy.init_node("main")

        self.NUM_FEATURE = 5

        # self.robot_pose = np.array([0, 0])

        self.result = False

        self.result_sub = rospy.Subscriber("/trajectory_finished", Bool, self.result_callback, queue_size=100)

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.distance = Distance2goal(gridsize=gridsize, resolution=resolution)

        self.laser = Laser2density(gridsize=gridsize, resolution=resolution)
 
        # self.traj_pred = TrajPred(gridsize=gridsize, resolution=resolution)
        self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        self.traj_feature = [[0.0] for i in range(gridsize[0] * gridsize[1])]

        self.goal = goal

        self.gridsize = gridsize

        self.resolution = resolution

        self.dis_thrd = 0.1

        self.goal_stamped = PoseStamped()
        self.goal_stamped.pose.position.x = self.goal[0]
        self.goal_stamped.pose.position.y = self.goal[1]
        self.goal_stamped.pose.position.z = 0
        self.goal_stamped.header.frame_id = "/map"

        self.nn_r = DeepIRLFC(self.NUM_FEATURE, 0.01, 3, 3)

        # print("before load weight")

        self.nn_r.load_weights()


        # self.traj_pred.session

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]


    def odom_callback(self,data):

        self.robot_pose = np.array([data.pose.pose.position.x, data.pose.pose.position.y])

    def get_feature(self, distance,laser,goal):
        distance_feature = distance.get_feature_matrix(self.goal_stamped)

        while(distance_feature == [0 for i in range(self.gridsize[0] * self.gridsize[1])]):
            distance_feature = distance.get_feature_matrix(self.goal_stamped)

        localcost_feature = laser.temp_result
        # print(self.distance_feature[0], self.localcost_feature[0])
        # traj_feature, _ = self.TrajPred.get_feature_matrix()
        current_feature = np.array([distance_feature[i] + localcost_feature[i] + self.traj_feature[i] for i in range(len(distance_feature))])
        return current_feature

    def get_reward_policy(self, feat_map, gridsize, gamma=0.9, act_rand=0):
        rmap_gt = np.ones([gridsize[0], gridsize[1]])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - act_rand)
        P_a = gw.get_transition_mat()
        reward, policy = get_irl_reward_policy(self.nn_r,feat_map.T, P_a,gamma)

        return reward, policy


    def result_callback(self,data):

        if(data.data):
            self.result = True


    def main(self):

        
        while(np.linalg.norm(self.goal-self.robot_pose, ord=2) > self.dis_thrd and not rospy.is_shutdown()):

            feature = self.get_feature(self.distance, self.laser, self.goal)

            reward, policy = self.get_reward_policy(feature, self.gridsize)

            policy = np.reshape(policy, self.gridsize)

            controller = PathPublisher(policy, self.resolution)

            controller.get_irl_path()       
            controller.irl_path.header.frame_id = 'map'
            controller.irl_path.header.stamp = rospy.Time.now()

            controller.path_pub.publish(controller.irl_path)
            rospy.sleep(0.5)
            # print(self.result)

            # while(self.result == False):
            #     print(self.result)
            #     rospy.sleep(0.01)
            self.result = False


if __name__ == "__main__":
    goal = np.array([13,16])
    gridsize = np.array([3, 3])
    resolution = 0.5
    agent = Agent(goal, gridsize, resolution)
    rospy.sleep(0.1)
    agent.main()


    