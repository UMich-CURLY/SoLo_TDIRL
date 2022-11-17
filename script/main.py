import sys
import os

from social_distance import SocialDistance

sys.path.append(os.path.abspath('./irl/'))
from distance2goal import Distance2goal
from laser2density import Laser2density
import numpy as np
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl import *
from utils import *
from controller import PathPublisher
import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from traj_predict import TrajPred
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

class Agent():
    def __init__(self, gridsize,resolution):
        
        rospy.init_node("main")

        self.NUM_FEATURE = 6

        # self.robot_pose = np.array([0, 0])

        self.result = False

        self.result_sub = rospy.Subscriber("/trajectory_finished", Bool, self.result_callback, queue_size=100)

        self.odom_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)

        rospy.wait_for_service('/move_base/NavfnROS/make_plan')
        self.get_plan = rospy.ServiceProxy('/move_base/NavfnROS/make_plan', GetPlan)

        self.goal_id = 0

        self.goal_id_sub = 0

        self.goal_sub = rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.goal_callback)

        self.distance = Distance2goal(gridsize=gridsize, resolution=resolution)

        self.laser = Laser2density(gridsize=gridsize, resolution=resolution)

        self.social_distance = SocialDistance(gridsize=gridsize, resolution=resolution)

        self.controller = PathPublisher(resolution, gridsize)
 
        # self.traj_pred = TrajPred(gridsize=gridsize, resolution=resolution)
        self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        self.reward_pub = rospy.Publisher("reward_map", OccupancyGrid, queue_size=1000)

        self.traj_feature = [[0.0] for i in range(gridsize[0] * gridsize[1])]

        # self.goal = goal

        self.path = []

        self.gridsize = gridsize

        self.resolution = resolution

        self.dis_thrd = 1.1 * self.resolution

        self.success = 0
        self.fail = 0

        # self.goal_stamped = PoseStamped()
        # self.goal_stamped.pose.position.x = self.goal[0]
        # self.goal_stamped.pose.position.y = self.goal[1]
        # self.goal_stamped.pose.position.z = 0
        # self.goal_stamped.header.frame_id = "/map"

        self.nn_r = DeepIRLFC(self.NUM_FEATURE, 0.01, 3, 3)

        # print("before load weight")

        self.nn_r.load_weights()

        # self.traj_pred.session
        print("Init Done!!!!")


    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]


    def amcl_callback(self,data):
        self.robot_pose = np.array([data.pose.pose.position.x, data.pose.pose.position.y])

    # def path_callback(self, data):
    #     if len(data.poses) != 0:
    #         current_waypoint = data.poses[0]
    #         goal = data.poses[-1]

    #         for waypoint in data.poses:
    #             if waypoint != goal and self.get_waypointdistance(current_waypoint, waypoint) < self.resolution * self.gridsize[1]:
    #                 # current_waypoint = waypoint
    #                 # self.path.append(waypoint)
    #                 continue
    #             else:
    #                 current_waypoint = waypoint
    #                 self.path.append(waypoint)
    #     else:
    #         self.path = []
        
    def goal_callback(self, data):

        reached_goal = False
        start_time = rospy.get_time()
        while(not reached_goal):
            self.goal_id_sub = data.header.seq
            # print(data.header.frame_id)
            path_srv = GetPlan()
            path_srv.start = self.nparray2posestamped(self.robot_pose)
            path_srv.goal = data.goal.target_pose
            path_srv.tolerance = 0.1
            path_response = self.get_plan(path_srv.start, path_srv.goal, path_srv.tolerance)
            path = path_response.plan

            if len(path.poses) != 0:
                current_waypoint = path.poses[0]
                goal = path.poses[-1]

                for waypoint in path.poses:
                    if waypoint != goal and self.get_waypointdistance(current_waypoint, waypoint) < self.resolution * self.gridsize[1]:
                        # current_waypoint = waypoint
                        # self.path.append(waypoint)
                        continue
                    else:
                        current_waypoint = waypoint
                        self.path.append(waypoint)
                self.main()
                self.path = []

                distance2goal = np.sqrt( (self.robot_pose[0] - data.goal.target_pose.pose.position.x)**2 + (self.robot_pose[1] - data.goal.target_pose.pose.position.y)**2 )
                if(distance2goal <= self.dis_thrd * 1.5):
                    print("Reached the goal!!!")
                    reached_goal = True
                    self.success += 1.0
                else:
                    print("Failed!!!")
                    reached_goal = False
                    self.fail += 1.0
            else:
                print("path.pose == 0!!!!!")
                self.path = []
            
        end_time = rospy.get_time()

        print("Time spent is: ", end_time - start_time)
        print("Sucessful rate: ", self.success/(self.success + self.fail))


        if(self.social_distance.robot_distance != 0):
            print("robot distance is: ", self.social_distance.robot_distance)
            print("Invade into social distance: ", self.social_distance.invade / self.social_distance.robot_distance)

    def multiplegoal_planner(self, goallist):
        for goal_pose in goallist:
            reached_goal = False
            while(not reached_goal):
                goal_posestamped = self.nparray2posestamped(goal_pose)
                path_srv = GetPlan()
                path_srv.start = self.nparray2posestamped(self.robot_pose)
                path_srv.goal = goal_posestamped
                path_srv.tolerance = 0.1
                path_response = self.get_plan(path_srv.start, path_srv.goal, path_srv.tolerance)
                path = path_response.plan

                if len(path.poses) != 0:
                    current_waypoint = path.poses[0]
                    goal = path.poses[-1]

                    for waypoint in path.poses:
                        if waypoint != goal and self.get_waypointdistance(current_waypoint, waypoint) < self.resolution * self.gridsize[1]:
                            # current_waypoint = waypoint
                            # self.path.append(waypoint)
                            continue
                        else:
                            current_waypoint = waypoint
                            self.path.append(waypoint)
                    self.main()
                    self.path = []
                else:
                    print("path.pose == 0!!!!!")
                    self.path = []

                distance2goal = np.sqrt( (self.robot_pose[0] - goal_pose[0])**2 + (self.robot_pose[1] - goal_pose[1])**2 )
                if(distance2goal <= self.dis_thrd * 1.5):
                    print("Reached the goal!!!")
                    reached_goal = True
                    self.success += 1.0
                else:
                    print("Failed!!!")
                    reached_goal = False
                    self.fail += 1.0
        
        print("Sucessful rate: ", self.success/(self.success + self.fail))


    def get_waypointdistance(self, a, b):
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y

        return np.sqrt(dx**2 + dy**2)

    def get_feature(self, distance,laser,goal):

        distance_feature = distance.get_feature_matrix(self.nparray2posestamped(goal))

        while(distance_feature == [0 for i in range(self.gridsize[0] * self.gridsize[1])]):
            distance_feature = distance.get_feature_matrix(self.nparray2posestamped(goal))

        localcost_feature = laser.temp_result

        social_distance_feature = np.ndarray.tolist(self.social_distance.get_features())
        # print(self.distance_feature[0], self.localcost_feature[0])
        # traj_feature, _ = self.TrajPred.get_feature_matrix()
        current_feature = np.array([distance_feature[i] + localcost_feature[i] + self.traj_feature[i] + social_distance_feature[i] for i in range(len(distance_feature))])
        return current_feature

    def get_reward_policy(self, feat_map, gridsize, gamma=0.9, act_rand=0):
        rmap_gt = np.ones([gridsize[0], gridsize[1]])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - act_rand)
        P_a = gw.get_transition_mat()
        reward, policy = get_irl_reward_policy(self.nn_r, feat_map.T, P_a,gamma)

        return reward, policy


    def result_callback(self,data):
        if(data.data):
            self.result = True

    def posestamped2nparray(self, pose):
        return np.array([pose.pose.position.x, pose.pose.position.y])

    def nparray2posestamped(self, pose_array):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = pose_array[0]
        pose.pose.position.y = pose_array[1]
        return pose

    def main(self):

        path_exec = self.path

        # print(path_exec)

        for waypoint in path_exec:

            self.goal = self.posestamped2nparray(waypoint)
            # print("Goes to the next goal")

            while(np.linalg.norm(self.goal-self.robot_pose, ord=2) > self.dis_thrd and not rospy.is_shutdown()):
                
                feature = self.get_feature(self.distance, self.laser, self.goal)

                # print(feature)

                reward, policy = self.get_reward_policy(feature, self.gridsize)

                reward_map = OccupancyGrid()

                reward_map.header.stamp = rospy.Time.now()
                reward_map.header.frame_id = "base_link"
                reward_map.info.resolution = self.resolution
                reward_map.info.width = self.gridsize[0]
                reward_map.info.height = self.gridsize[1]
                reward_map.info.origin.position.x = 0
                reward_map.info.origin.position.y = - (reward_map.info.width / 2.0) * reward_map.info.resolution
                reward_map.data = [int(cell*100) for cell in normalize(reward)]

                self.reward_pub.publish(reward_map)

                policy = np.reshape(policy, self.gridsize)

                self.controller.get_irl_path(policy)       
                self.controller.irl_path.header.frame_id = 'map'
                self.controller.irl_path.header.stamp = rospy.Time.now()

                if(self.controller.error):
                    print("Controller cannot get a valid path!!!")
                    break

                # print("length of controller is ",len(controller.irl_path.poses))

                self.controller.path_pub.publish(self.controller.irl_path)
                self.controller.irl_path = Path()
                rospy.sleep(0.5)
                # print(self.result)
                # print("Inside the while loop")
                # while(self.result == False):
                #     print(self.result)
                #     rospy.sleep(0.01)
                self.result = False
                # print("Inside the while loop")
        # if(self.social_distance.robot_distance != 0):
        #     print("Invade into social distance: ", self.social_distance.invade / self.social_distance.robot_distance)
        return True

    def test(self, goal):

        self.goal = goal
        # print("Goes to the next goal")

        # while(np.linalg.norm(self.goal-self.robot_pose, ord=2) > self.dis_thrd and not rospy.is_shutdown()):
        # while(not rospy.is_shutdown()):  
        feature = self.get_feature(self.distance, self.laser, self.goal)

        # print(feature)

        reward, policy = self.get_reward_policy(feature, self.gridsize)

        reward_map = OccupancyGrid()

        reward_map.header.stamp = rospy.Time.now()
        reward_map.header.frame_id = "base_link"
        reward_map.info.resolution = self.resolution
        reward_map.info.width = self.gridsize[0]
        reward_map.info.height = self.gridsize[1]
        reward_map.info.origin.position.x = 0
        reward_map.info.origin.position.y = - (reward_map.info.width / 2.0) * reward_map.info.resolution
        reward_map.data = [int(cell*100) for cell in normalize(reward)]

        self.reward_pub.publish(reward_map)

        policy = np.reshape(policy, self.gridsize)

        self.controller.get_irl_path(policy)       
        self.controller.irl_path.header.frame_id = 'map'
        self.controller.irl_path.header.stamp = rospy.Time.now()

        if(self.controller.error):
            # print("Controller cannot get a valid path!!!")
            self.result = False
            return False

        # print("length of controller is ",len(controller.irl_path.poses))

        self.controller.path_pub.publish(self.controller.irl_path)
        self.controller.irl_path = Path()
    #     rospy.sleep(0.5)
    #         # print(self.result)
    #         # print("Inside the while loop")
    #         # while(self.result == False):
    #         #     print(self.result)
    #         #     rospy.sleep(0.01)
    #         # self.result = True
    #         # print("Inside the while loop")
    # # if(self.social_distance.robot_distance != 0):
    # #     print("Invade into social distance: ", self.social_distance.invade / self.social_distance.robot_distance)
    #     self.result = True

    #     if(self.social_distance.robot_distance != 0):
    #         print("robot distance is: ", self.social_distance.robot_distance)
    #         print("Invade into social distance: ", self.social_distance.invade / self.social_distance.robot_distance)

        return self.result




if __name__ == "__main__":
    goal1 = np.array([0,4])
    goal2 = np.array([0,-4])
    gridsize = np.array([3, 3])
    resolution = 0.6
    agent = Agent(gridsize, resolution)
    success = 0
    # fail = 0
    while(not rospy.is_shutdown()):
        agent.test(goal1)
        rospy.sleep(0.5)
    # while(not rospy.is_shutdown()):
    #     # First Goal
    #     start = input("Input any key when you are ready: ")
    #     begin_time = rospy.get_time()
    #     agent.test(goal1)
    #     end_time = rospy.get_time()
    #     print("Time is: ", end_time - begin_time)

    #     if(agent.social_distance.robot_distance != 0):
    #         print("robot distance is: ", agent.social_distance.robot_distance)
    #         print("Invade into social distance: ", agent.social_distance.invade / agent.social_distance.robot_distance)

    #     # Second Goal

    #     start = input("Input any key when you are ready: ")
    #     begin_time = rospy.get_time()
    #     agent.test(goal2)
    #     end_time = rospy.get_time()
    #     print("Time is: ", end_time - begin_time)

    #     if(agent.social_distance.robot_distance != 0):
    #         print("robot distance is: ", agent.social_distance.robot_distance)
    #         print("Invade into social distance: ", agent.social_distance.invade / agent.social_distance.robot_distance)

    # goallist = np.array([
    #                      [14, 9.73],
    #                      [14, 19.13],
    #                      [14, 26.4], 
    #                      [14, 19.13],
    #                      [14, 9.73],
    #                      [14, 2.4]])

    # for i in range(10):
    #     agent.multiplegoal_planner(goallist)
    # rospy.spin()