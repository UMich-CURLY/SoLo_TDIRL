import sys
import os

from social_distance import SocialDistance

sys.path.append(os.path.abspath('/root/catkin_ws/src/SoLo_TDIRL/script/irl/'))
sys.path.append("/root/miniconda3/envs/habitat/lib/python3.7/site-packages")

from distance2goal import Distance2goal
from laser2density_new import laser2density
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
import tf2_ros, tf2_geometry_msgs
from IPython import embed
import tf
import img_utils
from visualization_msgs.msg import Marker, MarkerArray
import tensorflow 
from StringIO import StringIO
import matplotlib.pyplot as plt
# 

LOOKAHEAD_DIST = 1.5

class Agent():
    def __init__(self, gridsize,resolution):
        
        rospy.init_node("main")

        self.NUM_FEATURE = 2
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer) 
        # self.robot_pose = np.array([0, 0])
        self.listener1 = tf.TransformListener()
        self.use_tf1 = True
        self.result = False

        self.result_sub = rospy.Subscriber("/trajectory_finished", Bool, self.result_callback, queue_size=100)

        self.odom_sub = rospy.Subscriber('/sim/robot_pose', PoseStamped, self.pose_callback, queue_size = 100)
        self._pub_waypoint = rospy.Publisher("~waypoint", Marker, queue_size = 1)
        rospy.wait_for_service('/move_base/NavfnROS/make_plan')
        self.get_plan = rospy.ServiceProxy('/move_base/NavfnROS/make_plan', GetPlan)

        self.goal_id = 0

        self.goal_id_sub = 0

        self.goal_sub = rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.goal_callback, queue_size = 100)

        self.distance = Distance2goal(gridsize=gridsize, resolution=resolution)

        self.laser = laser2density(gridsize=gridsize, resolution=resolution)

        self.social_distance = SocialDistance(gridsize=gridsize, resolution=resolution)

        self.controller = PathPublisher(resolution, gridsize)
 
        # self.traj_pred = TrajPred(gridsize=gridsize, resolution=resolution)
        self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        self.reward_pub = rospy.Publisher("reward_map", OccupancyGrid, queue_size=1000)

        self.traj_feature = [[0.0] for i in range(gridsize[0] * gridsize[1])]

        self.goal = None

        self.path = []

        self.gridsize = gridsize

        self.resolution = resolution

        self.dis_thrd = 1.1 * self.resolution

        self.success = 0
        self.fail = 0
        self.counter = 0
        # self.goal_stamped = PoseStamped()
        # self.goal_stamped.pose.position.x = self.goal[0]
        # self.goal_stamped.pose.position.y = self.goal[1]
        # self.goal_stamped.pose.position.z = 0
        # self.goal_stamped.header.frame_id = "/map"

        self.nn_r = DeepIRLFC(self.NUM_FEATURE, 0.01, 30, 30)

        # print("before load weight")

        self.nn_r.load_weights()
        self.received_goal = False
        self.sess = tensorflow.Session()
        self.writer = tensorflow.summary.FileWriter("../logs")
        self.step = tensorflow.Variable(0, dtype=tensorflow.int64)
        self.step_update = self.step.assign_add(1)
        self.writer_flush = self.writer.flush()
        self.sess.run ([self.step.initializer])
        init = tensorflow.initialize_all_variables()
        self.sess.run(init)
        # self.traj_pred.session
        print("Init Done!!!!")

    def transform_pose(self,input_pose, from_frame, to_frame):

        # **Assuming /tf2 topic is being broadcasted
        

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now() - rospy.Duration(1.0)
        print("Time in main is ", pose_stamped.header.stamp)
        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, to_frame, timeout = rospy.Duration(1))
            print (output_pose_stamped)
            return output_pose_stamped

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("No Transform found?")
            pass

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]
        print(traj_feature)

    def pose_callback(self,data):
        # print(data.header)
        # a = self.listener.waitForTransform("/base_link", "/map", rospy.Time.now(), rospy.Duration(10))
        # # if (a):
        
        # pose_new = self.listener.transformPose("/map", data)
        
        pose_new = self.transform_pose(data.pose, 'base_link', 'map')
        self.robot_pose = np.array([pose_new.pose.position.x, pose_new.pose.position.y])
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
        self.received_goal = True
        print("Received goal?  ", self.received_goal)
        reached_goal = False
        # start_time = rospy.get_time()
        self.target_goal_pose = data.goal.target_pose
        path_srv = GetPlan()
        path_srv.start = self.nparray2posestamped(self.robot_pose)
        path_srv.goal = data.goal.target_pose
        path_srv.tolerance = 0.1
        path_response = self.get_plan(path_srv.start, path_srv.goal, path_srv.tolerance)
        path = path_response.plan

        if len(path.poses) != 0:
            self.current_waypoint = path.poses[0]
            self.goal = path.poses[-1]
        else:
            print ("No ptah found")
        # while(not reached_goal):
        #     self.goal_id_sub = data.header.seq
        #     # print(data.header.frame_id)
        #     path_srv = GetPlan()
        #     path_srv.start = self.nparray2posestamped(self.robot_pose)
        #     path_srv.goal = data.goal.target_pose
        #     path_srv.tolerance = 0.1
        #     path_response = self.get_plan(path_srv.start, path_srv.goal, path_srv.tolerance)
        #     path = path_response.plan

        #     if len(path.poses) != 0:
        #         current_waypoint = path.poses[0]
        #         goal = path.poses[-1]

        #         for waypoint in path.poses:
        #             if waypoint != goal and self.get_waypointdistance(current_waypoint, waypoint) < self.resolution * self.gridsize[1]:
        #                 # current_waypoint = waypoint
        #                 # self.path.append(waypoint)
        #                 continue
        #             else:
        #                 current_waypoint = waypoint
        #                 self.path.append(waypoint)
        #         self.main()
        #         self.path = []

        #         distance2goal = np.sqrt( (self.robot_pose[0] - data.goal.target_pose.pose.position.x)**2 + (self.robot_pose[1] - data.goal.target_pose.pose.position.y)**2 )
        #         if(distance2goal <= self.dis_thrd * 1.5):
        #             print("Reached the goal!!!")
        #             reached_goal = True
        #             self.success += 1.0
        #         else:
        #             print("Failed!!!")
        #             reached_goal = False
        #             self.fail += 1.0
        #     else:
        #         print("path.pose == 0!!!!!")
        #         self.path = []
            
        # end_time = rospy.get_time()

        # print("Time spent is: ", end_time - start_time)
        # print("Sucessful rate: ", self.success/(self.success + self.fail))


        # if(self.social_distance.robot_distance != 0):
        #     print("robot distance is: ", self.social_distance.robot_distance)
        #     print("Invade into social distance: ", self.social_distance.invade / self.social_distance.robot_distance)

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
                    # print("path.pose == 0!!!!!")
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

    def get_feature(self, distance, goal, laser = None):

        distance_feature = distance.get_feature_matrix(goal)
        print ("Distance feature is and goal is ", distance_feature, goal)
        # while(distance_feature == [0 for i in range(self.gridsize[0] * self.gridsize[1])]):
        #     distance_feature = distance.get_feature_matrix(goal)
        if(laser):
            localcost_feature = laser.get_feature_matrix()
            print("Local cost feature is ", localcost_feature)
            # social_distance_feature = np.ndarray.tolist(self.social_distance.get_features())
            # print("social_distance_feature is ", social_distance_feature)
            # print("traj_feature is ", self.traj_feature)
            # print("Distance_feature is ", distance_feature)
            # print(self.distance_feature[0], self.localcost_feature[0])
            # traj_feature, _ = self.TrajPred.get_feature_matrix()
            # print("Current feature is", [distance_feature[i] + localcost_feature[i] + self.traj_feature[i] + social_distance_feature[i] for i in range(len(distance_feature))])
            current_feature = np.array([distance_feature[i] + localcost_feature[i] + self.traj_feature[i] + [0.0] for i in range(len(distance_feature))])

        else:
            print("No laser right?")
            social_distance_feature = np.ndarray.tolist(self.social_distance.get_features())
            # print(self.distance_feature[0], self.localcost_feature[0])
            # traj_feature, _ = self.TrajPred.get_feature_matrix()
            current_feature = np.array([distance_feature[i] + [0.0] + [0.0] +[0.0] + self.traj_feature[i] + social_distance_feature[i] for i in range(len(distance_feature))])
            print("Shape of traj_feature is ", len(self.traj_feature))

            print("Shape of distance_feature is ", len(distance_feature))
            print("Shape of social_distance_feature is ", len(social_distance_feature))
        return current_feature

    def get_reward_policy(self, feat_map, gridsize, gamma=0.9, act_rand=0):
        rmap_gt = np.ones([gridsize[0], gridsize[1]])
        gw = gridworld.GridWorld(rmap_gt, {}, 1 - act_rand)
        P_a = gw.get_transition_mat()
        feat_map_new = np.array(feat_map[:,0:2])
        print(feat_map_new.shape)
        reward, policy = get_irl_reward_policy(self.nn_r, feat_map_new.T, P_a,gamma)

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
        if(not self.received_goal or not self.goal):
            return None
        if (False):
        # if (np.linalg.norm(self.goal-self.robot_pose, ord=2) > self.dis_thrd):
            print ("Sucess achived")
            return False
        else:
            print("Still Working on it ")
            path_srv = GetPlan()
            path_srv.start = self.nparray2posestamped(self.robot_pose)
            path_srv.goal = self.target_goal_pose
            path_srv.tolerance = 0.1
            path_response = self.get_plan(path_srv.start, path_srv.goal, path_srv.tolerance)
            path = path_response.plan

            # print("Goes to the next goal")
            if len(path.poses) != 0:
                current_waypoint = path.poses[0]
                self.goal = path.poses[-1]
                for waypoint in path.poses:
                    if waypoint != self.goal and self.get_waypointdistance(current_waypoint, waypoint) < self.resolution * self.gridsize[1]:
                        # current_waypoint = waypoint
                        # self.path.append(waypoint)
                        continue
                    elif self.get_waypointdistance(current_waypoint, waypoint) >= LOOKAHEAD_DIST:
                        self.current_waypoint = waypoint
                        break
                        # self.path.append(waypoint)
                    elif waypoint == self.goal:
                        self.current_waypoint = self.goal
                        break
                    else:
                        print("Not sure why ")
                        embed()
            temp_marker = Marker()
            temp_marker.header.frame_id = "map"
            temp_marker.type = 3
            temp_marker.pose.position.x = self.current_waypoint.pose.position.x
            temp_marker.pose.position.y = self.current_waypoint.pose.position.y
            temp_marker.scale.x = 0.2
            temp_marker.scale.y = 0.2
            temp_marker.scale.z = 0.1
            temp_marker.color.a = 1.0
            temp_marker.color.r = 0.0
            temp_marker.color.g = 1.0
            temp_marker.color.b = 0.0
            self._pub_waypoint.publish(temp_marker)
            # while(np.linalg.norm(self.goal-self.robot_pose, ord=2) > self.dis_thrd and not rospy.is_shutdown()):
                
            feature = self.get_feature(self.distance, self.current_waypoint, self.laser)

            print("Feature is", feature)

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

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            plt.subplot(2, 2, 1)
            ax1 = img_utils.heatmap2d(np.reshape(feature[:,0], (self.gridsize[0],self.gridsize[1])), 'Distance Feature', block=False)
            plt.subplot(2, 2, 2)
            ax2 = img_utils.heatmap2d(np.reshape(feature[:,1], (self.gridsize[0],self.gridsize[1])), 'Obstacle Feature', block=False)
            plt.subplot(2, 2, 3)
            ax3 = img_utils.heatmap2d(np.reshape(reward, (self.gridsize[0],self.gridsize[1])), 'Reward', block=False)
            plt.subplot(2, 2, 4)
            # ax3 = img_utils.heatmap2d(np.reshape(policy, (self.gridsize[0],self.gridsize[1])), 'Policy', block=False)
            s = StringIO()
            
            plt.savefig(s, format='png')
            img_sum = tensorflow.Summary.Image(encoded_image_string=s.getvalue())
            im_summaries = []
            im_summaries.append(tensorflow.Summary.Value(tag='%s/%d' % ("main", self.counter), image=img_sum))
            summary = tensorflow.Summary(value=im_summaries)
            self.writer.add_summary(summary, self.counter)
            self.sess.run(self.step_update)
            # self.sess.run(self.writer_flush)
            print ("The reward and pollicy are:", reward, policy)
            self.counter +=1
            self.controller.get_irl_path(policy)       
            self.controller.irl_path.header.frame_id = 'map'
            self.controller.irl_path.header.stamp = rospy.Time.now()

            if(self.controller.error):
                print("Controller cannot get a valid path!!!")
                return False

            # print("length of controller is ",len(controller.irl_path.poses))

            self.controller.path_pub.publish(self.controller.irl_path)
            self.controller.irl_path = Path()
            rospy.sleep(0.1)
            # print(self.result)
            # print("Inside the while loop")
            # while(self.result == False):
            #     print(self.result)
            #     rospy.sleep(0.01)
            self.result = False
            return True
                # print("Inside the while loop")
        # if(self.social_distance.robot_distance != 0):
        #     print("Invade into social distance: ", self.social_distance.invade / self.social_distance.robot_distance)
        


    def test(self, goal):

        self.goal = goal
        # print("Goes to the next goal")

        # while(np.linalg.norm(self.goal-self.robot_pose, ord=2) > self.dis_thrd and not rospy.is_shutdown()):
        # while(not rospy.is_shutdown()):  
        # feature = self.get_feature(self.distance, self.laser, self.goal)
        feature = self.get_feature(self.distance, self.goal, self.laser)
        print(feature, feature.shape)

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
        self.controller.irl_path.header.frame_id = 'my_map_frame'
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


    def log_images(self, tag, images, step):
        """Logs a list of images."""
        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                        height=img.shape[0],
                                        width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                    image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

if __name__ == "__main__":
    goal1 = np.array([0,4])
    goal2 = np.array([0,-4])
    gridsize = np.array([3, 3])
    resolution = 0.5
    agent = Agent(gridsize, resolution)
    success = 0
    
    # fail = 0
    while(not rospy.is_shutdown()):
        agent.main()
        # rospy.sleep(0.5)
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
    agent.writer.flush()