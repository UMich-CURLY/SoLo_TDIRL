from itertools import count
import rospy
from cmath import exp
from numpy import average
import tf
from math import sqrt, pow
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np

class PathPublisher():
    def __init__(self, resolution, gridsize):

        self.gridsize = gridsize
        self.resolution = resolution
        self.tf_listener = tf.TransformListener()

        self.irl_path = Path()
        self.irl_path.header.frame_id = 'map'
        self.irl_path.header.stamp = rospy.Time.now()

        self.path_pub = rospy.Publisher("/path", Path, queue_size=100)

        self.error = False
        
        pass

    def get_irl_path(self, policy):
        '''
        Policy be like:
        [['r' 'r' 'r']
        ['r' 'r' 'r']
        ['r' 'r' 'r']]
        '''
        self.policy = policy
        self.error = False
        direct = {'r':np.array([0, 1]), 'l':np.array([0, -1]), 'u':np.array([-1, 0]), 'd':np.array([1, 0]), 's':np.array([0, 0])}
        
        current_pose = np.array([2,1])

        goal_map = 0

        while not goal_map:
            goal_map = self.get_grid_center_position(current_pose)

        self.irl_path.poses.append(goal_map)

        first_act = self.policy[current_pose[0]][current_pose[1]]
        count = 0

        while(count < self.gridsize[0] * self.gridsize[1] - 1):
            
            next_goal = current_pose + direct[first_act]
            # print(next_goal)
            if next_goal[0] >= self.gridsize[0] or next_goal[1] >= self.gridsize[1] or \
                 next_goal[0] < 0 or next_goal[1] < 0:
                 break

            goal_map = 0

            while not goal_map:
                goal_map = self.get_grid_center_position(next_goal)
                # print(goal_map)

            # if goal_map == 0:
            #     continue

            self.irl_path.poses.append(goal_map)

            current_pose = next_goal

            first_act = self.policy[current_pose[0]][current_pose[1]]
            # Open loop maybe closed loop later
            count += 1

            

        if count == self.gridsize[0] * self.gridsize[1] - 1:
            self.error = True
        
        # self.path_pub.publish(self.irl_path)

    def get_grid_center_position(self, index):

        center_x = self.resolution / 2.0 + self.resolution * index[1] - self.resolution*(self.gridsize[0] / 2.0)
        center_y = self.resolution * (self.gridsize[0] - index[0] - 1) + self.resolution / 2.0

        center_x, center_y = center_y, -center_x

        goal_in_base = PoseStamped()
        # goal_in_base.header.stamp = rospy.Time.now()
        goal_in_base.pose.position.x = center_x
        goal_in_base.pose.position.y = center_y
        goal_in_base.pose.position.z = 0
        # goal_in_base.pose.orientation.w = 1

        goal_in_base.header.frame_id = "/base_link"


        # self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(4.0))
        # goal_in_map = self.tf_listener.transformPose("map", goal_in_base)


        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(4.0))
            goal_in_map = self.tf_listener.transformPose("map", goal_in_base)
        except:
            print("Do not get transform!!!!!!!!!")
            return 0

        return goal_in_map

if __name__ == "__main__": 

    rospy.init_node("publish_irl_path")

    policy = [['r', 'r', 'r'],
            ['r', 'r', 'u'],
            ['u', 'u', 'u']]

    controller = PathPublisher(policy=policy, resolution=0.5)

    # controller.irl_path.header.frame_id = '/map'
    controller.get_irl_path()       
    controller.irl_path.header.frame_id = 'map'
    controller.irl_path.header.stamp = rospy.Time.now()

    # while(not rospy.is_shutdown()):

    controller.path_pub.publish(controller.irl_path)
        # rospy.sleep(0.1)