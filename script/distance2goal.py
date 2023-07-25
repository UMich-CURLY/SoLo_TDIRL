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
from nav_msgs.msg import Odometry, OccupancyGrid, Path
import utils
from visualization_msgs.msg import MarkerArray, Marker

class Distance2goal():

    def __init__(self, gridsize=(3,3), resolution=1):
        # gridsize: a tuple describe the size of grid, default (3,3)
        self.gridsize = gridsize
        self.resolution = resolution
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer) 
        self.dist_pub = rospy.Publisher("dist_map", OccupancyGrid, queue_size=0)
        self._pub_markers = rospy.Publisher("~/dist/grid_points", MarkerArray, queue_size = 1)


    def transform_pose(self, input_pose, from_frame, to_frame):

        # **Assuming /tf2 topic is being broadcasted

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now() - rospy.Duration(1)
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
        markers = MarkerArray()
        goal_in_base = self.transform_pose(goal.pose, frame_id, "base_frame")
        if (not goal_in_base):
            return result
        # print("Goal in base link ", goal_in_base)  
        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                grid_center_x, grid_center_y = self.get_grid_center_position([x , y])
                
                distance = sqrt(pow(grid_center_x - goal_in_base.pose.position.x, 2) + pow(grid_center_y - goal_in_base.pose.position.y, 2))
                # distance = exp(abs(distance))
                result[x * self.gridsize[0] + y] = distance
                # print("Grid centers and coords, dist are: ", [x,y] , [grid_center_x, grid_center_y], distance)
                temp_marker = Marker()
                temp_marker.header.frame_id = "base_frame"
                temp_marker.id = self.gridsize[0]*x+y
                temp_marker.type = 1
                temp_marker.pose.position.x = grid_center_x 
                temp_marker.pose.position.y = grid_center_y
                temp_marker.scale.x = self.resolution/2
                temp_marker.scale.y = self.resolution/2
                temp_marker.scale.z = 0.1
                temp_marker.color.a = 1.0
                temp_marker.color.r = abs(3.0 - distance)/3.0 
                temp_marker.color.g = 0.0
                temp_marker.color.b = 0.0
                markers.markers.append(temp_marker)
        self._pub_markers.publish(markers)

        #### Normalizing the goal distance #####
        # max_distance = max(result)
        # min_distance = min(result)
        # result = [[(result[i]-min_distance) / (max_distance - min_distance)] for i in range(len(result))]

        ##########################################
        result = [[result[i]] for i in range(len(result))]
        dist_map = OccupancyGrid()
        dist_map.header.stamp = rospy.Time.now()
        dist_map.header.frame_id = "local"
        dist_map.info.resolution = self.resolution
        dist_map.info.width = self.gridsize[0]
        dist_map.info.height = self.gridsize[1]
        dist_map.info.origin.position.x = 5
        dist_map.info.origin.position.y = 0
        dist_map.data = [int(cell*100) for cell in utils.normalize(result)]
        self.dist_pub.publish(dist_map)
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
        origin_x = self.resolution/2
        origin_y = 0.0
        diff_x_index = index[0] - (self.gridsize[0] - 1) 
        diff_y_index = index[1] - ((self.gridsize[1] - 1) / 2)
        center_x = -diff_x_index*self.resolution + origin_x
        center_y = diff_y_index*self.resolution + origin_y
        # center_x = self.resolution / 2.0 + self.resolution * index[0] - self.resolution*(self.gridsize[0] / 2.0)
        # center_y = self.resolution * (self.gridsize[0] - index[1] - 1) + self.resolution / 2.0

        # center_x, center_y = center_x, -center_y
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
        data.header.frame_id = "base_frame"
        result = distance2goal.get_feature_matrix(data)
        print(result[:3])
        print(result[3:6])
        print(result[6:9])
        # print(result[12:16])
        # print(result)
        print("------------------------")





        