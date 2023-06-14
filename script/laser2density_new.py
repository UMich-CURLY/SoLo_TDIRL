from cmath import exp
from numpy import average
import tf
from math import sqrt, pow
import rospy
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
from IPython import embed
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from visualization_msgs.msg import MarkerArray, Marker

class laser2density():

    def __init__(self, gridsize=(3,3), resolution=1):
        # gridsize: a tuple describe the size of grid, default (3,3)
        self.gridsize = gridsize
        self.resolution = resolution
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer) 
        self.mapsub = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.map_callback, queue_size=10)
        self.map_update_sub = rospy.Subscriber("/move_base/local_costmap/costmap_updates", OccupancyGridUpdate, self.map_update_callback, queue_size=10)
        self._pub_markers = rospy.Publisher("~grid_points", MarkerArray, queue_size = 1)
        self.temp_result = [[1,0,0]]*gridsize[0] * gridsize[1]
        print("Initializing!")
        rospy.sleep(1.0)
        print("Initializing Done!")

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
        
    def map_callback(self, msg):
        self.origin_x, self.origin_y = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_resolution = msg.info.resolution
        self.map_h, self.map_w = msg.info.height, msg.info.width
        # grid_x, grid_y = self.get_grid_index()

    def convert_local_map_to_position(self, index):
        position_x = self.origin_x + (index // self.map_h) * self.map_resolution + self.map_resolution/2
        position_y = self.origin_y + (index % self.map_w)* self.map_resolution + self.map_resolution/2
        return (position_x, position_y)

    def map_update_callback(self, msg):
        markers = MarkerArray()
        for i in range(len(msg.data)):
            x,y  = self.convert_local_map_to_position(i)
            pose_now = Pose()
            pose_now.position.x = x
            pose_now.position.y = y
            pose_in_base = self.transform_pose(pose_now, "map", "base_link")
            position_x, position_y = pose_in_base.pose.position.x, pose_in_base.pose.position.y
            grid_x, grid_y = self.get_grid_center_position([position_x, position_y])
            temp_marker = Marker()
            temp_marker.header.frame_id = "base_link"
            temp_marker.id = i
            temp_marker.type = 1
            temp_marker.pose.position.x = grid_x
            temp_marker.pose.position.y = grid_y
            temp_marker.scale.x = self.map_resolution
            temp_marker.scale.y = self.map_resolution
            temp_marker.scale.z = 0.1
            temp_marker.color.a = 1.0
            temp_marker.color.r = 0.0
            temp_marker.color.g = 0.0
            temp_marker.color.b = 0.0
            markers.markers.append(temp_marker)
        self._pub_markers.publish(markers)

    def get_grid_index(self, position):
        # first calculate index in y direction
        y = (self.gridsize[1]*self.resolution - position[1]) // self.resolution

        x = (position[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
        return (x,y)    
        
    def get_grid_center_position(self, index):
        center_x = self.resolution / 2.0 + self.resolution * index[0] - self.resolution*(self.gridsize[0] / 2.0)
        center_y = self.resolution * (self.gridsize[0] - index[1] - 1) + self.resolution / 2.0

        center_x, center_y = center_y, -center_x
        return (center_x, center_y)
if __name__ == "__main__":
    rospy.init_node("laser2density",anonymous=False)
    # laser2density = Laser2density(gridsize=(25,25), resolution=1)
    laser2density = laser2density(gridsize=(3,3), resolution=0.5)

    while not rospy.is_shutdown():
        # result = laser2density.result
        # print(result[:3])
        # print(result[3:6])
        # print(result[6:9])
        # print("------------------------")
        rospy.spin()