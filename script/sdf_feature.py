from turtle import color
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from matplotlib import colors, markers
import cv2 as cv
from geometry_msgs.msg import PoseStamped, PoseArray
import tf
from IPython import embed
import yaml
from visualization_msgs.msg import MarkerArray, Marker

class sdf_feature():

    def __init__(self, gridsize=(3,3), resolution=1, image_path = None):
        # gridsize: a tuple describe the size of grid, default (3,3), always choose gridsize[0] to be odd number.
        self.gridsize = gridsize
        self.result = [0]*gridsize[0] * gridsize[1]
        self.resolution = resolution
        self.temp_result = [0]*gridsize[0] * gridsize[1]
        self.listener = tf.TransformListener()
        img = cv.imread(image_path)
        img_v = cv.flip(img, 0)
        self.sdf_dist = cv.cvtColor(img_v,cv.COLOR_BGR2GRAY)
        cv.imwrite("grayscale.png", self.sdf_dist)
        self._pub = rospy.Publisher("sdf_points", PoseArray, queue_size=0)
        with open(image_path[:-3]+"yaml", 'r') as file:
            map_config = yaml.safe_load(file)
        self.sdf_map_origin = map_config['origin'][0:2]
        self.sdf_map_resolution = map_config['resolution']
        self._pub_markers = rospy.Publisher("sdf_value", MarkerArray, queue_size = 1)

    
    def laser_callback(self, data):
        self.laser = data
    
    def get_feature_matrix(self):
        result = [0 for i in range(self.gridsize[0] * self.gridsize[1])]
        try:
            self.listener.waitForTransform("/base_frame", "/map", rospy.Time(0), rospy.Duration(4.0))
            pose = PoseStamped()
            pose.header.frame_id = "/base_frame"
            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 0
            pose.pose.orientation.x = 0 
            pose.pose.orientation.y = 0 
            pose.pose.orientation.z = 0 
            pose.pose.orientation.w = 1 
            markers = MarkerArray()
            base_in_map = self.listener.transformPose("/map", pose)
            pose_array = PoseArray()
            pose_array.header.frame_id = "map"
            pose_array.header.stamp = rospy.Time.now()
            for x in range(self.gridsize[0]):
                for y in range(self.gridsize[1]):
                    grid_center_x, grid_center_y = self.get_grid_center_position([x , y])
                    pose = PoseStamped()
                    pose.header.frame_id = "/base_frame"
                    pose.pose.position.z = 0
                    pose.pose.orientation.x = 0 
                    pose.pose.orientation.y = 0 
                    pose.pose.orientation.z = 0 
                    pose.pose.orientation.w = 1 
                    pose.pose.position.x = grid_center_x
                    pose.pose.position.y = grid_center_y
                    pose_in_map = self.listener.transformPose("/map", pose)
                    sdf_center_x, sdf_center_y = self.get_sdf_map_index([pose_in_map.pose.position.x, pose_in_map.pose.position.y])
                    sdf_value = self.sdf_dist[int(sdf_center_y), int(sdf_center_x)]
                    pose_array.poses.append(pose_in_map.pose)
                    temp_marker = Marker()
                    temp_marker.header.frame_id = "base_frame"
                    temp_marker.id = self.gridsize[0]*x+y
                    temp_marker.type = 1
                    temp_marker.pose.position.x = grid_center_x 
                    temp_marker.pose.position.y = grid_center_y
                    temp_marker.scale.x = self.resolution
                    temp_marker.scale.y = self.resolution
                    temp_marker.scale.z = 0.1
                    temp_marker.color.a = 1.0
                    temp_marker.color.r = sdf_value
                    temp_marker.color.g = sdf_value
                    temp_marker.color.b = sdf_value
                    markers.markers.append(temp_marker)
            self._pub.publish(pose_array)
            self._pub_markers.publish(markers)
        except:
            print("FML")
        return self.result

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

    def get_sdf_map_index(self, position):
        [x,y] = [position[0]-self.sdf_map_origin[0], position[1]-self.sdf_map_origin[1]]
        return [x/self.sdf_map_resolution, y/self.sdf_map_resolution]
        
    def inside_grid(self, x, y):
        if(x>=0 and x<self.gridsize[0] and y>=0 and y<self.gridsize[1]):
            return True
        else:
            return False


if __name__ == "__main__":
    rospy.init_node("sdf_feature",anonymous=False)
    # laser2density = Laser2density(gridsize=(25,25), resolution=1)
    sdf_feature = sdf_feature(gridsize=(30,30), resolution=0.05, image_path = "/root/catkin_ws/src/SoLo_TDIRL/maps/maps/sdf_resolution_Vvot9Ly1tCj_0.025.pgm")

    while not rospy.is_shutdown():
        sdf_feature.get_feature_matrix()
        # result = laser2density.result
        # print(result[:3])
        # print(result[3:6])
        # print(result[6:9])
        # print("------------------------")

        rospy.sleep(0.1)

        