from turtle import color
import numpy as np
import rospy
import matplotlib.pyplot as plt
from matplotlib import colors, markers
from nav_msgs.msg import Odometry, OccupancyGrid

class static_map():

    def __init__(self. gridsize=(3,3), resolution = 1):
        self.gridsize = gridsize
        self.result = [0]*gridsize[0] * gridsize[1]
        self.resolution = resolution
        self.mapsub = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.map_callback, queue_size=10)
        self.max_dis = 4
        self.temp_result = [[0]]*gridsize[0] * gridsize[1]
        print("Initializing!")
        rospy.sleep(1.0)
        print("Initializing Done!")

    def map_callback(self, msg):
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_h = msg.info.width 
        self.map_w = msg.info.height


    def convert_map_points(self, index):
        row = floor(index/self.map_h)
        
