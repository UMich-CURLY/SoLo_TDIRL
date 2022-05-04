import numpy as np
import rospy
from sensor_msgs.msg import LaserScan

class Laser2density():

    def __init__(self, gridsize=(3,3), resolution=1):
        # gridsize: a tuple describe the size of grid, default (3,3)
        self.gridsize = gridsize
        self.result = [0]*gridsize[0] * gridsize[1]
        self.resolution = resolution
        self.lasersub = rospy.Subscriber("/base_scan", LaserScan, self.laser_callback, queue_size=1)

    
    def laser_callback(self, data):
        self.result = [0]*self.gridsize[0] * self.gridsize[1]
        for i in range(len(data.ranges)):
            laser_angle = data.angle_min + data.angle_increment*i + np.pi/2.0
            laser_x = data.ranges[i]*np.cos(laser_angle)
            laser_y = data.ranges[i]*np.sin(laser_angle)
            if(abs(laser_x) <= self.gridsize[0]*self.resolution / 2.0 and laser_y <= self.gridsize[0]*self.resolution and laser_y >= 0):
                index_x, index_y = self.get_grid_index([laser_x, laser_y])
                self.result[np.int(self.gridsize[1]*index_y + index_x)] += 1
        print(self.result[:3])
        print(self.result[3:6])
        print(self.result[6:9])
        print("--------------------------------")
        

    def get_grid_index(self, position):
        # first calculate index in y direction
        y = (self.gridsize[1]*self.resolution - position[1]) // self.resolution

        x = (position[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
        return (x,y)

if __name__ == "__main__":
    rospy.init_node("laser2density",anonymous=False)
    laser2density = Laser2density()
    while not rospy.is_shutdown():
        rospy.spin()
    