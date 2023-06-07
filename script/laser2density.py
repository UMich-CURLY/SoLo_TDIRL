from turtle import color
import numpy as np
from moveit_python import FakeGroupInterface
import rospy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from matplotlib import colors, markers

class Laser2density():

    def __init__(self, gridsize=(3,3), resolution=1):
        # gridsize: a tuple describe the size of grid, default (3,3), always choose gridsize[0] to be odd number.
        self.gridsize = gridsize
        self.result = [0]*gridsize[0] * gridsize[1]
        self.resolution = resolution
        self.lasersub = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=10)
        self.laser = None
        self.max_laser_dis = 25
        self.temp_result = [[1,0,0]]*gridsize[0] * gridsize[1]
        print("Initializing!")
        rospy.sleep(1.0)
        print("Initializing Done!")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    def laser_callback(self, data):
        self.laser = data
        self.get_feature_matrix()
    
    def get_feature_matrix(self):
        if (not self.laser):
            return self.laser.temp_result
        self.result = [0]*self.gridsize[0] * self.gridsize[1]
        for i in range(len(self.laser.ranges)):
            if(self.laser.ranges[i] < self.max_laser_dis):
                laser_angle = self.laser.angle_min + self.laser.angle_increment*i + np.pi/2.0
                laser_x = -self.laser.ranges[i]*np.cos(laser_angle)
                laser_y = self.laser.ranges[i]*np.sin(laser_angle)

                if(abs(laser_x) <= self.gridsize[0]*self.resolution / 2.0 and laser_y <= self.gridsize[0]*self.resolution and laser_y >= 0):
                    self.increase_log_odds(laser_x, laser_y)
                self.scoreRay(laser_x, laser_y)


        # ave_dis = sum(result)/(self.gridsize[0]*self.gridsize[1])

        # std_dev = np.std(result)


        for i in range(len(self.result)):
            #### Cell is unknown
            if(self.result[i] < 4 and self.result[i] > -4):
                self.temp_result[i] =  [0,1,0]
            elif(self.result[i] > 4):
                #### Cell is obstacle
                self.temp_result[i] =  [0,0,1]
            else:
                #### Cell is freee 
                self.temp_result[i] =   [1,0,0]

        self.map_logs = np.reshape(self.result,(self.gridsize[1], self.gridsize[0]))


        # plt.ion() # enable real-time plotting
        # plt.figure(1) # create a plot
        # plt.plot(125,250, markersize=15, marker=10, color="red")
        # plt.imshow(1.0 - 1./(1.+np.exp(self.map_logs)), 'Greys')
        # plt.pause(0.005)
        

        # for i in range(self.gridsize[1]):
        #     print(self.temp_result[i*self.gridsize[1]:(i+1)*self.gridsize[1]])
        # print("------------------------")
        return self.result

    def get_grid_index(self, position):
        # first calculate index in y direction
        y = (self.gridsize[1]*self.resolution - position[1]) // self.resolution

        x = (position[0] + self.gridsize[1]*self.resolution / 2.0) // self.resolution
        return (x,y)

    def increase_log_odds(self, x, y):
        index_x, index_y = self.get_grid_index([x, y])
        if(self.result[int(index_y*self.gridsize[1] + index_x)] <= 10):
            self.result[int(index_y*self.gridsize[1] + index_x)] += 1

    def decrease_log_odds(self, x, y):
        index_x, index_y = x, y
        if(self.result[int(index_y*self.gridsize[1] + index_x)] >= -10):
            self.result[int(index_y*self.gridsize[1] + index_x)] -= 1

    def scoreRay(self,x2, y2):
        x1 = self.gridsize[0] // 2 + 1
        y1 = self.gridsize[1]

        x2, y2 = self.get_grid_index([x2,y2])

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if(x1<x2):
            sx = 1
        else:
            sx = -1

        if(y1<y2):
            sy = 1
        else:
            sy = -1

        err = dx - dy
        x = x1
        y = y1

        while(x != x2 or y != y2):
            
            if(self.inside_grid(x, y)):
                self.decrease_log_odds(x, y)
            # print("Inside while loop!!")
            e2 = 2*err
            if (e2 >= -dy):
                err -= dy
                x += sx
            if (e2 <= dx):
                err += dx
                y += sy
        
    def inside_grid(self, x, y):
        if(x>=0 and x<self.gridsize[0] and y>=0 and y<self.gridsize[1]):
            return True
        else:
            return False


if __name__ == "__main__":
    rospy.init_node("laser2density",anonymous=False)
    # laser2density = Laser2density(gridsize=(25,25), resolution=1)
    laser2density = Laser2density(gridsize=(250,250), resolution=0.01)

    while not rospy.is_shutdown():
        # result = laser2density.result
        # print(result[:3])
        # print(result[3:6])
        # print(result[6:9])
        # print("------------------------")
        rospy.spin()

        