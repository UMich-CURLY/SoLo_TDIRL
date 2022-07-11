from turtle import shape

from matplotlib.pyplot import axis
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from pedsim_msgs.msg import TrackedPersons
import numpy as np
import tf

class SocialDistance():
    def __init__(self, resolution=1.0, gridsize=(3, 3)):
        self.resolution = resolution
        self.gridsize = gridsize
        self.robot_pose = np.array([0,0])
        self.people_pose = np.empty((0,3), float)
        self.robot_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.robot_pose_callback, queue_size=1000)
        self.people_sub = rospy.Subscriber("/pedsim_visualizer/tracked_persons", TrackedPersons, self.people_pose_callback, queue_size=1000)
        self.listener = tf.TransformListener()
        self.alpha = 0.25
        self.beta = 0.2
        
        
    def robot_pose_callback(self, data):
        self.robot_pose[0] = data.pose.pose.position.x
        self.robot_pose[1] = data.pose.pose.position.y

    def people_pose_callback(self,data):
        self.people_pose = np.empty((0,3), float)
        for people in data.tracks:
            pose_temp = np.array([people.pose.pose.position.x, people.pose.pose.position.y, people.track_id - 1])
            self.people_pose = np.append(self.people_pose, np.array([pose_temp]), axis=0)

    def get_density(self, trackingID):
        density = 0
        people_pose = self.people_pose
        for i in range(len(people_pose)):
            if(i != trackingID):
                distance = np.sqrt((people_pose[trackingID][0]-people_pose[i][0])**2 + (people_pose[trackingID][1]-people_pose[i][1])**2)
                if distance < 2:
                    density += 1
        return density

    def social_distance(self, trackingID):
        density = self.get_density(trackingID)
        return 1.557 / (density + 1 - 0.8824)**0.215 - 0.967

    # def min_robot2people(self):
    #     min_distance = 1e6
    #     for people in self.people_pose:
    #         temp_norm = np.sqrt((people[0]-self.robot_pose[0])**2 + (people[1]-self.robot_pose[1])**2)
    #         if(temp_norm < min_distance):
    #             min_distance = temp_norm
    #     return min_distance

    def min_cell2people(self, cell_pose):
        min_distance = 1e6
        min_ID = None
        for people in self.people_pose:
            temp_norm = np.sqrt((people[0]-cell_pose[0])**2 + (people[1]-cell_pose[1])**2)
            if(temp_norm < min_distance):
                min_distance = temp_norm
                min_ID = int(people[2])
        return min_distance, min_ID

    def R_concave(self, cell_pose):
        dt, id = self.min_cell2people(cell_pose)
        d_comfort = self.social_distance(id)

        return self.alpha*(dt**self.beta - d_comfort**self.beta) / d_comfort**self.beta

    def get_features(self):
        feature = np.zeros(shape = (self.gridsize[0]*self.gridsize[1], 1))

        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                grid_center_x, grid_center_y = self.get_grid_center_position([x , y])
                pose_base = PoseStamped()
                pose_base.pose.position.x = grid_center_x
                pose_base.pose.position.y = grid_center_y
                pose_base.pose.position.z = 0
                pose_base.header.frame_id = "/base_link"


                try:
                    self.listener.waitForTransform("/map", "/base_link", rospy.Time(0), rospy.Duration(4.0))
                    pose_in_map = self.listener.transformPose("/map", pose_base)
                except:
                    print("Cannot get transform!!")
                    return None

                pose_in_map = np.array([pose_in_map.pose.position.x, pose_in_map.pose.position.y])
                dt, id = self.min_cell2people(pose_in_map)
                d_comfort = self.social_distance(id)
                if(dt < 0):
                    feature[y * self.gridsize[1] + x] = [-0.25]
                elif dt <= d_comfort:
                    feature[y * self.gridsize[1] + x] = [self.R_concave(pose_in_map)]
        print(feature)
        return feature

    def get_grid_center_position(self, index):
        center_x = self.resolution / 2.0 + self.resolution * index[0] - self.resolution*(self.gridsize[0] / 2.0)
        center_y = self.resolution * (self.gridsize[0] - index[1] - 1) + self.resolution / 2.0

        center_x, center_y = center_y, -center_x
        return (center_x, center_y)


if __name__ == "__main__":
    rospy.init_node("social_distance")
    social_distance = SocialDistance()
    while(not rospy.is_shutdown()):
        social_distance.get_features()
        rospy.sleep(1)

