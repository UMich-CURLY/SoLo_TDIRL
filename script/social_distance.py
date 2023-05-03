from turtle import pos, shape
from matplotlib.pyplot import axis
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import tf
from csv import writer
import tf2_ros, tf2_geometry_msgs


class SocialDistance():
    def __init__(self, resolution=1.0, gridsize=(3, 3)):
        self.resolution = resolution
        self.gridsize = gridsize
        self.robot_pose = np.array([0.0, 0.0], dtype=float)
        self.previous_robot_pose = []
        self.robot_distance = 0.0

        self.invade = 0.0
        self.invade_time = 0.0
        self.invade_id = []

        self.people_pose = np.empty((0,3), float)
        self.robot_sub = rospy.Subscriber("sim/robot_pose", PoseStamped, self.robot_pose_callback, queue_size=1000)
        self.people_sub = rospy.Subscriber("sim/agent_poses", PoseArray, self.people_pose_callback, queue_size=1)
        self.marker_distance_pub = rospy.Publisher("/social_distance_markers", MarkerArray, queue_size=1)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf.TransformListener()
        self.alpha = 0.25
        self.beta = 0.2
        self.people_size_offset = 0.2
        
    def transform_pose(self, input_pose, from_frame, to_frame):

        # **Assuming /tf2 topic is being broadcasted

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now()

        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, to_frame, timeout = rospy.Duration(1.0))
            return output_pose_stamped

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("No Transform found?")
            raise
    def robot_pose_callback(self, data):
        pose_new = self.transform_pose(data.pose, 'base_link', 'map')
        self.robot_pose = np.array([pose_new.pose.pose.position.x, pose_new.pose.pose.position.y])

        if(len(self.previous_robot_pose) == 0):
            self.previous_robot_pose = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        else:
            self.robot_distance += np.sqrt((self.robot_pose[0] - self.previous_robot_pose[0])**2 + (self.robot_pose[1] - self.previous_robot_pose[1])**2)
            # print(self.robot_distance)
            self.previous_robot_pose = np.array([data.pose.pose.position.x, data.pose.pose.position.y])

    def people_pose_callback(self,data):

        time_now = rospy.get_time()

        people_pose = np.empty((0,3), float)
        pose_record = []
        
        pose_record += [self.robot_pose[0], self.robot_pose[1]]
        people_id = 0
        for people in data.poses:
            pose_temp = np.array([people.position.x, people.position.y, people_id])
            people_pose = np.append(people_pose, np.array([pose_temp]), axis=0)
            pose_record += [people.position.x, people.position.y]
            people_id+=1
        
        self.people_pose = people_pose
        # print(self.people_pose)

        social_distance_markers = MarkerArray()
        people_id = 0
        for people in data.poses:
            social_distance = self.social_distance(people_id, self.people_pose)

            distance = np.sqrt((people.position.x - self.robot_pose[0])**2 + (people.position.y - self.robot_pose[1])**2)

            if distance < social_distance:
                print("Probably invade!!")
                if self.invade_time == 0.0:
                    self.invade_time = time_now
                    self.invade_id.append(people_id)
                    self.invade += 1.0
                elif time_now - self.invade_time > 1.0:
                    self.invade_id = []
                    self.invade += 1.0
                    self.invade_time = time_now
                    self.invade_id.append(people_id)
                elif time_now - self.invade_time <= 1.0:
                    if (people.id) not in self.invade_id:
                        self.invade += 1.0
                        self.invade_id.append(people_id)

            # if(social_distance > distance):
            #     if(people.track_id not in self.invade_id):
            #         self.invade_time = time_now
            #         self.invade += 1.0

            temp_marker = Marker()
            temp_marker.header.frame_id = "my_map_frame"
            temp_marker.id = people_id
            temp_marker.type = 3
            temp_marker.pose = people
            temp_marker.scale.x = social_distance * 2
            temp_marker.scale.y = social_distance * 2
            temp_marker.scale.z = 0.1
            temp_marker.color.a = 0.5
            temp_marker.color.r = 1.0
            temp_marker.color.g = 0.0
            temp_marker.color.b = 0.0
            social_distance_markers.markers.append(temp_marker)
            people_id +=1
            # print("id: ", people.id, "position: ", [people.pose.position.x, people.pose.position.y])
        self.marker_distance_pub.publish(social_distance_markers)

        # with open('positions_irl.csv', 'a') as f:
        #     writer_object = writer(f)
        #     writer_object.writerow(pose_record)


    def get_density(self, trackingID, people_pose ):
        density = 0
        # people_pose = self.people_pose
        for i in range(len(people_pose)):
            if(i != trackingID):
                distance = np.sqrt((people_pose[trackingID][0]-people_pose[i][0])**2 + (people_pose[trackingID][1]-people_pose[i][1])**2)
                if distance < 2:
                    density += 1
        distance2robot = np.sqrt((people_pose[trackingID][0]-self.robot_pose[0])**2 + (people_pose[trackingID][1]-self.robot_pose[1])**2)
        if distance2robot < 2:
            density += 1
        return density

    def social_distance(self, trackingID, people_pose):
        density = self.get_density(trackingID, people_pose)
        return 1.557 / (density + 1 - 0.8824)**0.215 - 0.967 + self.people_size_offset

    # def min_robot2people(self):
    #     min_distance = 1e6
    #     for people in self.people_pose:
    #         temp_norm = np.sqrt((people[0]-self.robot_pose[0])**2 + (people[1]-self.robot_pose[1])**2)
    #         if(temp_norm < min_distance):
    #             min_distance = temp_norm
    #     return min_distance

    def min_cell2people(self, cell_pose, people_pose):
        min_distance = 1e6
        min_ID = None
        
        for people in people_pose:
            temp_norm = np.sqrt((people[0]-cell_pose[0])**2 + (people[1]-cell_pose[1])**2)
            if(temp_norm < min_distance):
                min_distance = temp_norm
                min_ID = int(people[2])
        return min_distance, min_ID

    def R_concave(self, cell_pose, people_pose):
        dt, id = self.min_cell2people(cell_pose, people_pose)
        d_comfort = self.social_distance(id, people_pose)

        return self.alpha*(dt**self.beta - d_comfort**self.beta) / d_comfort**self.beta

    def get_features(self):
        feature = np.zeros(shape = (self.gridsize[0]*self.gridsize[1], 1))
        people_pose = self.people_pose
        if(people_pose.shape[0] == 0):
            return feature
        # print("people pose length: ",people_pose.shape)
        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                grid_center_x, grid_center_y = self.get_grid_center_position([x , y])
                pose_base = PoseStamped()
                pose_base.pose.position.x = grid_center_x
                pose_base.pose.position.y = grid_center_y
                pose_base.pose.position.z = 0
                pose_base.header.frame_id = "/base_link"


                try:
                    self.listener.waitForTransform("/my_map_frame", "/base_link", rospy.Time(0), rospy.Duration(4.0))
                    pose_in_map = self.listener.transformPose("/my_map_frame", pose_base)
                except:
                    print("Cannot get transform!!")
                    return None
                pose_in_map = np.array([pose_in_map.pose.position.x, pose_in_map.pose.position.y])
                dt, id = self.min_cell2people(pose_in_map, people_pose)
                
                d_comfort = self.social_distance(id, people_pose)
                if(dt < 0):
                    feature[y * self.gridsize[1] + x] = [-0.25]
                elif dt <= d_comfort:
                    feature[y * self.gridsize[1] + x] = [self.R_concave(pose_in_map, people_pose)]
        # print(feature)
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
        if(social_distance.robot_distance != 0):
            print("robot distance is: ", social_distance.robot_distance)
            print("Invade into social distance: ", social_distance.invade / social_distance.robot_distance)

        rospy.sleep(1)
