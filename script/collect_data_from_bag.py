import os
import yaml 
import argparse
import rospy
import feature_expect
from IPython import embed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import img_utils
try:
  from StringIO import StringIO
except:
  from io import StringIO
import tensorflow as tf
import numpy as np


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-n', '--new_fol', default= False, type=bool, help='new')
PARSER.add_argument('-s', '--scene', default= "17DRP5sb8fy", type=str, help='scene')

ARGS = PARSER.parse_args()
make_new_folder = ARGS.new_fol
scene = ARGS.scene
max_num = 0
for foldername in os.listdir("../"):
    number_str = "_"
    valid = False
    if (foldername == "config.yml"):
        continue
    if (not foldername[0:7] == "dataset"):
        continue
    
    for m in foldername:
        if m.isdigit():
            valid = True
            max_num = max(max_num,int(m))
        else:
            valid = False
if (make_new_folder):
    max_num = max_num+1
    new_foldername = "../dataset_" + str(max_num)
    __ = os.system("mkdir " + new_foldername)
    __ = os.system("mkdir " + new_foldername+"/demo_0")
    __ = os.system("mkdir " + new_foldername+"/demo_0/fm")
    __ = os.system("mkdir " + new_foldername+"/demo_0/trajs")
    next_folder_name = new_foldername+"/demo_0"
else:
    ### Fix the dataset to add more demos, comment otherwise 
    # max_num = 1
    new_foldername = "../dataset_" + str(max_num)
    max_num = 0
    for foldername in os.listdir(new_foldername):
        number_str = "_"
        valid = False
        if (foldername == "config.yml"):
            continue
        for m in foldername:
            if m.isdigit():
                valid = True
                max_num = max(max_num,int(m))
            else:
                valid = False
    next_folder_name = new_foldername+"/demo_"+str(max_num+1)
    print ("new folder is , continue?", next_folder_name)
    __ = os.system("mkdir " + next_folder_name)
    __ = os.system("mkdir " + next_folder_name + "/fm")
    __ = os.system("mkdir " + next_folder_name + "/trajs")

if __name__ == "__main__":
    rospy.init_node("Feature_expect",anonymous=False)
    # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
    resolution = 0.4
    gridsize = (11,11)
    lookahead_dist = gridsize[0]*resolution
    rospy.sleep(1)
    feature = feature_expect.FeatureExpect(resolution= resolution, gridsize=gridsize)
    feature.folder_path = next_folder_name
    feature.lookahead_dist = lookahead_dist
    feature.sdf_image_path = "/root/catkin_ws/src/ros2lcm/maps/sdf_resolution_"+scene+"_0.025.pgm"
    sampling_time = resolution/1.5
    config_vals = {'resolution': resolution, 'grid_size': [gridsize[0], gridsize[1]], 'scene': scene, 'lookahead_dist': lookahead_dist, 'sampling time': sampling_time, 'notes': "Saving trajs when out of grid not based on time, static scenes only, Using Topomap and sdf"}
    with open(next_folder_name+"/config.yml", 'w') as file:
        yaml.dump(config_vals, file)
    # while(not feature.initpose_get):
    #     rospy.sleep(0.1)
    # feature.reset_robot()
    
    # while(not feature.received_goal):
    #     rospy.sleep(0.1)
    # np.savez(fm_file, *feature.feature_maps)
    print("Feature map is ", feature.feature_maps)
    print("Rospy shutdown", rospy.is_shutdown())
    full_start_time = rospy.Time.now()
    
    while(not rospy.is_shutdown()):
        start_time = rospy.Time.now()
        feature.get_expect()
        end_time = rospy.Time.now()
        time_diff = (end_time - start_time).to_sec()

        if(time_diff >sampling_time and feature.received_goal == True):
            print("Slow down bag file ")
            embed()
        if (not time_diff == 0.0):
            rospy.sleep(sampling_time-time_diff)
        else:
            rospy.sleep(sampling_time)
        full_loop_time = rospy.Time.now()
        if (feature.reached_goal):
            train_summary_writer = tf.summary.FileWriter(next_folder_name+"/logs1")

            train_summary_writer.flush()
            
            for j in range(0,len(feature.feature_maps)):
                traj = feature.all_trajs[j]
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                plt.subplot(2, 2, 1)
                ax1 = img_utils.heatmap2d(np.reshape(feature.feature_maps[j][0], (gridsize[0], gridsize[1])), 'Distance Feature', block=False)
                plt.subplot(2, 2, 2)
                if (feature.feature_maps[j].shape[0] > 1):
                    ax2 = img_utils.heatmap2d(np.reshape(feature.feature_maps[j][1], (gridsize[0], gridsize[1])), 'Obstacle Feature', block=False)
                plt.subplot(2, 2, 3)
                if (feature.feature_maps[j].shape[0] > 1):
                    ax2 = img_utils.heatmap2d(np.reshape(feature.feature_maps[j][4], (gridsize[0], gridsize[1])), 'SDF Feature', block=False)
                # ax3 = img_utils.heatmap2d(np.reshape(rewards, (hight, width)), 'Reward', block=False)
                
                s = StringIO()

                traj_viz = np.zeros(gridsize[0]*gridsize[1])
                maxval = 1.0
                i = 0
                for index in traj:
                    traj_viz[int(index)] = maxval - i*0.075
                    i+=1

                plt.subplot(2, 2, 4)
                ax4 = img_utils.heatmap2d(np.reshape(traj_viz, (gridsize[0], gridsize[1])), 'Observed Traj', block=False)
                plt.savefig(s, format='png')
                plt.close('all')
                img_sum = tf.Summary.Image(encoded_image_string=s.getvalue())
                s.close()
                im_summaries = []
                im_summaries.append(tf.Summary.Value(tag='%s/%d' % ("train", j), image=img_sum))
                summary = tf.Summary(value=im_summaries)
                train_summary_writer.add_summary(summary, j)
            exit(0)