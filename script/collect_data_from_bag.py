import os
import yaml 
import argparse
import rospy
import feature_expect
from IPython import embed
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-n', '--new_fol', default= False, type=bool, help='new')
ARGS = PARSER.parse_args()
make_new_folder = ARGS.new_fol
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
    embed()
    __ = os.system("mkdir " + next_folder_name)
    __ = os.system("mkdir " + next_folder_name + "/fm")
    __ = os.system("mkdir " + next_folder_name + "/trajs")

if __name__ == "__main__":
    rospy.init_node("Feature_expect",anonymous=False)
    # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
    resolution = 0.2
    gridsize = (31,31)
    lookahead_dist = gridsize[0]*resolution
    feature = feature_expect.FeatureExpect(resolution= resolution, gridsize=gridsize)
    feature.folder_path = next_folder_name
    feature.lookahead_dist = lookahead_dist
    config_vals = {'resolution': resolution, 'grid_size': [gridsize[0], gridsize[1]], 'lookahead_dist': lookahead_dist, 'notes': "Saving trajs when out of grid not based on time"}
    with open(next_folder_name+"/config.yml", 'w') as file:
        yaml.dump(config_vals, file)
    # while(not feature.initpose_get):
    #     rospy.sleep(0.1)
    # feature.reset_robot()
    rospy.sleep(1)
    # while(not feature.received_goal):
    #     rospy.sleep(0.1)
    # np.savez(fm_file, *feature.feature_maps)
    print("Feature map is ", feature.feature_maps)
    print("Rospy shutdown", rospy.is_shutdown())
    while(not rospy.is_shutdown()):
        feature.get_expect()
        rospy.sleep(0.1)