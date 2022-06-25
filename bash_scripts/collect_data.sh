roslaunch fetch_irl launch_dynamic.launch &
cd ../scripts/ &
python feature_expect.py &
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
