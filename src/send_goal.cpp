#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <string>
#include <std_msgs/String.h>

/*
Painting1: Position(5.953, -1.314, 0.000), Orientation(0.000, 0.000, -0.230, 0.973) = Angle: -0.464
Painting2: Position(6.232, 2.696, 0.000), Orientation(0.000, 0.000, 0.486, 0.874) = Angle: 1.016
Painting3: Position(-5.799, 6.806, 0.000), Orientation(0.000, 0.000, 0.986, 0.168) = Angle: 2.805
Painting4: Position(0.245, -1.690, 0.000), Orientation(0.000, 0.000, 0.876, -0.483) = Angle: -2.133
*/

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;


void audiocallback(std_msgs::String audio){


  //tell the action client that we want to spin a thread by default
  MoveBaseClient ac("move_base", true);

  //wait for the action server to come up
  while(!ac.waitForServer(ros::Duration(5.0))){
    ROS_INFO("Waiting for the move_base action server to come up");
  }

  move_base_msgs::MoveBaseGoal goal;
  
  std::string location_name;
  // std::cin>>location_name;

  location_name = audio.data;

  if(location_name == "Painting1"){
      goal.target_pose.pose.position.x = 5.953;
      goal.target_pose.pose.position.y = -1.314;
      goal.target_pose.pose.orientation.w = 1.0;
  }
  else if (location_name == "Painting2")
  {
      goal.target_pose.pose.position.x = 6.232;
      goal.target_pose.pose.position.y = 2.696;
      goal.target_pose.pose.orientation.w = 1.0;
  }
  else if (location_name == "Painting3")
  {
      goal.target_pose.pose.position.x = -5.799;
      goal.target_pose.pose.position.y = 6.806;
      goal.target_pose.pose.orientation.w = 1.0;
  }
  else if (location_name == "Painting4")
  {
      goal.target_pose.pose.position.x = 0.245;
      goal.target_pose.pose.position.y = -1.690;
      goal.target_pose.pose.orientation.w = 1.0;
  }
  
  

  //we'll send a goal to the robot to move 1 meter forward

  goal.target_pose.header.frame_id = "map";
  goal.target_pose.header.stamp = ros::Time::now();

//   goal.target_pose.pose.position.x = -1.0;
//   goal.target_pose.pose.position.y = -1.0;
//   goal.target_pose.pose.orientation.w = 1.0;

  ROS_INFO("Sending goal");
  ac.sendGoal(goal);

  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Hooray, the base moved 1 meter forward");
  else
    ROS_INFO("The base failed to move forward 1 meter for some reason");
}



int main(int argc, char **argv)
{
  ros::init(argc, argv, "navigate2goals");
  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("audio", 1000, audiocallback);
  ROS_INFO("Ready to add two ints.");
  ros::spin();

  return 0;
}