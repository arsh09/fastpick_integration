cmake_minimum_required(VERSION 3.0.2)
project(fastpick_msgs)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  actionlib_msgs
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  StrawberryObject.msg
  StrawberryObjects.msg
)

  
add_action_files(
  FILES
  StrawberryPick.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs  
)


catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs actionlib_msgs
)