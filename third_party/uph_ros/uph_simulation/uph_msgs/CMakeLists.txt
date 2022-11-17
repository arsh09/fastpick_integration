cmake_minimum_required(VERSION 3.0.2)
project(uph_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DetectedBerry.msg
  DetectedBerries.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  
  sensor_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs 
)
