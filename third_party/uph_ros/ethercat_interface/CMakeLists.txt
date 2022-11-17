cmake_minimum_required(VERSION 2.8.3)
project(ethercat_interface)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  ethercat_soem
  std_msgs
  message_generation)

add_message_files(
  FILES
  Cmd.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
## Declare a catkin package
catkin_package(CATKIN_DEPENDS std_msgs message_runtime)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(ethercat_interface src/ethercat_interface.cpp)
target_link_libraries(ethercat_interface ${catkin_LIBRARIES})


