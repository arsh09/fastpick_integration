cmake_minimum_required(VERSION 3.0.2)
project(fastpick_pick)
 
find_package(catkin REQUIRED
    actionlib
    fastpick_msgs
)
  
catkin_package( 
    CATKIN_DEPENDS
    fastpick_msgs
)
 
include_directories( 
)
    