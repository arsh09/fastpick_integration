## FastPick ROS Package 


The packages in this repository are for FastPick project. These packages are the integration of the different components developed by different team members. 


### Package Description: 

- *fastpick_bringup* - contains the bringup launch file of franka arm, its moveit instance, perception pipeline and camera driver. 

- *fastpick_description* - contains URDF/Xacro that connect Franka Panda arm with a camera and its gripper. The package also contains the hand-in-eye calibration parameters i.e. the TF from *fastpick_grasp_link* to *camera_color_optical_frame*. 

**P.S**: If you feel the need to recalibrate the hand-in-eye configuration, you can use the pre-configured joint values in the *calib/* folder

- *fastpick_msgs* - contains the custom msg type used to publish the results of the ENet berry detection model. 


- *fastpick_percetion* - contains the deep machine learning (ENet) model for berry detection, its inference script and two python nodes. One of them uses camera color image for berry detection and publishes it on a ROS topic whereas the other node subscribe to this detected_berry inference and further post process it using contour detection and ellipse fitting. A number of TF frames with respect to *camera_color_optical_frame* are also published for each berry that can be useful in the pick and place pipeline. The same filtering node re-publishes the color image, camera info, as well as remove detected berry masks from the depth image and publish the depth_berry and depth_background topic which are later used for OctoMap generation.

- *fastpick_pick* - contains a node that takes in the berry frame and try to plan a path from robot EE i.e. *fastpick_grasp_link* to given berry frame. The berry frames are the frames published from *fastpick_perception* pipeline.

- *moveit_calibration* - This is a MoveIt package for hand-in-eye and hand-to-eye calibration. 

- *panda_moveit_config* - contains the Franka Panda MoveIt config. The package is slightly modified from the actual *panda_moveit_config* in the sense that the real panda controller action server were provided to moveit instead of the fake_controllers, the camera collisions were added (and disabled) in the SRDF file, OctoMap was added using the background depth mask published by *fastpick_perception* pipeline 



### How to Run: 

1) Make sure you have a conda environment setup 
2) Create a virtual environment. Its not neccessary but ENet model requires torchvision library. Install the requirement.txt inside *fastpick_perception/requirements.txt*
3) Activate the virtual environment
4) Clone this repo 
5) Run rosdep for neccessary ros packages 
6) Catkin make it 
7) Source the workspace 

```bash
# terminal 1 // control, moveit, camera 
$ roslaunch fastpick_bringup fastpick_bringup.launch
```

```bash
# terminal 2 // perception pipeline
$ roslaunch fastpick_perception fastpick_perception_bringup.launch
```

```bash
# terminal 2 // perception pick & place pipeline
$ roslaunch fastpick_pick fastpick_pick_berry.launch which_berry:=BERRY_FRAME

# BERRY_FRAME: Should be any berry frame that are available in Berry-Detection-Results open-cv window.
```

### ToDo

- The pick and place pipeline should be converted into an action server instead. 
- Options to dynamically add and remove octomap in planning using the said action server message 
- Perform path planning in camera optical frame to avoid confusion or post rotation multiplication for pre-align, pre-grasp, grasp and post-grasp poses. 
- Add scene analyzer action client. 
- Add comments and reformat the code 
- Add a few tests
