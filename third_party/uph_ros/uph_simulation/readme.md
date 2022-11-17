### How to use 

Pleae run the following launch file 


```bash
$ cd /path/to/catkin_ws/src
$ git clone https://github.com/s-parsa/uph_simulation.git
$ cd ..
$ rosdep install --from-paths src --ignore-src -r -y
$ catkin_make 
$ source devel/setup.bash
$ roslaunch uph_control uph_control_sim.launch
```

