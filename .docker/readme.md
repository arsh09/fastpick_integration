### Base Image: 

https://hub.docker.com/r/einstein25/pytorch-ros-gpu


GUI exanple (if you have NVidia-GPU)

```bash

$ xhost + 
$ nvidia-docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device=/dev/video1:/dev/video1 --name test1 einstein25/pytorch-ros-gpu

```

### build targets 

1) Install MoveIt 
2) Install RealSense 
3) Install Realsense-ros 
4) Install Libfranka 
5) Install Franka-ros
6) Install 