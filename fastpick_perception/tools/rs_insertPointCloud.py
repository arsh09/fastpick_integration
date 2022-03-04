#!/usr/bin/env python

import glooey
from glooey.containers import VBox
import imgviz
import numpy as np
from numpy.core.shape_base import block
import pyglet
import trimesh
import trimesh.transformations as tf
import trimesh.viewer
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image,CameraInfo
#######################################################
#######################################################
from datetime import datetime
# import pyrealsense2 as rs
import numpy as np
from open3d import *
import open3d as o3d
import pyrealsense2 as rs
# from op3d_tools.rlpt2op3dpt_PlyPcd import *
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
# import Open3D.examples.python.open3d_tutorial as o3dtut
from scipy.spatial.transform import *
import sys
########################################################

import octomap
import os,cv2
output_path = '/home/fpick/FPICK/Rlsense3DPrjTrack/results'
# https://github.com/wkentaro/octomap-python/blob/master/examples/insertPointCloud.py

def pointcloud_from_depth(depth, fx, fy, cx, cy):
    if type(depth) is not None:
        # assert depth.dtype.kind == 'f', 'depth must be float and have meter values'
        assert depth.dtype == 'f', 'depth must be float and have meter values'

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    return pc


def labeled_scene_widget(scene, label):
    vbox = glooey.VBox()
    vbox.add(glooey.Label(text=label, color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene))
    # fig = plt.figure('trimesh.viewer', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    # plt.title('trimesh.viewer')
    # plt.imshow(trimesh.viewer.SceneWidget(scene),cmap='gray')
    # plt.show(block=True)
    # fig = trimesh.viewer.SceneWidget(scene)    
    return vbox



def rs_callback_o3d_example(data):
    # data = imgviz.data.arc2017()
    camera_info = data['camera_info']
    K = np.array(camera_info['K']).reshape(3, 3)
    rgb = data['rgb']
    depth_image = data['depth']

    # Stack both images horizontally
    src2 = np.zeros_like(rgb)
    src2[:,:,0] = depth_image
    src2[:,:,1] = depth_image
    src2[:,:,2] = depth_image
    images = np.hstack((src2,rgb))
    fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Rs Depth & Colors')
    plt.imshow(images,cmap='gray')
    plt.show(block=True)

    # 加载点云，并采样2000个点
    N = 2000
    """    pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
    # 点云归一化
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())
    # 点云着色
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    # 可视化
    o3d.visualization.draw_geometries([pcd])

    # 创建八叉树， 树深为4
    octree = o3d.geometry.Octree(max_depth=4)
    # 从点云中构建八叉树，适当扩展边界0.01m
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    # 可视化
    o3d.visualization.draw_geometries([octree])
    """
    # 从体素网格中构造八叉树
    # 从点云中创建体素网格， 体素大小为0.05m
    pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    # 体素可视化
    o3d.visualization.draw_geometries([voxel_grid])

    # 创建八叉树， 树深为4
    octree = o3d.geometry.Octree(max_depth=4)
    # 从体素网格中构建八叉树
    octree.create_from_voxel_grid(voxel_grid)
    # 可视化
    o3d.visualization.draw_geometries([octree])

    ###########################################
       
    camera_parameters = camera.PinholeCameraParameters()
    camera_parameters.extrinsic = np.array([[1,0,0,1],
                                           [0,1,0,0],
                                           [0,0,1,2],
                                           [0,0,0,1]])
    camera_parameters.intrinsic.set_intrinsics(width=1280, height=720, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

    # Creates Open3D visualizer
    viewer = o3d.visualization.Visualizer()
    # viewer.create_window(window_name = "octree map",visible=False)
    viewer.create_window()
    viewer.add_geometry(pcd)

    # coord = open3d.geometry.TriangleMesh.create_coordinate_frame(1, [0, 0, 0])
    # viewer.add_geometry(coord)

    viewer.run()
    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(camera_parameters)
    depth = viewer.capture_depth_float_buffer()
    """    print("show depth")
    # print(np.asarray(depth))
    plt.figure('numpy image')
    plt.title('octree depth mesh')
    plt.imshow(np.asarray(depth))
    plt.imsave("octree_depth.png", np.asarray(depth), dpi = 1)
    plt.show()"""


    ##############################################
    ############################################## 
class MapManager(object):
    def __init__(self):
        
        self.aligned_frames = None # choose webcam False: choose realsense camera        
        self.depth_frame = None
        self.color_frame = None
        self.depth_image = None
        self.color_image = None 
        # self.depth_image_oct   = 0.0

        self.obj_data = None
        self.ctl_data = None               
        self.camera_info = CameraInfo()
        self.width = 1280
        self.height = 720
        self.fx = 908.594970703125
        self.fy = 908.0234375
        self.ppx = 650.1152954101562
        self.ppy = 361.9811096191406

        self.flag = True

        self.K = np.array([[self.fx, 0, self.ppx],
                [0, self.fy, self.ppy],
                [0, 0, 1]])

        self.K[0, 0] = self.fx
        self.K[1, 1] = self.fy
        self.K[0, 2] = self.ppx
        self.K[1, 2] = self.ppy

    def visualize(self, occupied, empty, K, width, height, rgb, pcd, mask, resolution, aabb):
        window = pyglet.window.Window(width=int(width * 0.5 * 3), height=int(height * 0.75))
        ##############################################################
        # testing on text window drawing - dom
        ##############################################################
        # creating alabel
        scale_w =0.5  # 1 - one image to display, 0.5 - two image
        label = pyglet.text.Label('strawb',
                                font_name ='Times New Roman',
                                color=(255,0, 0, 255),
                                font_size = 36,
                                x = scale_w*window.width//2, y = window.height//2,
                                anchor_x ='center', anchor_y ='center')
        
        # drawing label
        @window.event
        def on_draw():
            # window.clear()
            label.draw()
        ##############################################################
        ##############################################################
        # continuity
        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.on_close()

        gui = glooey.Gui(window)
        hbox = glooey.HBox()
        hbox.set_padding(5)

        camera = trimesh.scene.Camera(
            resolution=(width, height), focal=(K[0, 0], K[1, 1])
        )
        camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.1)

        # initial camera pose
        # camera_transform = np.array(
        #     [
        #         [0.73256052, -0.28776419, 0.6168848, 0.66972396],
        #         [-0.26470017, -0.95534823, -0.13131483, -0.12390466],
        #         [0.62712751, -0.06709345, -0.77602162, -0.28781298],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        # )

        camera_transform = np.array(
            [
                [1.0, -0.0, 0.0, 0.0],
                [-0.0, -1, -0.0, -0.0],
                [0.0, -0.0, -1.0, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

        aabb_min, aabb_max = aabb
        bbox = trimesh.path.creation.box_outline(
            aabb_max - aabb_min,
            tf.translation_matrix((aabb_min + aabb_max) / 2),
        )

        geom = trimesh.PointCloud(vertices=pcd[mask], colors=rgb[mask])
        scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
        scene.camera_transform = camera_transform
        VBox= labeled_scene_widget(scene, label='pointcloud')
        hbox.add(VBox)  
        
        # gui.add(hbox)  
        # viewplet = pyglet.app.run()

        # return
       
        # color_scene = np.asanyarray(VBox,dtype=np.uint8)       
        # num = 100
        # figname='strw_octomap_depth_rgb_'+str(num)+'.png'   
        # nampepath = os.path.join(output_path, figname) 
        # # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(nampepath,color_scene)
        # fig = plt.figure('scene', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('scene')
        # plt.imshow(hbox,cmap='gray')
        # plt.show(block=True)


        #######################################################
        geom = trimesh.voxel.ops.multibox(
            occupied, pitch=resolution, colors=[1.0, 0, 0, 0.5]
        )
        scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
        scene.camera_transform = camera_transform
        hbox.add(labeled_scene_widget(scene, label='occupied'))

        gui.add(hbox)  
        # @window.event
        # def on_draw():
        #     # window.clear()
        #     label.draw()
        #     window.draw()
        # viewplet = pyglet.app.event_loop.sleep(0.5)
        viewplet = pyglet.app.run()
        return
        ##########################################################
        geom = trimesh.voxel.ops.multibox(
            empty, pitch=resolution, colors=[0.5, 0.5, 0.5, 0.5]
        )
        scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
        scene.camera_transform = camera_transform
        hbox.add(labeled_scene_widget(scene, label='empty'))

        gui.add(hbox)  
        viewplet = pyglet.app.run()
        # fig = plt.figure('scene', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('scene')
        # plt.imshow(viewplet,cmap='gray')
        # plt.show(block=True)
    def rs_callback_octomap(self):
        # data = imgviz.data.arc2017()
        # camera_info = data['camera_info']
        # K = np.array(camera_info['K']).reshape(3, 3)
        # rgb = data['rgb']
        # depth_image = data['depth']


        """
        self.K = [[self.fx, 0, self.ppx],
                [0, self.fy, self.ppy],
                [0, 0, 1]]"""

        """
        # Stack both images horizontally
        src2 = np.zeros_like(self.color_image)
        src2[:,:,0] = self.depth_image
        src2[:,:,1] = self.depth_image
        src2[:,:,2] = self.depth_image
        images = np.hstack((src2,self.color_image))
        fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.title('Rs Depth & Colors')
        plt.imshow(images,cmap='gray')
        plt.show(block=True)"""

        # pcd = pointcloud_from_depth(
        #     self.depth_image, fx=self.fx, fy=self.fy, cx=self.ppx, cy=self.ppy)
        pcd = pointcloud_from_depth(
            self.depth_image, fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2])
        # length of array
        # n = pcd.size
        # pcd = pcd.reshape(720,1280,3)
        nonnan = ~np.isnan(pcd).any(axis=2)
        mask = np.less(pcd[:, :, 2], 2)

        resolution = 0.05
        octree = octomap.OcTree(resolution)
        octree.insertPointCloud(
            pointcloud=pcd[nonnan],
            origin=np.array([0, 0, 0], dtype=float),
            maxrange=2,
        )
        occupied, empty = octree.extractPointCloud()

        # fig = plt.figure('octree', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('occupied & empty')
        # plt.imshow(octree,cmap='gray')
        # plt.show(block=True)           

        aabb_min = octree.getMetricMin()
        aabb_max = octree.getMetricMax()

        self.visualize(
            occupied=occupied,
            empty=empty,
            K=self.K,
            width=self.width,
            height=self.height,
            rgb=self.color_image,
            pcd=pcd,
            mask=mask,
            resolution=resolution,
            aabb=(aabb_min, aabb_max),
        )
        return occupied, empty 

    def rs_callback_o3d_rltime(self):
        # testing the tutorial
        # rs_callback_o3d_example(data)

        # data = imgviz.data.arc2017()
        # camera_info = data['camera_info']
        # K = np.array(camera_info['K']).reshape(3, 3)
        # rgb = data['rgb']
        # depth_image = data['depth']

        """    # Stack both images horizontally
        src2 = np.zeros_like(rgb)
        src2[:,:,0] = depth_image
        src2[:,:,1] = depth_image
        src2[:,:,2] = depth_image
        images = np.hstack((src2,rgb))
        fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.title('Rs Depth & Colors')
        plt.imshow(images,cmap='gray')
        plt.show(block=True)"""

        # depth_frame =data['depth_frame']
        # color_frame = data['color_frame']
        # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        ###########################################
        
        camera_parameters = camera.PinholeCameraParameters()
        camera_parameters.extrinsic = np.array([[1,0,0,1],
                                    [0,1,0,0],
                                    [0,0,1,2],
                                    [0,0,0,1]],dtype='float64')

        
        # camera_parameters.extrinsic = np.array(depth_to_color_extrin)

        # camera_parameters.extrinsic=np.array([[0.9981 ,-0.0160041,  0.0593907 ,   -1.0298],
        #                     [0.0123974 ,  0.998087 , 0.0605806  , -1.59618],
        #                     [-0.060248, -0.0597236 ,  0.996391  ,  1.32826],
        #                     [       0.  ,        0. ,       0.     ,     1.]],dtype='float64')



        camera_parameters.intrinsic.set_intrinsics(width=self.width, height=self.height, fx=self.fx, fy=self.fy, cx=self.ppx, cy=self.ppy)
        #camera_parameters.intrinsic.set_intrinsics(depth_intrin.width, depth_intrin.height, depth_intrin.fx, depth_intrin.fy, depth_intrin.ppx, depth_intrin.ppy)


        """    
        # convert points using realsens built in 
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())  """

        # depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
        # pcd.colors = open3d.Vector3dVector(colors / 255.0)
        img_depth = o3d.geometry.Image (self.depth_image)
        # img_depth = o3d.geometry.Image (bg_removed)
        img_color = o3d.geometry.Image (self.color_image)

        #method 1. image直接生成点云 :this is one channel image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth)            

        # Generate a Point Cloud from RGBD image
        # If there is a RGBD image PointCloud.create_from_rgbd_imageusing the You can create a Point Cloud in one shot.
        pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd0.transform ([[ 1 , 0 , 0 , 0 ], [ 0 , - 1 , 0 , 0 ], [ 0 , 0 , - 1 , 0 ], [ 0 , 0 , 0 , 1 ]])

        ##########################################
        ##########################################
        # method2 Odometry pcd：读两幅RGBD图，计算刚体变换
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth (img_color, img_depth, convert_rgb_to_intensity = False )            
        # depth_intrin = frm_profile.as_video_stream_profile().get_intrinsics ()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic (self.width, self.height, self.fx, self.fy, self.ppx, self.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image (rgbd, camera_parameters.intrinsic)
        pcd.transform ([[ 1 , 0 , 0 , 0 ], [ 0 , -1 , 0 , 0 ], [ 0 , 0 , -1 , 0 ], [ 0 , 0 , 0 , 1 ]])            


        """# convert into grid for octree - variant method
        # 点云归一化
        pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
        # 点云着色
        N=2000
        pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))    
        
        # 可视化
        o3d.visualization.draw_geometries([pcd])

        # 从点云中创建体素网格， 体素大小为0.05m
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
        # 体素可视化
        o3d.visualization.draw_geometries([voxel_grid])

        # 创建八叉树， 树深为4

        octree = o3d.geometry.Octree(max_depth=4)
        # 从体素网格中构建八叉树
        octree.create_from_voxel_grid(voxel_grid)
        # 可视化
        o3d.visualization.draw_geometries([octree])
        """


        # camera_parameters = camera.PinholeCameraParameters()
        # camera_parameters.extrinsic = np.array([[1,0,0,1],
        #                                        [0,1,0,0],
        #                                        [0,0,1,2],
        #                                        [0,0,0,1]])
        # camera_parameters.intrinsic.set_intrinsics(width=1280, height=720, fx=1000, fy=1000, cx=959.5, cy=539.5)
        
        # 创建八叉树， 树深为4
        octree = o3d.geometry.Octree(4)  
        # 从点云中构建八叉树，适当扩展边界0.01m
        # octree.convert_from_point_cloud(pcd, size_expand=0.01)
        octree.convert_from_point_cloud(pcd)
        # 可视化
        # o3d.visualization.draw_geometries([octree])

        print("show o3d octree")
        # Creates Open3D visualizer
        viewer = o3d.visualization.Visualizer()
        # viewer.create_window(window_name = "octree map",visible=True)
        # viewer.create_window(width=camera_info['width'], height=camera_info['height'], left=50, right=50)
        viewer.create_window(width=int(self.width), height=int(self.height))
        # viewer.create_window()
        viewer.add_geometry(octree)    
        
        # coord = open3d.geometry.TriangleMesh.create_coordinate_frame(1, [0, 0, 0])
        # viewer.add_geometry(coord)
        viewer.run()
        control = viewer.get_view_control()
        control.convert_from_pinhole_camera_parameters(camera_parameters)
        depth = viewer.capture_depth_float_buffer()
        # cv2.imshow('depth',np.asarray(depth))
        # cv2.waitKey() 
        # print(viewer)
        # print(viewer.get_window_name())
        # print(viewer.get_view_control())
        # plt.title('octree depth mesh')
        # plt.imshow(np.asarray(depth,dtype=np.unit8),cmap='gray')
        # # cv2.imwrite("octree_depth_1.png", np.asarray(depth))
        # plt.show(block=False)           
        # plt.close()

        return depth,depth




        #############################################################
        #############################################################

        """
        # 点云归一化
        # pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
                # center=pcd.get_center())
        # 点云着色
        # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
        # 可视化
        # o3d.visualization.draw_geometries([pcd])
        pcd = pointcloud_from_depth(
                np.ndarray(depth_image,type=np.float32), fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
            )
        # 创建八叉树， 树深为4
        octree = o3d.geometry.Octree(max_depth=4)
        # 从点云中构建八叉树，适当扩展边界0.01m
        octree.convert_from_point_cloud(pcd, size_expand=0.01)
        # 可视化
        o3d.visualization.draw_geometries([octree])

        # o3d.visualization.draw_geometries(pcd)
        # length of array
        # n = pcd.size
        # pcd = pcd.reshape(720,1280,3)
        nonnan = ~np.isnan(pcd).any(axis=2)
        mask = np.less(pcd[:, :, 2], 2)

        resolution = 0.01
        octree = octomap.OcTree(resolution)
        octree.insertPointCloud(
            pointcloud=pcd[nonnan],
            origin=np.array([0, 0, 0], dtype=float),
            maxrange=2,
        )
        occupied, empty = octree.extractPointCloud()
        # pcd = open3d.cpu.pybind.geometry.Geometry(pcd)
        # o3d.visualization.draw_geometries([occupied])
        occupied= np.asanyarray(occupied,np.uint8)
        # o3d.visualization.draw_geometries([occupied])
        # fig = plt.figure('octree', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.title('occupied & empty')
        # plt.imshow(octree,cmap='gray')
        # plt.show(block=True)
            

        aabb_min = octree.getMetricMin()
        aabb_max = octree.getMetricMax()

        visualize(
            occupied=occupied,
            empty=empty,
            K=K,
            width=camera_info['width'],
            height=camera_info['height'],
            rgb=rgb,
            pcd=pcd,
            mask=mask,
            resolution=resolution,
            aabb=(aabb_min, aabb_max),
        )"""
        # return occupied, empty 
    

    

# using vtex to form pcd and very slow............   
def rs_callback_o3d_vtx(data):
    # testing the tutorial
    # rs_callback_o3d_example(data)

    # data = imgviz.data.arc2017()
    camera_info = data['camera_info']
    K = np.array(camera_info['K']).reshape(3, 3)
    rgb = data['rgb']
    depth_image = data['depth']

    # Stack both images horizontally
    src2 = np.zeros_like(rgb)
    src2[:,:,0] = depth_image
    src2[:,:,1] = depth_image
    src2[:,:,2] = depth_image
    images = np.hstack((src2,rgb))
    fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Rs Depth & Colors')
    plt.imshow(images,cmap='gray')
    plt.show(block=True)


    ###########################################
       
    camera_parameters = camera.PinholeCameraParameters()
    camera_parameters.extrinsic = np.array([[1,0,0,1],
                                           [0,1,0,0],
                                           [0,0,1,2],
                                           [0,0,0,1]])
    camera_parameters.intrinsic.set_intrinsics(width=1280, height=720, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
   


    depth_frame =data['depth_frame']
    color_frame = data['color_frame']

    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())  

    pcd = o3d.geometry.PointCloud()
    # pcd.points =  np.asanyarray(o3d.utility.Vector3dVector(vtx))
    N=2000
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))

    # pcd.colors = open3d.Vector3dVector(colors / 255.0)

    
    vtx = np.asanyarray(points.get_vertices())  

    print(vtx.shape)
    npy_vtx = np.zeros((len(vtx), 3), float)

    for i in range(len(vtx)):
        npy_vtx[i][0] = np.float(vtx[i][0])
        npy_vtx[i][1] = np.float(vtx[i][1])
        npy_vtx[i][2] = -np.float(vtx[i][2])
    print("npy_vtx",npy_vtx.shape,npy_vtx)  
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy_vtx)
    
    

    # 点云归一化
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())

    # 从点云中创建体素网格， 体素大小为0.05m
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    # 体素可视化
    o3d.visualization.draw_geometries([voxel_grid])

    # 创建八叉树， 树深为4
    octree = o3d.geometry.Octree(max_depth=4)
    # 从体素网格中构建八叉树
    octree.create_from_voxel_grid(voxel_grid)
    # 可视化
    o3d.visualization.draw_geometries([octree])
    
    # 点云着色
    N=2000
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    # 可视化
    o3d.visualization.draw_geometries([pcd])

    # camera_parameters = camera.PinholeCameraParameters()
    # camera_parameters.extrinsic = np.array([[1,0,0,1],
    #                                        [0,1,0,0],
    #                                        [0,0,1,2],
    #                                        [0,0,0,1]])
    # camera_parameters.intrinsic.set_intrinsics(width=1280, height=720, fx=1000, fy=1000, cx=959.5, cy=539.5)
    
    # 创建八叉树， 树深为4
    octree = o3d.geometry.Octree(4)  

    # 从点云中构建八叉树，适当扩展边界0.01m
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    # 可视化
    o3d.visualization.draw_geometries([octree])
    # Creates Open3D visualizer
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name = "octree map",visible=True)
    viewer.add_geometry(pcd)

    # coord = open3d.geometry.TriangleMesh.create_coordinate_frame(1, [0, 0, 0])
    # viewer.add_geometry(coord)

    viewer.run()
    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(camera_parameters)
    depth = viewer.capture_depth_float_buffer()
    # print("show depth")
    # print(np.asarray(depth))
    plt.figure('numpy image')
    plt.title('octree depth mesh')
    plt.imshow(np.asarray(depth))
    plt.imsave("octree_depth.png", np.asarray(depth), dpi = 1)
    plt.show()





    #############################################################
    #############################################################


    # 点云归一化
    # pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            # center=pcd.get_center())
    # 点云着色
    # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    # 可视化
    # o3d.visualization.draw_geometries([pcd])
    pcd = pointcloud_from_depth(
            depth_image, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
    # 创建八叉树， 树深为4
    octree = o3d.geometry.Octree(max_depth=4)
    # 从点云中构建八叉树，适当扩展边界0.01m
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    # 可视化
    o3d.visualization.draw_geometries([octree])

    # o3d.visualization.draw_geometries(pcd)
    # length of array
    # n = pcd.size
    # pcd = pcd.reshape(720,1280,3)
    nonnan = ~np.isnan(pcd).any(axis=2)
    mask = np.less(pcd[:, :, 2], 2)

    resolution = 0.01
    octree = octomap.OcTree(resolution)
    octree.insertPointCloud(
        pointcloud=pcd[nonnan],
        origin=np.array([0, 0, 0], dtype=float),
        maxrange=2,
    )
    occupied, empty = octree.extractPointCloud()
    # pcd = open3d.cpu.pybind.geometry.Geometry(pcd)
    # o3d.visualization.draw_geometries([occupied])
    occupied= np.asanyarray(occupied,np.uint8)
    # o3d.visualization.draw_geometries([occupied])
    # fig = plt.figure('octree', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    # plt.title('occupied & empty')
    # plt.imshow(octree,cmap='gray')
    # plt.show(block=True)
        

    aabb_min = octree.getMetricMin()
    aabb_max = octree.getMetricMax()

    visualize(
        occupied=occupied,
        empty=empty,
        K=K,
        width=camera_info['width'],
        height=camera_info['height'],
        rgb=rgb,
        pcd=pcd,
        mask=mask,
        resolution=resolution,
        aabb=(aabb_min, aabb_max),
    )
    return occupied, empty 
    

def rs_callback(data):
    # data = imgviz.data.arc2017()
    camera_info = data['camera_info']
    K = np.array(camera_info['K']).reshape(3, 3)
    rgb = data['rgb']
    depth_image = data['depth']

    # Stack both images horizontally
    src2 = np.zeros_like(rgb)
    src2[:,:,0] = depth_image
    src2[:,:,1] = depth_image
    src2[:,:,2] = depth_image
    images = np.hstack((src2,rgb))
    fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Rs Depth & Colors')
    plt.imshow(images,cmap='gray')
    plt.show(block=True)

    pcd = pointcloud_from_depth(
        depth_image, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    # length of array
    # n = pcd.size
    # pcd = pcd.reshape(720,1280,3)
    nonnan = ~np.isnan(pcd).any(axis=2)
    mask = np.less(pcd[:, :, 2], 2)

    resolution = 0.01
    octree = octomap.OcTree(resolution)
    octree.insertPointCloud(
        pointcloud=pcd[nonnan],
        origin=np.array([0, 0, 0], dtype=float),
        maxrange=2,
    )
    occupied, empty = octree.extractPointCloud()

    # fig = plt.figure('octree', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    # plt.title('occupied & empty')
    # plt.imshow(octree,cmap='gray')
    # plt.show(block=True)
        

    aabb_min = octree.getMetricMin()
    aabb_max = octree.getMetricMax()

    visualize(
        occupied=occupied,
        empty=empty,
        K=K,
        width=camera_info['width'],
        height=camera_info['height'],
        rgb=rgb,
        pcd=pcd,
        mask=mask,
        resolution=resolution,
        aabb=(aabb_min, aabb_max),
    )
    return occupied, empty 

def callback(data):
    # data = imgviz.data.arc2017()
    camera_info = data['camera_info']
    K = np.array(camera_info['K']).reshape(3, 3)
    rgb = data['rgb']
    depth_image = data['depth']


    # Stack both images horizontally
    src2 = np.zeros_like(rgb)
    src2[:,:,0] = depth_image
    src2[:,:,1] = depth_image
    src2[:,:,2] = depth_image
    images = np.hstack((src2,rgb))
    fig = plt.figure('rs_callback', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Rs Depth & Colors')
    plt.imshow(images,cmap='gray')
    plt.show(block=True)


    pcd = pointcloud_from_depth(
        depth_image, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )

    nonnan = ~np.isnan(pcd).any(axis=2)
    mask = np.less(pcd[:, :, 2], 2)

    resolution = 0.01
    octree = octomap.OcTree(resolution)
    octree.insertPointCloud(
        pointcloud=pcd[nonnan],
        origin=np.array([0, 0, 0], dtype=float),
        maxrange=2,
    )
    occupied, empty = octree.extractPointCloud()

    aabb_min = octree.getMetricMin()
    aabb_max = octree.getMetricMax()

    visualize(
        occupied=occupied,
        empty=empty,
        K=K,
        width=camera_info['width'],
        height=camera_info['height'],
        rgb=rgb,
        pcd=pcd,
        mask=mask,
        resolution=resolution,
        aabb=(aabb_min, aabb_max),
    )
    return occupied, empty 

def main():
    data = imgviz.data.arc2017()
    camera_info = data['camera_info']
    K = np.array(camera_info['K']).reshape(3, 3)
    rgb = data['rgb']

    fig = plt.figure('Octomap', figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Depth & Colors')
    # res = np.hstack((color_image,frame)) #stacking images side-by-side
    # display with meshing purpose
    # res_mesh = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(depth_colormap_aligned, cv2.COLOR_BGR2RGB)))      
    # res_mesh = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap_aligned))      

    # display with tracking purose
    # res_track = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    # figname='strw_mesh_'+str(num)+'.png'
    # nampepath = os.path.join(output_path, figname) 
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(nampepath,cv2.cvtColor(res_mesh, cv2.COLOR_BGR2RGB))
    # plt.imshow(rgb,cmap='gray')

    depth_image = data['depth']
    # Stack both images horizontally
    src2 = np.zeros_like(rgb)
    src2[:,:,0] = depth_image
    src2[:,:,1] = depth_image
    src2[:,:,2] = depth_image
    # src2 = cv2.resize(depth_image, color_image.shape[1::-1])
    images = np.hstack((src2, rgb))


    
    plt.imshow(images,cmap='gray')
    plt.show(block=True)
    # plt.show()
    # num = num+1
    fxx=K[0, 0]
    fyy=K[1, 1]
    cxx=K[0, 2]
    cyy=K[1, 2]

    """pcd = pointcloud_from_depth(
        data['depth'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )"""

    pcd = pointcloud_from_depth(
        depth_image, fx=fxx, fy=fyy, cx=cxx, cy=cyy)

    nonnan = ~np.isnan(pcd).any(axis=2)
    mask = np.less(pcd[:, :, 2], 2)

    resolution = 0.01
    octree = octomap.OcTree(resolution)
    octree.insertPointCloud(
        pointcloud=pcd[nonnan],
        origin=np.array([0, 0, 0], dtype=float),
        maxrange=2,
    )
    occupied, empty = octree.extractPointCloud()

    aabb_min = octree.getMetricMin()
    aabb_max = octree.getMetricMax()

    visualize(
        occupied=occupied,
        empty=empty,
        K=K,
        width=camera_info['width'],
        height=camera_info['height'],
        rgb=rgb,
        pcd=pcd,
        mask=mask,
        resolution=resolution,
        aabb=(aabb_min, aabb_max),
    )


if __name__ == '__main__':
    main()
