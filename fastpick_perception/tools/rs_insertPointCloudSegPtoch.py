#!/usr/bin/env python
from math import inf
import numpy as np
from numpy.core.shape_base import block
import matplotlib.pyplot as plt
# from sensor_msgs.msg import Image, CameraInfo
#######################################################
#######################################################
from datetime import datetime
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import *
import sys
########################################################
from tools.rsDetect_TrackSeg import StawbDetTracker
# import rospy  # Python library for ROS
# from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
import pyrealsense2 as rs
import numpy as np
# import octomap
import os
import cv2

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

    ##############################################
    ##############################################


class MapManager(object):
    def __init__(self):

        self.aligned_frames = None  # choose webcam False: choose realsense camera
        self.depth_frame = None
        self.color_frame = None
        self.depth_image = None
        self.color_image = None
        self.num = -1
        self.num_images = inf
        # self.depth_image_oct   = 0.0
        self.messages_img=None
        self.messages_alignedepth = None
        
        self.obj_data = None
        self.ctl_data = None
        # self.camera_info = CameraInfo()
        self.width = 1280
        self.height = 720
        self.fx = 908.594970703125
        self.fy = 908.0234375
        self.ppx = 650.1152954101562
        self.ppy = 361.9811096191406
        self.output_path = 'results'
        if os.path.isdir(self.output_path)==False:
            os.mkdir(self.output_path)
            print("self.output_path '% s' created" % self.output_path)


        self.flag = True

        self.K = np.array([[self.fx, 0, self.ppx],
                           [0, self.fy, self.ppy],
                           [0, 0, 1]])

        self.K[0, 0] = self.fx
        self.K[1, 1] = self.fy
        self.K[0, 2] = self.ppx
        self.K[1, 2] = self.ppy

        self.depth_intrin = rs.pyrealsense2.intrinsics()
        self.color_intrin = None #rs.pyrealsense2.intrinsics()
        self.depth_to_color_extrin = None
        self.depth_sensor = None
        self.depth_scale = 0.001
        # Intrinsics & Extrinsics
        self.fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        self.fourcc2 = cv2.VideoWriter_fourcc('M','J','P','G')
        self.avi_width = 2560
        self.avi_height = 720
        self.outRes = None
        # self.depth_intrin =  [ 1280x720  p[650.115 361.981]  f[908.595 908.023]  Inverse Brown Conrady [0 0 0 0 0] ]
        self.depth_intrin.width =1280
        self.depth_intrin.height=720
        self.depth_intrin.ppx=650.1152954101562
        self.depth_intrin.ppy=361.9811096191406
        self.depth_intrin.fx = 908.594970703125
        self.depth_intrin.fy = 908.0234375
        self.depth_intrin.model=rs.distortion.none
        self.depth_intrin.coeffs=[0.0, 0.0, 0.0, 0.0, 0.0]
       
        
        print('simu_depth_intrin = ',self.depth_intrin)

        self.lowerLimit = np.array([150, 150, 60], np.uint8)
        self.upperLimit = np.array([179, 255, 255], np.uint8)
        #####################################################
        self.rsDetTrck = StawbDetTracker()
        # detection and tracking points from segmentation in image plate: (x,y)
        self.centers = None
        self.depth_point = None  # 3D position [x, y, z]
        #####################################################

    def ShutdownPub(self):
        print("shutdown Strawbs Pos Publisher!")
        cv2.destroyAllWindows()
        rospy.is_shutdown = True
        exit()

    def DetTrckPtPublisher(self):
        # https://www.ncnynl.com/archives/201611/1065.html
        pub_strwpos = rospy.Publisher('floats', numpy_msg(Floats))
        rospy.init_node('rs_insertPointCloudSeg', anonymous=True)
        rate = rospy.Rate(0.1)  # 10hz

        while not rospy.is_shutdown():

            # Print debugging information to the terminal
            rospy.loginfo('publishing realsense cam rosbag frame')
            self.centers = np.array(
                [1.0, 2.1, 3.2, 4.3, 5.4, 6.5], dtype=np.float32)
            pos = np.array(self.centers, dtype=np.float32)
            pub_strwpos.publish(pos)

            rate.sleep()
        key = cv2.waitKey(1) & 0xFF   # Esc - kill, q, skipping
        if key == ord('q') or key == 27:
            rospy.on_shutdown(self.ShutdownPub())
   
       

    def rs_callback_cvSeg_cam(self,depth_image,img_ori, img_inst,figsave_path=''):

        self.color_image = img_ori
        self.depth_image = depth_image

        plt.imshow(self.color_image)   
        plt.title('self.color_image')
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()

        height, width, ch = img_inst.shape
        # img = np.zeros([height, width, 1], dtype=np.uint8)
        img = np.zeros([height, width], dtype=np.uint8)
        # info = np.iinfo(img_inst.dtype) # Get the information of the incoming image type
        # data = img_inst.astype(np.float64) / info.max # normalize the data to 0 - 1
        # data = 255 * data # Now scale by 255
        # img = data.astype(np.uint8)
        
        # _img_inst = np.copy(img_inst)
        # img[img_inst>0.1] = 0.0
        img_inst_ch = img_inst[:,:,0]
        img[img_inst_ch<0.1] = 255   
        # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # converting to its binary form
        # ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
        kernel = np.ones((9, 9), 'uint8')
        self.inst_image = cv2.dilate(img, kernel, iterations=1)

        # numpy_vertical = np.vstack((image, grey_3_channel))
        numpy_horizontal = np.hstack((img, self.inst_image))

        # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
        # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
        cv2.imshow('Ori-Dilated Image', numpy_horizontal)
        cv2.waitKey(500)
        cv2.destroyAllWindows()       


        recxy = 10     
        h, w, ch = self.color_image.shape
        dim = (w,h)
        self.inst_image = cv2.resize(self.inst_image,dim, interpolation=cv2.INTER_LINEAR)   

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.color_image,cmap='gray')
        axarr[1].imshow(self.inst_image,cmap='gray')

        name = 'rsiz_inst_enhance_'+str(self.num)+'.png' 
        sved_inst = figsave_path+name   
        plt.savefig(sved_inst, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()    

        self.frame, centers,det_store = self.rsDetTrck.detectMore_trackOneNN(
            self.color_image, self.inst_image,5000, debugMode=True, name_pref='strawbs detect')
        
        # strawbs detection
        name = 'straw_det_' + str(self.num)+'.png'
        sved_det = figsave_path+name  
        plt.imshow(self.frame)   
        plt.savefig(sved_det, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()
        # option for Rob to choose  - list ,dictionary  adapted into saga's tracking system
        for id, item in det_store.items():   # ditctionary = [centrr , and [w, h]]
            print(item)
            print(det_store[id])

        det_store_lst = list(sorted(det_store.keys()))
        print(det_store_lst)
        
        depth_points = []
        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            ix = int(cx)
            iy = int(cy)
            # just for the verification of fomular - seems agree to each other
            depth = self.depth_scale*self.depth_image[iy, ix]
            depth = self.depth_frame.get_distance(ix, iy)
            depth_point = rs.rs2_deproject_pixel_to_point(self.color_intrin, [ix, iy], depth)
            print ('result:', depth_point)
            # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
            # sys.stdout.flush()
            depth_points = []

            # Detect object maxctroid: the largest contour
            # distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            #     self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            if depth_point is not None:
                
                depth_points.append(depth_point)

                text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                    depth_point[0], depth_point[1], depth_point[2])
                # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                cv2.putText(self.color_image, text, (cx + recxy,
                            cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
                # drawn on colormap - acting as mesh at momoment:

                cv2.putText(self.frame, text,(cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.frame,(cx, cy), 8, (0, 0, 255), 1)

                cv2.putText(self.inst_image, text,(cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.inst_image,(cx, cy), 8, (0, 0, 255), 1)

                """figname = 'strw_xyz_frm'+str(self.num)+'.png'
                # os.chdir(OUTDIR)
                nampepath = os.path.join(output_path, figname)
                depth_colormap = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(nampepath, depth_colormap)"""
                # Stack both images horizontally
                # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
                # os.chdir("..")  

        figname = 'strw_xyz_frm'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        depth_colormap = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, depth_colormap)

        avi_res = np.hstack((depth_colormap, self.frame))
        self.inst_image3 = cv2.merge((self.inst_image,self.inst_image,self.inst_image))
        avi_res = np.hstack((avi_res, self.inst_image3))

        avi_res = cv2.resize(avi_res,(self.avi_width,self.avi_height), interpolation=cv2.INTER_LINEAR) 
        key = cv2.waitKey(100)
        self.outRes.write(avi_res)
        if key & 0xFF == ord('s'):
            cv2.waitKey(0)
        if key==27 & 0xFF == ord('q'):    # Esc key to stop, 113: q
            cv2.destroyAllWindows()
            return
        
        numpy_horizontal
        return centers, depth_points
    
    # using rosbag as input data 
    def rs_callback_cvSeg_rosbag2(self,depth_image,img_ori, img_inst,figsave_path=''):
        self.color_image = img_ori
        self.depth_image = depth_image

        plt.imshow(self.color_image)   
        plt.title('self.color_image')
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()

        height, width, ch = img_inst.shape
        # img = np.zeros([height, width, 1], dtype=np.uint8)
        img = np.zeros([height, width], dtype=np.uint8)
        # info = np.iinfo(img_inst.dtype) # Get the information of the incoming image type
        # data = img_inst.astype(np.float64) / info.max # normalize the data to 0 - 1
        # data = 255 * data # Now scale by 255
        # img = data.astype(np.uint8)
        
        # _img_inst = np.copy(img_inst)
        # img[img_inst>0.1] = 0.0
        img_inst_ch = img_inst[:,:,0]
        img[img_inst_ch<0.1] = 255   
        # gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # converting to its binary form
        # ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
        kernel = np.ones((9, 9), 'uint8')
        self.inst_image = cv2.dilate(img, kernel, iterations=1)

        # numpy_vertical = np.vstack((image, grey_3_channel))
        numpy_horizontal = np.hstack((img, self.inst_image))

        # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
        # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
        cv2.imshow('Ori-Dilated Image', numpy_horizontal)
        cv2.waitKey(500)
        cv2.destroyAllWindows()      

        recxy = 10     
        h, w, ch = self.color_image.shape
        dim = (w,h)
        self.inst_image = cv2.resize(self.inst_image,dim, interpolation=cv2.INTER_LINEAR)   

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.color_image,cmap='gray')
        axarr[1].imshow(self.inst_image,cmap='gray')

        name = 'rsiz_inst_enhance_'+str(self.num)+'.png' 
        sved_inst = figsave_path+name   
        plt.savefig(sved_inst, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()    

        self.frame, centers,det_store = self.rsDetTrck.detectMore_trackOneNN(
            self.color_image, self.inst_image,5000, debugMode=True, name_pref='strawbs detect')
        
        # strawbs detection
        name = 'straw_det_' + str(self.num)+'.png'
        sved_det = figsave_path+name  
        plt.imshow(self.frame)   
        plt.savefig(sved_det, bbox_inches='tight')     
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()
        # option for Rob to choose  - list ,dictionary  adapted into saga's tracking system
        for id, item in det_store.items():   # ditctionary = [centrr , and [w, h]]
            print(item)
            print(det_store[id])

        det_store_lst = list(sorted(det_store.keys()))
        print(det_store_lst)

        src2 = np.zeros_like(self.color_image)
        src2[:,:,0] = self.inst_image[:,:]
        src2[:,:,1] = self.inst_image[:,:]
        src2[:,:,2] = self.inst_image[:,:]
        self.inst_image = src2

        depth_points = []
        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            ix = int(cx)
            iy = int(cy)
            # just for the verification of fomular - seems agree to each other
            depth = self.depth_scale*self.depth_image[iy, ix]
            # depth = self.depth_frame.get_distance(ix, iy)
            depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [ix, iy], depth[0])
            print ('result:', depth_point)
            # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
            # sys.stdout.flush()
            depth_points = []

            # Detect object maxctroid: the largest contour
            # distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            #     self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            if depth_point is not None:
                
                depth_points.append(depth_point)

                text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                    depth_point[0], depth_point[1], depth_point[2])
                # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                cv2.putText(self.color_image, text, (cx + recxy,
                            cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
                # drawn on colormap - acting as mesh at momoment:

                cv2.putText(self.frame, text,(cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.frame,(cx, cy), 8, (0, 0, 255), 1)

                cv2.putText(self.inst_image, text,(cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
                cv2.circle(self.inst_image,(cx, cy), 8, (0, 0, 255), 1)

                """figname = 'strw_xyz_frm'+str(self.num)+'.png'
                # os.chdir(OUTDIR)
                nampepath = os.path.join(output_path, figname)
                depth_colormap = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(nampepath, depth_colormap)"""
                # Stack both images horizontally
                # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
                # os.chdir("..")  

        figname = 'strw_xyz_frm'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        depth_colormap = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, depth_colormap)

        # avi_res = np.hstack((depth_colormap, self.frame))

        avi_res = np.hstack((depth_colormap, src2))
        avi_res = cv2.resize(avi_res,(self.avi_width,self.avi_height), interpolation=cv2.INTER_LINEAR) 
        cv2.imshow('avi_res', avi_res)
        
        key = cv2.waitKey(500)
        self.outRes.write(avi_res)
        if key & 0xFF == ord('s'):
            cv2.waitKey(0)
        if key==27 & 0xFF == ord('q'):    # Esc key to stop, 113: q
            cv2.destroyAllWindows()
            return
        
        numpy_horizontal
        return centers, depth_points
    ##################################################################################
    ##################################################################################

    def rs_callback_cvSeg_bag(self):
        # data = imgviz.data.arc2017()
        # camera_info = data['camera_info']
        # K = np.array(camera_info['K']).reshape(3, 3)
        # rgb = data['rgb']
        
        # depth_image = data['depth']
        #  https://github.com/IntelRealSense/realsense-ros/issues/1342

        recxy = 10

        centers, frame = self.rsDetTrck.detectMore_trackOne(
            self.color_image, self.lowerLimit, self.upperLimit, 800, debugMode=True, name_pref='strawbs detect')

        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            # Detect object maxctroid: the largest contour
            distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
                self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                depth_point[0], depth_point[1], depth_point[2])
            # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(self.color_image, text, (cx + recxy,
                        cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
            # drawn on colormap - acting as mesh at momoment:

            cv2.putText(self.depth_colormap_aligned, text,
                        (cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.depth_colormap_aligned,
                       (cx, cy), 8, (0, 0, 255), 1)

            figname = 'strw_xyz_msh'+str(self.num)+'.png'
            # os.chdir(OUTDIR)
            nampepath = os.path.join(self.output_path, figname)
            # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            cv2.imwrite(nampepath, self.color_image)
            # Stack both images horizontally
            # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
            # os.chdir("..")

        distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
        text = "xyz: %.5lf, %.5lf, %.5lfm" % (
            depth_point[0], depth_point[1], depth_point[2])
        # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(self.color_image, text, (cx + recxy,
                    cy - recxy), 0, 0.5, (0, 200, 255), 2)
        cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
        # drawn on colormap - acting as mesh at momoment:
        # cv2.putText(self.depth_colormap_aligned,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
        # cv2.circle(self.depth_colormap_aligned, (cx,cy), 8, (0, 0, 255), 1)

        figname = 'strw_xyz_pos'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, self.color_image)      

        # pcd = pointcloud_from_depth(
        #     self.depth_image, fx=self.fx, fy=self.fy, cx=self.ppx, cy=self.ppy)

        return centers, depth_point
    def rs_callback_cvSeg_rosbag(self):
        # data = imgviz.data.arc2017()
        # camera_info = data['camera_info']
        # K = np.array(camera_info['K']).reshape(3, 3)
        # rgb = data['rgb']
        # depth_image = data['depth']
        recxy = 10

        centers, frame = self.rsDetTrck.detectMore_trackOne(
            self.color_image, self.lowerLimit, self.upperLimit, 800, debugMode=True, name_pref='strawbs detect')

        for count, value in enumerate(centers):
            print(count, value)
            cx = int(centers[count][0])
            cy = int(centers[count][1])
            if cx < recxy or cy < recxy:
                break
            #################################################################################
            # 3D translation and transformation with centroilds, realsense. open3d
            #################################################################################
            # Detect object maxctroid: the largest contour
            distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
                self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
            text = "xyz: %.5lf, %.5lf, %.5lfm" % (
                depth_point[0], depth_point[1], depth_point[2])
            # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(self.color_image, text, (cx + recxy,
                        cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
            # drawn on colormap - acting as mesh at momoment:

            cv2.putText(self.depth_colormap_aligned, text,
                        (cx + recxy, cy - recxy), 0, 0.5, (0, 200, 255), 2)
            cv2.circle(self.depth_colormap_aligned,
                       (cx, cy), 8, (0, 0, 255), 1)

            figname = 'strw_xyz_msh'+str(self.num)+'.png'
            # os.chdir(OUTDIR)
            nampepath = os.path.join(self.output_path, figname)
            # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            cv2.imwrite(nampepath, self.color_image)
            # Stack both images horizontally
            # images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
            # os.chdir("..")

        distance, depth_point = self.rsDetTrck.Pos2DPixels3Dxyz(
            self.depth_frame, cx, cy, self.color_intrin, self.depth_intrin, self.depth_scale)
        text = "xyz: %.5lf, %.5lf, %.5lfm" % (
            depth_point[0], depth_point[1], depth_point[2])
        # cv2.putText(color_frame, "Measured xyz{}m".format(distance), (depth_point[0], depth_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        # cv2.putText(frame, text, (int(depth_point[0]), int(depth_point[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(self.color_image, text, (cx + recxy,
                    cy - recxy), 0, 0.5, (0, 200, 255), 2)
        cv2.circle(self.color_image, (cx, cy), 8, (0, 0, 255), 1)
        # drawn on colormap - acting as mesh at momoment:
        # cv2.putText(self.depth_colormap_aligned,text, (cx + recxy, cy - recxy), 0, 0.5, (0,200,255), 2)
        # cv2.circle(self.depth_colormap_aligned, (cx,cy), 8, (0, 0, 255), 1)

        figname = 'strw_xyz_pos'+str(self.num)+'.png'
        # os.chdir(OUTDIR)
        nampepath = os.path.join(self.output_path, figname)
        # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nampepath, self.color_image)
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

        return centers, depth_point     