#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
import cv2 
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, PoseStamped
import numpy as np
import moveit_commander
import sys
import tf 
from tf2_msgs.msg import TFMessage


class UPHImageSubscriber:

    def __init__ (self):

        rospy.init_node('uph_multiple_camera_subs_node')

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group.set_max_velocity_scaling_factor(0.05)

        # tracking
        self.trackers = cv2.MultiTracker_create()
        self.is_tracker_box_added = False
        self.bboxes_tracking = []

        self.bridge = CvBridge()
        self.camera_1_img_received = False
        self.camera_2_img_received = False
        self.camera_1_img = None
        self.camera_2_img = None
        self.camera_1_info = CameraInfo()
        self.camera_2_info = CameraInfo()

        camera_1_subs = rospy.Subscriber('/camera_1/image_raw', Image, self.camera_1_img_cb)
        camera_2_subs = rospy.Subscriber('/camera_2/image_raw', Image, self.camera_2_img_cb)
        camera_1_calib_subs = rospy.Subscriber('/camera_1/camera_info', CameraInfo, self.camera_1_calib_cb)
        camera_2_calib_subs = rospy.Subscriber('/camera_2/camera_info', CameraInfo, self.camera_2_calib_cb)

        self.berry_in_camera_1 = [-100,-100,1]
        self.berry_in_camera_2 = [-100,-100,1]

        self.loop()

    def camera_1_calib_cb(self, msg):
        self.camera_1_info = msg

    def camera_2_calib_cb(self, msg):
        self.camera_2_info = msg

    def berry_correctin_cb(self, msg) :
        x, y = msg.x, msg.y 
        threshold = 15 
        rospy.loginfo("X: {}\tY: {}".format(x, y))


    def camera_1_img_cb(self , data):        

        self.camera_1_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.camera_1_img = cv2.cvtColor(self.camera_1_img, cv2.COLOR_BGR2RGB)
        # self.calculate_point_to_pixel(self.berry_in_camera_1, camera="cam1")
        self.project_to_2d(self.berry_in_camera_1, self.camera_1_info, self.camera_1_img, biasX = 40, biasY = 20 )
        self.camera_1_img = cv2.rotate(self.camera_1_img, cv2.ROTATE_90_CLOCKWISE)
        self.camera_1_img_received = True        
    
    def camera_2_img_cb(self , data):        

        self.camera_2_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.camera_2_img = cv2.cvtColor(self.camera_2_img, cv2.COLOR_BGR2RGB)
        # self.calculate_point_to_pixel(self.berry_in_camera_2, camera="cam2")
        self.project_to_2d(self.berry_in_camera_2, self.camera_2_info, self.camera_2_img, biasX = -20, biasY = 0)
        self.camera_2_img = cv2.rotate(self.camera_2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.camera_2_img_received = True
 
    def project_to_2d(self, point, camera_info, image, biasX = 0, biasY = 0):

        point3d = np.array( [point], dtype = np.float64 )
        height = camera_info.height
        width = camera_info.width
        tx = np.array ( [ [0], [0], [0] ] , dtype=np.float64)
        Rx = np.array ( [ [ 1,0,0], [0,1,0], [0,0,1] ] , dtype=np.float64)
        A = np.array ( camera_info.K , dtype=np.float64).reshape(3,3) # pinhole camera model
        d = np.array ( camera_info.D , dtype=np.float64)
        p = np.array ( camera_info.P , dtype=np.float64).reshape(3,4)
        k = np.array ( camera_info.K , dtype=np.float64)
        r = np.array ( camera_info.R , dtype=np.float64)
        
        point2d = np.zeros((2,1), dtype=np.float64)
        point2d_new, H = cv2.projectPoints( point3d , Rx, tx, A, d, point2d )
        px, py = point2d_new[0][0][0], point2d_new[0][0][1]
        py = height - py + biasY
        px = width - px + biasX 

        # _point3d = np.array( [point[0], point[1], point[2], 1], dtype = np.float64).reshape(4,1)
        # project2d = np.matmul( p, _point3d ).reshape(1,-1)
        # x,y,z = project2d[0][0], project2d[0][1], project2d[0][2]
        # x /= z
        # y /= z
        # px , py = width - x, height - y

        if px > 0 and py > 0 and px < width and py < height:
            cv2.circle(image, ( int(px), int(py) ), 10, 255, -1 )
            x, y, w, h = int(px), int (py), 20, 20
            box = (x,y,w,h)
            if len(self.bboxes_tracking) > 2:
                self.bboxes_tracking = []

            self.bboxes_tracking.append(box)

    def handle_tracker_init(self, image):
        if not self.is_tracker_box_added and len(self.bboxes_tracking) > 0:
            print ("adding tracking bboxes")
            for box in self.bboxes_tracking:
                tracker = cv2.TrackerTLD_create()
                self.trackers.add( tracker, image, box )

            self.is_tracker_box_added = True

        return image

    def handle_tracker_update(self, image_copy):
        (success, boxes) = self.trackers.update(image_copy)
        # if success:
        #     for count, box in enumerate(boxes): 
        #         (x,y,w,h) = [int(v) for v in box]
        #         cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0,255,0), 2)
        #         text = "id:{}".format(count+1)
        #         cv2.putText( image_copy, text, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        return image_copy

    # def stereo_vision_camera(self):



    def loop(self):

        while not rospy.is_shutdown(): 

            if self.camera_1_img_received and self.camera_2_img_received:

                image_cpy1 = self.camera_1_img.copy()
                image_cpy2 = self.camera_2_img.copy()
                shape_1 = image_cpy1.shape[0]
                shape_2 = image_cpy2.shape[0]
                if shape_1 == shape_2 and shape_1 > 300:                
                    both_camera = np.hstack( (image_cpy2, image_cpy1) )
                    # both_camera = self.handle_tracker_init(both_camera)
                    # both_camera = self.handle_tracker_update(both_camera)
                    cv2.imshow('bottom cameras', both_camera  )

                # tf_waittime = 0.25
                # listerner = tf.TransformListener()                    
                # try: 
                #     listerner.waitForTransform( '/uph_camera_1_optical_link','/berry_point', rospy.Time(0), rospy.Duration(tf_waittime) )
                #     (trans1, rot1) = listerner.lookupTransform(  '/uph_camera_1_optical_link','/berry_point', rospy.Time(0))
                #     self.berry_in_camera_1 = trans1
                #     (trans2, rot2) = listerner.lookupTransform( '/uph_camera_2_optical_link', '/berry_point', rospy.Time(0))
                #     self.berry_in_camera_2 = trans2
                # except :
                #     pass

                k = cv2.waitKey(1)

                if k == 27:   
                    break

        cv2.destroyAllWindows()



if __name__ == "__main__": 

    image_subs_multiple = UPHImageSubscriber()
