#!/usr/bin/env python3

'''
- This node subscribes (color image) and publishes (inference results) to ROS. 
- Subcribed color image is used for berry inference. 
- The inferred results are packed in DetectedBerry messages (defined under fastpick_msg). 
- These are published to a ROS topic

Muhammad Arshad 
01/25/2022
'''

from __future__ import print_function
import rospkg
import sys
import rospy
import os 
import numpy as np
import cv2
import imutils
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from fastpick_msgs.msg import StrawberryObject, StrawberryObjects
from fastpick_enet_model import ENetModelPredictions


class FastPickBerryDetection: 

    def __init__(self): 

        # get topic names 
        rgb_image_topic = rospy.get_param("~rbg_image", "/color/image_raw")
        detect_berry_topic = rospy.get_param("~berry_results", "~/detected_berries")
        model_path = rospy.get_param("~model_path", "")

        # bridge to convert sensor_msgs/Image -> Numpy Array
        self.bridge = CvBridge()

        # susbcribe to image topic 
        self.img_rgb = None
        self.img_rbg_subscriber = rospy.Subscriber( rgb_image_topic, Image, self.img_rbg_callback )
        self.cv_window_name = "Berry-Detection-Results"

        # # publish back detected berries 
        self.berry_publisher = rospy.Publisher( detect_berry_topic, StrawberryObjects, queue_size = 10)

        # ENET mode class
        if (model_path == ""):
            model_path = os.path.join( rospkg.RosPack().get_path("fastpick_perception") , "save/ENet_Rasberry_v2/ENet" )
        if (os.path.exists(model_path)):
            self.model_client = ENetModelPredictions ( model_path = model_path)  
        else: 
            rospy.logerr("The ENet model path provided does not exists. Please set the 'model_path' param correctly")
            sys.exit(0)

        # loop function 
        self.handle_loop()
 
    def img_rbg_callback(self, data): 
        self.img_rgb = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
 
    def handle_find_contour_and_bounding_box (self, image, mask ):


        use_contour = False
        use_circle = True
        use_rect = False
        use_ellipse = False

        # object to publish
        berries_msg = StrawberryObjects()

        # preprocess before contour
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY )[1] 

        # morph filtering
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=7)
        
        # contours find
        cnts = cv2.findContours(thresh.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        cnts = imutils.grab_contours(cnts)
        
        # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
        cnts = sorted(cnts, key = lambda cnt: cv2.boundingRect(cnt)[0], reverse = False)[:20]

        for count, cnt in enumerate(cnts): 

            area = cv2.contourArea(cnt)
            if (area > 150):

                hull = cv2.convexHull(cnt)

                # rectangle
                x,y,w,h = cv2.boundingRect(cnt)
                # ellipse
                if len(cnt) >= 5: # to avoid <= error for ellipse fitting
                    ellipse = cv2.fitEllipse(cnt)

                # min enclosing circle
                (xc,yc),radius = cv2.minEnclosingCircle(cnt)
                center = (int(xc),int(yc))
                radius = int(radius)
                # check convexity
                isConvex = cv2.isContourConvex(cnt)

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = ( int (x), int(y+20) )
                fontScale              = 0.65
                fontColor              = (255,255,255)
                thickness              = 1
                lineType               = 2

                cv2.putText(image,'berry_{}'.format(count+1), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                    
                berry_mask = np.zeros( (image.shape[:2]) , dtype=np.uint8 )
                if use_contour:
                    # mask
                    cv2.drawContours(image, [cnt], -1, (50, 255, 100), 1)
                    cv2.drawContours( berry_mask, [cnt], -1, 255, -1 ) 

                elif use_rect:
                    image = cv2.rectangle(image, (x,y), (x+w,y+h), (60,255,200),2)
                    berry_mask = cv2.rectangle(berry_mask, (x,y), (x+w,y+h), (60,255,200),-1)

                elif use_circle:
                    image = cv2.circle(image, center, radius+10,(0,255,50), 1 )
                    berry_mask = cv2.circle(berry_mask, center, radius+10,(0,255,50), -1 )
                    cv2.drawContours( berry_mask, [cnt], -1, 255, -1 ) 

                elif use_ellipse and len(cnt) >= 5:
                    image = cv2.ellipse(image,ellipse,(200,0,52), 1)
                    berry_mask = cv2.ellipse(berry_mask,ellipse,(200,0,52), -1)

                indices = np.where( berry_mask == 255 ) 
                bbox = np.array( [x, y, w + x, h + y] )
                indices = np.array( indices , dtype=np.uint16 )

                bbox = bbox.flatten().tolist()
                indices = indices.flatten().tolist()

                berry = StrawberryObject()
                berry.mask = indices 
                berry.bbox = bbox
                berry.id = count+1

                berries_msg.berries.append( berry )

        self.berry_publisher.publish( berries_msg )

        return image 
 
    def handle_loop(self):
        
        rospy.loginfo("Please select '{}' window and press ESC to close the node.".format( self.cv_window_name ))
        while not rospy.is_shutdown(): 
            try:
                if not isinstance(self.img_rgb, type(None)):
                    factor = 1
                    H, W = self.img_rgb.shape[:2] 
                    image = imutils.resize( self.img_rgb.copy(), width = int( W/ factor ), height = int (H/factor))
                    image, mask = self.model_client.predict_live( image )
                    image = cv2.cvtColor( image, cv2.COLOR_RGB2BGR)
                    mask = cv2.cvtColor( mask, cv2.COLOR_RGB2BGR)
                    image = self.handle_find_contour_and_bounding_box ( image, mask )
                    image = imutils.resize( image, width = W, height = H )
                    cv2.imshow(self.cv_window_name, image ) 
                else: 
                    rospy.logwarn("Unable to find image in the topic. Did you subscribe to correct topic?")

                k = cv2.waitKey(1)
                if k == 27: 
                    cv2.destroyAllWindows()
                    break

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':

    rospy.init_node("fastpick_berry_perception_node")
    Ros = FastPickBerryDetection()
