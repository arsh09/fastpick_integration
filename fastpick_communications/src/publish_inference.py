#! /usr/bin/env python

import rospy 
import numpy as np 
import cv2 
from sensor_msgs.msg import Image
import ros_numpy
import os 

class PublishInference: 

    def __init__(self): 


        rospy.init_node("inference_publish_node")

        self.img_topic = rospy.Publisher("/detected/image_raw", Image, queue_size=10)
        self.read_directory()
        self.handle_loop()

    def read_directory(self):

        image_folder = "/home/fc/Downloads/mqtt-inference"
        image_files = []
        if os.path.exists( image_folder ): 
            image_files = os.listdir( image_folder )
            image_files = [ os.path.join( image_folder, image_file ) for image_file in image_files ]
        else: 
            rospy.logerr("{} is not correct path".format(image_folder))

        return image_files

    def handle_loop(self): 

        img_count = 0
        img_files = self.read_directory()
        while not rospy.is_shutdown():
            try: 
                if img_count >= len( img_files): 
                    img_count = 0
                else: 
                    img = cv2.imread( img_files[img_count] )
                    img_count += 1
                    
                    ros_img = ros_numpy.msgify( Image , img , encoding = 'rgb8')
                    self.img_topic.publish( ros_img )

                    rospy.Rate(30).sleep()
                    # cv2.imshow("img", img)
                    # k = cv2.waitKey(30)
                    # if k == 27:
                    #     cv2.destroyAllWindows() 
                    #     break

            except KeyboardInterrupt: 
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":

    PublishInference()