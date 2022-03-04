#! /usr/bin/env python
'''
    File name         : detectors.py
    Description       : Object detector used for detecting the objects in a video /image
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

# Import python libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random, math
from collections import deque
from datetime import datetime
import pyrealsense2 as rs

# import cv2
# https://www.davidhampgonsalves.com/opencv/python-color-tracking/
color_tracker_window = "Color Tracker"

class StawbDetTracker():

    def __init__(self):
        # cv2.NamedWindow( color_tracker_window, 1 )
        # self.capture = cv2.CaptureFromCAM(0)
        self.frame = None
        self.net = None
        self.detections = None
        self.args = {}
        self.args["prototxt"] = '/home/fpick/FPICK/KFTracking2DObj/models/deploy.prototxt'
        self.args["model"] = '/home/fpick/FPICK/KFTracking2DObj/models/res10_300x300_ssd_iter_140000.caffemodel'
        self.args['confidence']=0.5

        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])


    def objdetectorNN(self,frame,W,H):

        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []
        for i in range(0, detections.shape[2]):
                if detections[0, 0, i, 2] > self.args['confidence']:
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    rects.append(box.astype("int"))

                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return rects,frame

    def create_blank(self,width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

    def run(self):
        camera = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        img_counter = 0
        while True:
            ret, img = camera.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", img)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, img)
                print("{} written!".format(img_name))
                img_counter += 1
            #blur the source image to reduce color noise
            img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
            #convert the image to hsv(Hue, Saturation, Value) so its
            #easier to determine the color to track(hue)
            height, width, channels = img.shape
            # white = (255,255,255)
            # hsv_img =self.create_blank( width,  height,rgb_color=white)
            # hsv_img = cv2.CreateImage(cv2.GetSize(img), 8, 3)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #limit all pixels that don't match our criteria, in this case we are
            #looking for purple but if you want you can adjust the first value in
            #both turples which is the hue range(120,140).  OpenCV uses 0-180 as
            #a hue range for the HSV color model
            red = (255, 0, 0)
            
            thresholded_img = self.create_blank(width, height, rgb_color=red)
            red_channel = thresholded_img[:,:,2]
            # thresholded_img =  cv2.CreateImage(cv2.GetSize(hsv_img), 8, 1)
            cv2.inRange(hsv_img, (120, 80, 80), (140, 255, 255), red_channel)

            #determine the objects moments and check that the area is large
            #enough to be our object
            moments = cv2.moments(red_channel)
            area = cv2.GetCentralMoment(moments, 0, 0)

            #there can be noise in the video so ignore objects with small areas
            if(area > 100000):
                #determine the x and y coordinates of the center of the object
                #we are tracking by dividing the 1, 0 and 0, 1 moments by the area
                x = cv2.GetSpatialMoment(moments, 1, 0)/area
                y = cv2.GetSpatialMoment(moments, 0, 1)/area

                #print 'x\: ' + str(x) + ' y\: ' + str(y) + ' area\: ' + str(area)

                #create an overlay to mark the center of the tracked object
                overlay = cv2.CreateImage(cv2.GetSize(img), 8, 3)

                cv2.Circle(overlay, (x, y), 2, (255, 255, 255), 20)
                cv2.Add(img, overlay, img)
                #add the thresholded image back to the img so we can see what was
                #left after it was applied
                cv2.Merge(thresholded_img, None, None, None, img)

            #display the image
            cv2.ShowImage(color_tracker_window, img)

            # if cv2.WaitKey(10) == 27:
            #     break
        camera.release()
        cv2.destroyAllWindows()


    def detect_red_rgb(self,image,lower,upper,debugMode=True,name_pref='strawbs detect'):
        
        orig_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        listimages = [orig_frame]
        listtitles = ["Original"]


        masked = np.copy(orig_frame)
        warped = np.copy(orig_frame)

        hsv_green_mask = cv2.inRange(masked, lower, upper)
        hsv_frame = cv2.bitwise_and(masked, masked, mask = hsv_green_mask)      


        # lowerLimit = np.array([170, 150, 60])
        # lowerLimit = np.array([150, 150, 60])
        # upperLimit = np.array([179, 255, 255])  
        # Convert frame from BGR to GRAY
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ###############################################
        ###############################################



        # hsv_frame = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        res = np.hstack((warped,hsv_frame)) #stacking images side-by-side
        # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig11RGB2HSVrhm.jpg',res)
        listimages.append(hsv_frame)
        listtitles.append('hsv_frame')
        ##############################################
        # ori_img_gt = np.copy(warped)#cv2.imread('/media/dom/Elements/data/TX2_Risehm/redblobs3.png')
        # hsv_frame_gt = cv2.cvtColor(ori_img_gt, cv2.COLOR_BGR2HSV)
        # blobCentrods=m_detector.get_cvBlobCentroids(masked,hsv_frame,mask_gt)
        # lanes_gt=get_dic_gt(gt_keys, blobCentrods,bandwith)
        ##############################################    
        # mask = cv2.inRange(hsv_frame, low_green, up_green)
        # hsv_green_mask = cv2.inRange(hsv_frame, lowerLimit, upperLimit)

        #######################################
        """
        # normalizedImg = cv2.normalize(hsv_green_mask,  hsv_green_mask, 0, 255, cv2.NORM_MINMAX)
        # plt.figure('Paper writing')
        # plt.title('Normalized hsv_green_mask')        
        # plt.imshow(normalizedImg,cmap='gray')
        # plt.show(block=True)
        ## slice the green
        imask = mask>0
        greenInOriframe = np.zeros_like(warped, np.uint8)
        greenInOriframe[imask] = warped[imask]
        greenInOriframe = cv2.bitwise_and(snip, snip, mask=normalizedImg)
        """
        imask = hsv_green_mask>0
        greenInOriframe = np.zeros(hsv_frame.shape, np.uint8)
        greenInOriframe[imask] = orig_frame[imask]
        listimages.append(greenInOriframe)
        listtitles.append('greenInOriframe')
        # print ("hsv_green_mask and greenInOriframe dimension:", hsv_green_mask.shape, greenInOriframe.shape)
        # stacked_hsv_3d = np.stack((hsv_green_mask,)*3, axis=-1)
        # res = np.hstack((stacked_hsv_3d,greenInOriframe)) #stacking images side-by-side
        # plt.figure('Paper writing',figsize=(36,24))
        # plt.title('Green space HSV and RGB image')        
        # plt.imshow(res,cmap='gray')
        # plt.show(block=True)
        # plt.pause(0.25)
        # plt.close()
        # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig12GREEHSVandRGBrhm.jpg',res)
        #########################################
        # Create our shapening kernel, it must equal to one eventually
        kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])

        # applying the sharpening kernel to the input image & displaying it.
        hsv_green_mask = cv2.filter2D(hsv_green_mask, -1, kernel_sharpening)
        # print('dimension of hsv_green_mask inRanged :', hsv_green_mask.shape)
        listimages.append(hsv_green_mask)
        listtitles.append('hsv_green_mask-sharpened')
        # plt.gcf().suptitle('hsv_green_mask width: ' + str(int(snip.shape[1])) +
                            # '\ncen_hight height: ' + str(int(snip.shape[0])))

        equ = cv2.equalizeHist(hsv_green_mask)
        # res = np.hstack((hsv_green_mask,equ)) #stacking images side-by-side
        # cv.imwrite('res.png',res)
        # plt.title('before and after image')
        # plt.imshow(res,cmap='gray')
        # plt.show(block=True)
        # plt.show() 
        listimages.append(equ)
        listtitles.append('His equilization')

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(hsv_green_mask)
        # rescl1 = np.hstack((equ,cl1))
        # plt.title('His and Clahe image')
        # plt.imshow(rescl1,cmap='gray')
        # plt.show(block=True)
        # plt.show()
        listimages.append(cl1)
        listtitles.append('Clahe image')

        #blured image to help with edge detection
        # blurred = cv2.GaussianBlur(hsv_green_mask,(21,21),0); 
        """
        src – Source 8-bit or floating-point, 1-channel or 3-channel image.
        dst – Destination image of the same size and type as src .
        d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
        sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
        sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .

        """
        hsv_green_mask=equ
        # hsv_green_mask=cl1
        dims=5
        sgmcolor=75
        sgmspace=75
        blurred = cv2.bilateralFilter(hsv_green_mask,dims,sgmcolor,sgmspace)
        green_bilateral = cv2.bitwise_and(orig_frame, warped, mask=blurred)
        listimages.append(green_bilateral)
        listtitles.append('Bila Filter')
        # plt.gcf().suptitle('filtered dim: ' + str(dims) +
                            # '\nsigma color: ' + str(sgmcolor)+
                            # '\nsigma space: ' + str(sgmspace))

        # cv2.imshow("blurred", blurred)
        # cv2.waitKey(100)
        # cv2.imwrite('./output_images/blurred_wheat.png',blurred) 

        # morphological operation kernal - isolate the row - the only purpose
        # small kernel has thining segmentation , which may help with later hough transform, then group the segments of line
        #https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/morph_lines_detection/morph_lines_detection.py
        # 1st erode then dialted , in most case like this :   erode - dilate in above link
        
        rows=5
        cols=rows
        kernel = np.ones((rows,cols),dtype=np.uint8)
        # Opening = erosion followed by dilation
        erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines

        rows=25
        cols=rows
        kernel = np.ones((rows,cols),dtype=np.uint8)
        dilation = cv2.dilate(blurred,kernel,iterations = 1) #thicking, the lines
        # res = np.hstack((erosion,dilation)) #stacking images side-by-side
        green_erosion = cv2.bitwise_and(warped, warped, mask=erosion)
        
        listimages.append(green_erosion)
        listtitles.append('erosion')
        
        green_dilation = cv2.bitwise_and(warped, warped, mask=dilation)
        listimages.append(green_dilation)
        listtitles.append('dilation')


        rows=5
        cols=rows
        kernel = np.ones((rows,cols),dtype=np.uint8)
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        green_opening = cv2.bitwise_and(warped, warped, mask=opening)
        listimages.append(green_opening)
        listtitles.append('opening')

        # cv2.imshow("opening", opening)
        # cv2.waitKey(100)
        # cv2.imwrite('./output/green.png',opening)
        
        # Closing  = Dilation followed by Erosion  input from opening, and will be a bit different kernal
        rows=25
        cols=rows
        kernel = np.ones((rows,cols),dtype=np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        green_closing = cv2.bitwise_and(warped, warped, mask=closing)
        listimages.append(green_closing)
        listtitles.append('closing')
        # plt.gcf().suptitle('closing cols: ' + str(cols) +
                            # '\nsigma rows: ' + str(rows))

        thrimage = closing#erosion#cdilation#erosion#closing#opening

        blcksiz = 5
        bordsize = 2
        maxval =225
        ret,th1 = cv2.threshold(thrimage,0,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        green_OTSU = cv2.bitwise_and(warped, warped, mask=th1)
        listimages.append(green_OTSU)
        listtitles.append('adapt OTSU-th1')


        th2 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,blcksiz,bordsize)
        green_MEAN = cv2.bitwise_and(warped, warped, mask=th2)
        listimages.append(green_MEAN)
        listtitles.append('adapt MEANC-th2')
        # plt.gcf().suptitle('MEAN_C dim: ' + str(blcksiz) +
                            # '\nbordsize: ' + str(bordsize))

        th3 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,blcksiz,bordsize)

        green_GAUSSIAN = cv2.bitwise_and(warped, warped, mask=th3)
        listimages.append(green_GAUSSIAN)
        listtitles.append('adapt GAUSSC-th3')
        # plt.gcf().suptitle('GAUSSIAN_C dim: ' + str(blcksiz) +
                            # '\nbordsize: ' + str(bordsize))

        # convolute with proper kernels
        # ddepth = cv2.CV_16S#cv2.CV_64F#cv2.CV_16S
        # kernel_size =5

        sobelinput =thrimage#th1# closing#thrimage#th1##, th2, th3
        ##############################################
        ##############################################
        minthres=150#50
        maxtres=255#200
        
        img_edges = cv2.Canny(sobelinput, minthres, maxtres) #150 
        # img_edges = cv2.Canny(sobelinput,  50, 190, 3)
        # Convert to black and white image
        ret, img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)
        listimages.append(img_thresh)
        listtitles.append('Edges')
        # Find contours
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(warped, contours, -1, (0,255,0), 3)

        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        # print('Max Size: ', contour_sizes)
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x,y,w,h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,0),4)
        cv2.drawContours(warped, biggest_contour, -1, (0,0,0), 4)
        listimages.append(warped)
        listtitles.append('contours')



        # Set the accepted minimum & maximum radius of a detected object
        min_radius_thresh= 25
        max_radius_thresh= 400
        
        centers=[]
        for c in contours:
            # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = int(radius)

            #Take only the valid circle(s)
            if (radius > min_radius_thresh) and (radius < max_radius_thresh):
                centers.append(np.array([[x], [y]]))            
        # cv2.imshow('contours', img_thresh)

        """
        """
        # Plot the all resulted processed images
        ncols = 5
        nrows = 5
        
        # plot first image on its own row
        if (debugMode):
            fig = plt.figure('green ROI', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
            # fig.canvas.set_window_title('Various RGB Space')  
            plt.subplot(nrows,ncols,1)
            plt.imshow(listimages[0],'gray')
            plt.title(listtitles[0])
            plt.xticks([]),plt.yticks([])    
            
            numofl = len(listimages)
            print('total number of displays: ',numofl)
            for i in range(1,ncols*(nrows-1)+1):#range(0,ncols*(nrows-1)+1): #range(0,ncols*(nrows-1)+1): if separated one row above 
                try:
                    # plt.subplot(nrows,ncols,i+ncols),plt.imshow(listimages[i],'gray')
                    plt.subplot(nrows,ncols,i+ncols)
                    plt.imshow(listimages[i],'gray')
                    plt.title(listtitles[i])
                    plt.xticks([]),plt.yticks([])
                except IndexError:
                    print ('IndexError in plot.')
                    break

            #plt.gcf().suptitle(filename)
            plt.gcf().canvas.manager.set_window_title(name_pref)
            plt.show(block=True)
            plt.pause(0.25)
            plt.close()   

        # green_wheat = np.copy(green_sobely)
        # return sobelyedges    
        return centers,warped# orig_frame
    def point_cloud(self,depth,cp,rp):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.

        """
        # Distance factor from the cameral focal angle
        # factor = 2.0 * np.tan(cam.data.angle_x/2.0)
        angle_x = 1.0
        factor = 2.0 * np.tan(angle_x/2.0)
        
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        # Valid depths are defined by the camera clipping planes
        # valid = (depth > cam.data.clip_start) & (depth < cam.data.clip_end)
        clip_start = 0.2
        clip_end = 1.0
        valid = (depth > clip_start) & (depth < clip_end)
        
        # Negate Z (the camera Z is at the opposite)
        z = -np.where(valid, depth, np.nan)
        # Mirror X
        # Center c and r relatively to the image size cols and rows
        ratio = max(rows,cols)
        x = -np.where(valid, factor * z * (c - (cols / 2)) / ratio, 0)
        y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, 0)
        
        return np.dstack((x, y, z))


    def Pos2DPixels3Dxyz(self,depth_frame,c, r,color_intrin, depth_intrin,depth_scale):
        depth = depth_frame.get_distance(c, r)
        #THE FOLLOWING CALL EXECUTES FINE IF YOU PLUGIN A BOGUS DEPH
        depth_point_mts_cam_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [c, r], depth)
        return depth ,depth_point_mts_cam_coords

    def pcd2Octomap(self,depth_image,color_image):    

        color_image_aligned=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
        
        images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        # if show_pic:
        # cv2.imshow('aligned_images', images_aligned)
        # cv2.waitKey(100)
        return color_image_aligned,depth_colormap_aligned


    def detectMore_trackOneNN(self,warped, thrimage,areaSize,debugMode=True,name_pref='strawbs detect'):
        
        # contours, _ = cv2.findContours(thrimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find all contours

        # h, w, ch = warped.shape
        # dim = (w,h)
        # thrimage = cv2.resize(thrimage,dim, interpolation=cv2.INTER_LINEAR)

        contours, hierarchy = cv2.findContours(thrimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(warped, contours, -1, (0,255,0), 3)  
        
        # https://www.instructables.com/Opencv-Object-Tracking/
        # (_,contours,_) = cv2.findContours(img_thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours: 		
            area = cv2.contourArea(contour)

            if(area > areaSize):
                x,y,w,h = cv2.boundingRect(contour)
                warped = cv2.rectangle(warped, (x,y),(x+w,y+h),(0,0,255),10)


        # To keep track of all point where object visited
        center_points = deque()
        centers=[]
        centers_bg = []
        det_store = dict()
        """ 
        for c in contours:
            # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = int(radius)

            #Take only the valid circle(s)
            if (radius > min_radius_thresh) and (radius < max_radius_thresh):
                centers.append(np.array([[x], [y]]))  
        """

        id = -1
        for contour in contours: 		
            area = cv2.contourArea(contour)
            if(area > 100 and area < areaSize):
                id = id +1
                x,y,w,h = cv2.boundingRect(contour)
                warped = cv2.rectangle(warped, (x,y),(x+w,y+h),(255, 128, 0),10)
                moments = cv2.moments(contour)
                if moments['m00']!=0:
                    centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                    cv2.circle(warped, centre_of_contour, 2, (0, 0, 255), -1)

                    # Bound the contour with circle
                    if len(contour)>5:
                        ellipse = cv2.fitEllipse(contour)
                        cv2.ellipse(warped, ellipse, (0, 255, 0), 2)

                    # Save the center of contour so we draw line tracking it
                    center_points.appendleft(centre_of_contour)
                    centers.append(np.array([[center_points[0][0]], [center_points[0][1]]]))  
                
                    # keep in dictionary for later use
                    det_store[str(id)] = [centre_of_contour, (w,h)]
                


        # only proceed if at least one contour was found - for future application purpose
        if len(contours) > 0:
            # Find the biggest contour
            biggest_contour = max(contours, key=cv2.contourArea)
            numofsize = len(biggest_contour)
            # Find center of contour and draw filled circle
            moments = cv2.moments(biggest_contour)
            
            if moments['m00']!=0:
                centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                cv2.circle(warped, centre_of_contour, 5, (0, 0, 255), -1)

                # Bound the contour with circle
                if numofsize>5: # for the purpose of elipse drawing
                    ellipse = cv2.fitEllipse(biggest_contour)
                    cv2.ellipse(warped, ellipse, (0, 255, 255), 2)

                # Save the center of contour so we draw line tracking it
                # center_points.appendleft(centre_of_contour)
                # centers_bg.append(np.array([[center_points[0][0]], [center_points[0][1]]]))  
                # centers_bg.append(np.array(center_points[0][0], center_points[0][1])) 
                # Draw a bounding box around the first contour
                # x is the starting x coordinate of the bounding box
                # y is the starting y coordinate of the bounding box
                # w is the width of the bounding box
                # h is the height of the bounding box
                """        
                x, y, w, h = cv2.boundingRect(contours[0])
                cv2.rectangle(first_contour,(x,y), (x+w,y+h), (255,0,0), 5)
                cv2.imshow('First contour with bounding box', first_contour)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """
            else:
                print('Ambugity on big circle!')

        # Draw line from center points of contour
        print('lenth of: ',len(center_points))
        for i in range(1, len(center_points)):
            b = random.randint(230, 255)
            g = random.randint(100, 255)
            r = random.randint(100, 255)
            if math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + (
                    (center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 50:
                cv2.line(warped, center_points[i - 1], center_points[i], (b, g, r), 4)
     

        # green_wheat = np.copy(green_sobely)
        # return sobelyedges    
        return warped ,center_points,det_store# orig_frame


    def detect_3c_hsv(self,image,lowerLimit,upperLimit,debugMode=True,name_pref='strawbs detect'):

        #  int iLowH = 170;
        #  int iHighH = 179;
        #  int iLowS = 150; 
        #  int iHighS = 255;
        #  int iLowV = 60;
        #  int iHighV = 255;
        # orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2RGB)
        orig_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        listimages = [orig_frame]
        listtitles = ["Original"]

        # lowerLimit = np.array([170, 150, 60])
        # lowerLimit = np.array([150, 150, 60])
        # upperLimit = np.array([179, 255, 255])  
        # Convert frame from BGR to GRAY
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ###############################################
        ###############################################
        masked = np.copy(image)
        imageFrame = np.copy(orig_frame)


        hsv_frame = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        res = np.hstack((imageFrame,hsv_frame)) #stacking images side-by-side
        # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig11RGB2HSVrhm.jpg',res)
        listimages.append(hsv_frame)
        listtitles.append('hsv_frame')
        ##############################################
        # Set range for green color and 
        # define mask
        # Set range for red color and 
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    
        # Set range for green color and 
        # define mask
        green_lower = np.array([25, 52, 72], np.uint8)
        green_upper = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    
        # Set range for blue color and
        # define mask
        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        ##############################################    
        # mask = cv2.inRange(hsv_frame, low_green, up_green)
        # hsv_green_mask = cv2.inRange(hsv_frame, lowerLimit, upperLimit)
        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")
        
        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(orig_frame, orig_frame, 
                                mask = red_mask)
        listimages.append(res_red)
        listtitles.append('redInOriframe')

        # For green color
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(orig_frame, orig_frame,
                                    mask = green_mask)
        listimages.append(res_green)
        listtitles.append('greenInOriframe')
        
        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(orig_frame, orig_frame,
                                mask = blue_mask)

        listimages.append(res_blue)
        listtitles.append('blueInOriframe')
        #######################################
        """
        # normalizedImg = cv2.normalize(hsv_green_mask,  hsv_green_mask, 0, 255, cv2.NORM_MINMAX)
        # plt.figure('Paper writing')
        # plt.title('Normalized hsv_green_mask')        
        # plt.imshow(normalizedImg,cmap='gray')
        # plt.show(block=True)
        ## slice the green
        imask = mask>0
        greenInOriframe = np.zeros_like(warped, np.uint8)
        greenInOriframe[imask] = warped[imask]
        greenInOriframe = cv2.bitwise_and(snip, snip, mask=normalizedImg)
        
        imask = hsv_green_mask>0
        greenInOriframe = np.zeros(hsv_frame.shape, np.uint8)
        greenInOriframe[imask] = orig_frame[imask]
        listimages.append(greenInOriframe)
        listtitles.append('greenInOriframe')
        
        """
        # print ("hsv_green_mask and greenInOriframe dimension:", hsv_green_mask.shape, greenInOriframe.shape)
        # stacked_hsv_3d = np.stack((hsv_green_mask,)*3, axis=-1)
        # res = np.hstack((stacked_hsv_3d,greenInOriframe)) #stacking images side-by-side
        # plt.figure('Paper writing',figsize=(36,24))
        # plt.title('Green space HSV and RGB image')        
        # plt.imshow(res,cmap='gray')
        # plt.show(block=True)
        # plt.pause(0.25)
        # plt.close()
        # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig12GREEHSVandRGBrhm.jpg',res)
        #########################################
        # Create our shapening kernel, it must equal to one eventually
    
        ##########################################################
        ##########################################################
        # Creating contour to track red color
        # Run median blurring
        red_mask = cv2.medianBlur(red_mask, 5)
        contours, hierarchy = cv2.findContours(red_mask,
                                            cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                
                cv2.putText(imageFrame, "Red Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))    

        # Set the accepted minimum & maximum radius of a detected object
        min_radius_thresh= 25
        max_radius_thresh= 400
        
        centers=[]
        for c in contours:
            # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = int(radius)

            #Take only the valid circle(s)
            if (radius > min_radius_thresh) and (radius < max_radius_thresh):
                centers.append(np.array([[x], [y]]))         

        #########################################################################
        ##########################################################################
    
        # Creating contour to track green color
        green_mask = cv2.medianBlur(green_mask, 5)
        contours, hierarchy = cv2.findContours(green_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h),
                                        (0, 255, 0), 2)
                
                cv2.putText(imageFrame, "Green Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 255, 0))
    
        # Creating contour to track blue color
        blue_mask = cv2.medianBlur(blue_mask, 5)
        contours, hierarchy = cv2.findContours(blue_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                        (x + w, y + h),
                                        (255, 0, 0), 2)
                
                cv2.putText(imageFrame, "Blue Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 0, 0))
        
        
    
        # Plot the all resulted processed images
        ncols = 5
        nrows = 5
        
        # plot first image on its own row
        if (debugMode):
            fig = plt.figure('green ROI', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
            # fig.canvas.set_window_title('Various RGB Space')  
            plt.subplot(nrows,ncols,1)
            plt.imshow(listimages[0],'gray')
            plt.title(listtitles[0])
            plt.xticks([]),plt.yticks([])    
            
            numofl = len(listimages)
            print('total number of displays: ',numofl)
            for i in range(1,ncols*(nrows-1)+1):#range(0,ncols*(nrows-1)+1): #range(0,ncols*(nrows-1)+1): if separated one row above 
                try:
                    # plt.subplot(nrows,ncols,i+ncols),plt.imshow(listimages[i],'gray')
                    plt.subplot(nrows,ncols,i+ncols)
                    plt.imshow(listimages[i],'gray')
                    plt.title(listtitles[i])
                    plt.xticks([]),plt.yticks([])
                except IndexError:
                    print ('IndexError in plot.')
                    break

            #plt.gcf().suptitle(filename)
            plt.gcf().canvas.manager.set_window_title(name_pref)
            plt.show(block=True)
            plt.pause(0.25)
            plt.close()   

        # green_wheat = np.copy(green_sobely)
        # return sobelyedges    
        return centers,imageFrame# orig_frame



    # The original formula
    def detect(self,frame,debugMode):
        # Convert frame from BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (debugMode):
            cv2.imshow('gray', gray)

        # Edge detection using Canny function
        img_edges = cv2.Canny(gray,  50, 190, 3)
        if (debugMode):
            cv2.imshow('img_edges', img_edges)

        # Convert to black and white image
        ret, img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)
        if (debugMode):
            cv2.imshow('img_thresh', img_thresh)

        # Find contours
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set the accepted minimum & maximum radius of a detected object
        min_radius_thresh= 3
        max_radius_thresh= 30

        centers=[]
        for c in contours:
            # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = int(radius)

            #Take only the valid circle(s)
            if (radius > min_radius_thresh) and (radius < max_radius_thresh):
                centers.append(np.array([[x], [y]]))
        cv2.imshow('contours', img_thresh)
        return centers



if __name__=="__main__":
    color_tracker = StawbDetTracker()
    color_tracker.run()