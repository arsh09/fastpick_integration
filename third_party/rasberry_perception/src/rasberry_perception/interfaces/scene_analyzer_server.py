# Muhammad Arshad Khan
# Date: 22-June-2022

import ros_numpy

from rasberry_perception.interfaces.registry import DETECTION_REGISTRY
from rasberry_perception.interfaces.default import BaseDetectionServer
from rasberry_perception.msg import Detections, ServiceStatus, Detection, RegionOfInterest, SegmentOfInterest
from rasberry_perception.srv import GetDetectorResultsRequest, GetDetectorResultsResponse

import time 
import rospy
import os
from sensor_msgs.msg import Image

#general imports
import os, sys, numpy as np, pickle, logging
import pickle            

# detectron imports
from detectron2.config import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer


@DETECTION_REGISTRY.register_detection_backend("scene_analyzer_server")   
class SceneAnalyzerServer(BaseDetectionServer):  

    def __init__(self, model_path, metadata_path, cfg_path ): 

        # initialize mask predictor
        self.busy = True
        self.scene_analyser_predictor = None
        if model_path == "" : 
            raise ("Please provide scene analyser model file absolute path") 
        elif metadata_path == "" : 
            raise ("Please proide scene analyser meta file absolute path")
        elif cfg_path == "": 
            raise ("Please proide scene analyser config name")
        elif not os.path.exists( model_path ): 
            raise ("'model_path' does not exists" )
        elif not os.path.exists( metadata_path ) : 
            raise ("'metadata_path' does not exists" )
        else: 
            self.scene_analyser_predictor = MasksPredictor( model_path, cfg_path, metadata_path )

        self.busy = False 
        rospy.loginfo("Initializing the scene analyzer server...")
        time.sleep(1)
        BaseDetectionServer.__init__(self)  # Spins the server and waits for requests!

    def get_detector_results(self, request):  # (3)
        if self.busy:  # Example of other status responses
            return GetDetectorResultsResponse(status=ServiceStatus(BUSY=True))
                
        # Populate a detections message
        detections = Detections()
        detections = self.scene_analyser_predictor.get_predictions( request.image )

        return GetDetectorResultsResponse(status=ServiceStatus(OKAY=True), results=detections)
    



class MasksPredictor:
    """
    General MaskRCNN model class. 
    """
    def __init__(self, model_file, config_file, metadata_file, num_classes  = 1):
        """ A rather simple wrapper around mask-rcnn model using detectron2 library
            for inference.

        Args: 
            - mode_file (str): absolute path to maskrcnn .pth file. 
            - metadata_file (str): absolute path to maskrcnn meta data file that stores classes name etc.
            - config_file (str): detectron2 config used during model training. 
        Kwargs:
            - num_classes (int): 
        """

        self.metadata=self.get_metadata(metadata_file)
        self.cfg = self.init_config(model_file, config_file, num_classes)

        try:
            self.predictor=DefaultPredictor(self.cfg)
        except Exception as e:
            logging.error(e)
            print(e)
            raise Exception(e)

    def init_config(self, model_file, config_file, num_classes):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except Exception as e:
            logging.error(e)
            print(e)
            raise Exception(e)

        cfg.MODEL.WEIGHTS = model_file  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes   
        return cfg

    def get_metadata(self, metadata_file):

        try:
            file = open(metadata_file, 'rb')
        except Exception as e:
            logging.error(e)
            print(e)
            raise Exception(e)

        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self, ros_rgb_image ):
        
        # get the inference from model
        np_rgb_image = ros_numpy.numpify( ros_rgb_image )
        outputs = self.predictor( np_rgb_image )

        # though not absolutely neccessary, but we can use 
        # detectron2 visualizer to draw the inference. 
        v = Visualizer(np_rgb_image[:, :, ::-1], self.metadata , scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # get the inference in as numpy arrays
        pred_classes = outputs["instances"].get_fields()["pred_classes"].to("cpu").numpy()
        scores = outputs["instances"].get_fields()["scores"].to("cpu").numpy()
        pred_masks = outputs["instances"].get_fields()["pred_masks"].to("cpu").numpy()
        pred_boxes = outputs["instances"].get_fields()["pred_boxes"].to("cpu").tensor.numpy()

        # fill the ROS detection message 
        # for service client.
        detections = Detections()
        detections.camera_frame = ros_numpy.msgify( Image, output_image, encoding=ros_rgb_image.encoding)

        for detected_object, score in enumerate(scores): 

            detection = Detection()            
            roi = RegionOfInterest()
            soi = SegmentOfInterest()
            roi.x1, roi.y1, roi.x2, roi.y2 = pred_boxes[detected_object]

            mask = np.where( pred_masks[detected_object]  )
            soi.x , soi.y = list( mask[1] ) , list( mask[0] ) 
        
            detection.roi = roi
            detection.seg_roi = soi
            detection.class_name = "Strawberry"
            detection.confidence = score
            detections.objects.append( detection )

        return detections