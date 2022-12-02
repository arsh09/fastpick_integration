
#general imports
import os, numpy as np, cv2,pickle, logging
from enum                        import Enum,unique
from scipy                       import ndimage

# detectron imports
from detectron2.config           import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2                  import model_zoo

# project imports
from fastpick_visualizer        import FastPickVisualizer, ColorMode

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class MasksPredictor:

    def __init__(self, model_file,config_file,metadata_file,num_classes=1, scale=1.0, instance_mode=ColorMode.SEGMENTATION):

        self.instance_mode=instance_mode
        self.scale=scale
        self.metadata=self.get_metadata(metadata_file)
        cfg = self.init_config(model_file, config_file, num_classes)

        try:
            self.predictor=DefaultPredictor(cfg)
        except Exception as e:
            logging.error(e)

    def init_config(self, model_file, config_file, num_classes=1):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except Exception as e:
            logging.error(e)


        cfg.MODEL.WEIGHTS = os.path.join(model_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 1.strawberry (only strawberry coming from model)
        return cfg

    def get_metadata(self,metadata_file):

        #metadata file has name of classes, it is created to avoid having custom dataset and taking definitions
        # from annotations, instead. It has structure of MetaDataCatlog output of detectron2
        try:
            file = open(metadata_file, 'rb')
        except Exception as e:
            logging.error(e)

        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self,rgbd_image,class_list):

        if (bool(class_list)==False): 
            logging.error('class list empty')


        depth_image = rgbd_image[:, :, 3]
        rgb_image=rgbd_image[:, :, :3]

        outputs = self.predictor(rgb_image)
        # [16/02/22]:format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = FastPickVisualizer(rgb_image,
                       metadata=self.metadata,
                       scale=self.scale,
                       instance_mode=self.instance_mode
                       )


        predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        berry_mask = predictions.get_image()[:, :, ::-1]
        temp_mask = np.bitwise_and(berry_mask[:,:,0]==255,berry_mask[:,:,1]==255)
        berry_mask =np.bitwise_and(temp_mask==1,berry_mask[:,:,2]==0)
        not_berry=np.bitwise_not(berry_mask)

        canopy_mask=self.get_canopy(rgb_image)
        canopy_mask = self.smooth_seg(canopy_mask,ClassNames.CANOPY)
        canopy_mask =np.bitwise_and(canopy_mask,not_berry)
        bg_mask = self.get_background(depth_image)
        bg_mask = self.smooth_seg(bg_mask,ClassNames.BACKGROUND)


        temp_mask=np.bitwise_and(canopy_mask,bg_mask)
        temp_mask = np.bitwise_not(temp_mask)
        bg_mask = np.bitwise_and(temp_mask, bg_mask)
        bg_mask = np.bitwise_and(bg_mask, not_berry)

        rigid_mask=np.bitwise_not(np.bitwise_or(canopy_mask,bg_mask))
        rigid_mask = np.bitwise_and(rigid_mask, not_berry)

        all_mask=np.bitwise_or(rigid_mask, bg_mask)
        all_mask = np.bitwise_or(all_mask, berry_mask)
        all_mask = np.bitwise_or(all_mask, canopy_mask)
        unseg_mask=np.bitwise_not(all_mask)
        bg_mask = np.bitwise_or(unseg_mask, bg_mask)
        fg_masks = (np.dstack((berry_mask, canopy_mask, rigid_mask,bg_mask)) * 1)
        return self.get_masks(fg_masks, depth_image,class_list)

    def get_background(self,depth_image):

        depth_zero = (depth_image == 0)*1
        depth_zero = depth_zero * np.amax(depth_image)
        depth_image = depth_image + depth_zero
        bg_mask = depth_image > 5
        return bg_mask


    def get_canopy(self,rgb_image):

        I = np.asarray(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV))
        red_range   = [40,122]
        green_range = [21,223]
        blue_range  = [0,223]

        canopy_mask=(I[:,:, 0] >= red_range[0] ) & (I[:,:, 0] <= red_range[1]) & \
        (I[:,:, 1] >= green_range[0] ) & (I[:,:, 1] <= green_range[1]) & \
        (I[:,:, 2] >= blue_range[0] ) & (I[:,:, 2] <= blue_range[1])


        return canopy_mask

    def smooth_seg(self,input_mask,class_name):
        h, w = input_mask.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        input_mask=input_mask*255
        mask=input_mask.astype('uint8')
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if class_name==ClassNames.CANOPY:
            cv2.floodFill(mask, flood_mask, (0, 0), 255)
            flood_mask=flood_mask[:-2,:-2]
        else:
            flood_mask=mask
        flood_mask=ndimage.binary_fill_holes(flood_mask,structure=np.ones((5,5)))
        return flood_mask

    def get_masks(self,fg_masks, depth_image, class_list):

        # input three foreground class' masks and calculate leftover as background mask
        # then output requested depth masks as per class_list order


        depth_masks=list()
        for classes in class_list:
            if   classes==ClassNames.STRAWBERRY:
                depth_masks.append(fg_masks[:,:,0]*depth_image)
            elif classes == ClassNames.CANOPY:
                depth_masks.append(fg_masks[:,:,1]*depth_image)
            elif classes == ClassNames.RIGID_STRUCT:
                depth_masks.append(fg_masks[:,:,2]*depth_image)
            elif classes == ClassNames.BACKGROUND:
                depth_masks.append(fg_masks[:,:,3]*depth_image)
        return (np.dstack(depth_masks))

@unique
class ClassNames(Enum):
    """
    Enum of different class names
    """

    STRAWBERRY   = 1
    """
    Class strawberry, depicted by yellow colour
    """
    CANOPY       = 2
    """
    Class canopy, depicted by green colour
    """
    RIGID_STRUCT = 3
    """
    Class rigid structure, depicted by red colour
    """

    BACKGROUND   = 4
    """
    Class background, depicted by blue colour
    """


