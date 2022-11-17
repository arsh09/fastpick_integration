import os
import socket, cv2, pickle, struct
import sys
from venv import logger

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.data import MetadataCatalog

# Prepare detectron environment
working_dir = "/data/localdrive/myprojects/StrawDI_Db1/"
strawberry_metadata = MetadataCatalog.get("strawberry_train")

cfg = get_cfg()
cfg.OUTPUT_DIR = working_dir + '/checkpoint'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("strawberry_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 1100
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
# only has one class (strawberries)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# NOTE: this config means the number of classes,
# but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)


def inference(frame):

    x, y = 0, 0

    # Inference
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(frame)
    outputs = outputs['instances'].to('cpu')
    boxes = outputs.pred_boxes if outputs.has("pred_boxes") else None
    # print(boxes)
    boxes = boxes.tensor.numpy()

    if boxes.shape == (1, 4):
        x, y = boxes[0][2] + boxes[0][0], boxes[0][3] + boxes[0][1]
        frame = cv2.circle(frame, (int(x/2), int(y/2)), radius=50, color=(255, 0, 0))
    cv2.imshow('RECEIVING FRAME', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detectron visualisation
    # v = Visualizer(frame[:, :, ::-1],
    #                metadata=strawberry_metadata,
    #                scale=0.5,
    #                # remove the colors of unsegmented pixels.
    #                # This option is only available for segmentation models
    #                instance_mode=ColorMode.IMAGE_BW
    # )
    # out = v.draw_instance_predictions(outputs)
    # cv2.imshow('inference', out.get_image()[:, :, ::-1])

    return x, y


