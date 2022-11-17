####! /home/rick/anaconda3/envs/fruitcast/bin/python3
# i###mport some common detectron2 utilities
import json
import os
import pathlib
import random
import sys

#import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.data import MetadataCatalog, DatasetCatalog

# sys.path.insert(1, '/data/lincoln/weight_estimation')
#
# import model_prediction_realsense
#
# from model_prediction_realsense import predict_weight

working_dir = "/home/robofruit/localdrive/myprojects/robofruit_weight_estimation/"


def read_json(file_path):
    with open(file_path, 'r') as read_file:
        data = json.load(read_file)
    read_file.close()
    return data


for d in ["train"]:
    DatasetCatalog.register("strawberry_" + d, lambda d=d: read_json(working_dir + d))
    MetadataCatalog.get("strawberry_" + d).set(thing_classes=["strawberry"])
strawberry_metadata = MetadataCatalog.get("strawberry_train")


cfg = get_cfg()
cfg.OUTPUT_DIR = working_dir + '/checkpoint'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("strawberry_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# dataset_dicts = read_json(working_dir + '/val')
# for d in dataset_dicts:
#     im = cv2.imread(d["file_name"])
#     print(im.shape)
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=strawberry_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('inference', out.get_image()[:, :, ::-1])
#     key = cv2.waitKey(0)
#     if key == ord('q'):
#         break
#     else:
#         continue


# img_dir = '/data/localdrive/strawberry_multi_cam/rgb_image/'
img_dir = '/home/robofruit/localdrive/datasets/dyson_dense_rgb/'
# img_dir = '/data/localdrive/StrawDI_Db1/test/img/'
# img_dir = '/data/localdrive/needle_insertion/cam3'

for rgb_file in pathlib.Path(img_dir).rglob('*.jpg'):

    im = cv2.imread(rgb_file.as_posix())
    # print(im.shape)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # print('outputs: ', outputs['instances'].to('cpu'))
    outputs = outputs['instances'].to('cpu')
    # boxes = outputs.pred_boxes if outputs.has("pred_boxes") else None
    # boxes = boxes.tensor.numpy()

    # print('boxes shape: ')
    # if boxes.shape == (1,4):
    #     predict_weight(im)

    # boxes = boxes.tensor.numpy()
    # print(dir(outputs['instances'].to('cpu')))
    # print('outputs: ', outputs)
    v = Visualizer(im[:, :, ::-1],
                   metadata=strawberry_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )

    out = v.draw_instance_predictions(outputs)
    cv2.imshow('inference', out.get_image()[:, :, ::-1])
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord('w'):
        continue
