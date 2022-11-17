#! /home/rick/anaconda3/envs/fruitcast/bin/python3
# import some common detectron2 utilities
import json
import os
import pathlib
import random
import re
import sys

#import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.data import MetadataCatalog, DatasetCatalog

# sys.path.insert(1, '/data/lincoln/weight_estimation')

# import model_prediction_realsense

# from model_prediction_realsense import predict_weight

working_dir = "/data/localdrive/myprojects/StrawDI_Db1/"


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


img_dir = '/data/localdrive/dyson/raw'
# img_dir = '/data/localdrive/StrawDI_Db1/test/img/'
# img_dir = '/data/localdrive/needle_insertion/cam3'

for rgb_file in pathlib.Path(img_dir).rglob('*_1_rgb.png'):
    weight = None
    berry_x, berry_y = None, None
    print('rgb_file: ', rgb_file.parent)
    for label_file in pathlib.Path(rgb_file.parent).rglob('*_label.npy'):
        x = np.load(label_file.as_posix())
        if x.shape[1] == 7:
            # print('data: ', x[0])
            assert x[0][0] == 1
            weight = x[0][1]
            print('weight: ', weight)
        if '_1_label.npy' in label_file.as_posix():
            x = np.load(label_file.as_posix())
            assert x[0][0] == 1.0
            berry_x, berry_y = x[0][-2:]
            print('berry_x, berry_y: ', berry_x, berry_y)

    im = cv2.imread(rgb_file.as_posix())
    # cv2.circle(im, (berry_x, berry_y), color=(255, 0, 0), thickness=10, radius=10)

    # cv2.imshow('im', im)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     break
    # else:
    #     continue

    # print(im.shape)
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    # print('outputs: ', outputs['instances'].to('cpu'))
    outputs = outputs['instances'].to('cpu')

    # print('outputs: ', outputs.pred_boxes.tensor.numpy(), outputs.pred_masks.numpy().shape)

    for bbox, pred_mask in zip(outputs.pred_boxes.tensor.numpy(), outputs.pred_masks.numpy()):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        # print(x1, y1)
        # print(x2, y2)
        berry_info = {}
        if x1 < berry_x < x2 and y1 < berry_y < y2:

            bbox_area = (x2 - x1) * (y2 - y1)
            # cv2.circle(im, (x1, y1), color=(0, 255, 0), thickness=10, radius=10)
            # cv2.circle(im, (x2, y2), color=(0, 255, 0), thickness=10, radius=10)
            # cv2.circle(im, (berry_x, berry_y), color=(255, 0, 0), thickness=10, radius=10)
            saving_path = re.sub('_1_rgb.png', '_1_detectron', rgb_file.as_posix())
            print('saving_path: ', saving_path, pred_mask.shape)
            berry_info['berry_id'] = 1
            berry_info['bbox'] = bbox
            berry_info['pred_mask'] = pred_mask
            berry_info['weight'] = weight
            berry_info['init_marking'] = [berry_x, berry_y]
            berry_info['depth'] = 20
            berry_info['bbox_area'] = bbox_area
            berry_info['polygon_area'] = pred_mask.sum() / bbox_area

            np.save(saving_path, berry_info)
    # v = Visualizer(im[:, :, ::-1],
    #                metadata=strawberry_metadata,
    #                scale=0.5,
    #                instance_mode=ColorMode.IMAGE_BW
    #                # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #                )
    #
    # out = v.draw_instance_predictions(outputs)
    # cv2.imshow('inference', out.get_image()[:, :, ::-1])
    # cv2.imshow('img', im)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     break
    # else:
    #     continue
