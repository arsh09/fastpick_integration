import os
import re

#import cv2
import numpy as np
import pathlib
import logging
import json

from inference_method import prediction
from process_depth import process_depth, get_avg_depth
# import matplotlib.pyplot as plt

logger = logging.getLogger('my-app')
logger.setLevel(logging.INFO)
logging.basicConfig()

img_dir = '/data/localdrive/robofruit_weightdataset/dataset'


for data_dir in pathlib.Path(img_dir).glob('*/*'):

    logger.debug('label data_dir: ' + data_dir.as_posix())
    collector = data_dir.as_posix().split('/')[-2]
    logger.debug('collector: ' + collector)
    folder_no = data_dir.as_posix().split('/')[-1]
    logger.debug('folder_no: ' + folder_no)
    sample_id = collector + '_' + folder_no
    logger.debug('sample_id: ' + sample_id)

    berry_info = {sample_id: ['image_id', {}]}
    # berry_json = {'sample': sample_id, 'image': []}

    for label_file in pathlib.Path(data_dir).glob('*_label.npy'):

        logger.info('label filename: ' + label_file.as_posix())

        weight = None
        berry_x, berry_y = 0, 0

        berries = np.load(label_file.as_posix())
        image_id = re.search('_([0-9])_label', label_file.as_posix())[1]
        logger.debug('image id: ' + image_id)
        rgb_file = re.sub('_label.npy', '_rgb.png', label_file.as_posix())
        # depth_file = re.sub('_label.npy', '_rdepth.npy', label_file.as_posix())
        depth_file = re.sub('_label.npy', '_pdepth.png', label_file.as_posix())
        outputs, im = prediction(rgb_file)

        # berry_info = {sample_id: ['image_id', {image_id: []}]}
        berry_info[sample_id][1].update({image_id: ['berry_id', {}]})
        # logger.debug('image_id: ' + str(image_id))
        # berry_json['image'].append({'id': image_id})

        for berry in berries:
            berry_id = str(berry[0])
            logger.debug('berry_id: ' + berry_id)

            # berry_info = {sample_id: ['image_id', {image_id: ['berry', {berry_id: ['fields', {}]}]}]}
            berry_info[sample_id][1][image_id][1].update({berry_id: ['fields', {}]})
            # berry_json['image']

            if berry.shape[0] == 7:
                weight = berry[1]
                logger.debug('berry_weight: ' + str(weight))
                berry_info[sample_id][1][image_id][1][berry_id][1].update({'weight': weight})
            berry_x, berry_y = berry[-2:]
            logger.debug('berry_x, berry_y: ' + str(berry_x) + ' ' + str(berry_y))
            for bbox, pred_mask in zip(outputs.pred_boxes.tensor.numpy(), outputs.pred_masks.numpy()):
                logger.debug('bbox: ' + str(bbox))
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                if x1 < berry_x < x2 and y1 < berry_y < y2:
                    # logger.info('bbox area: ' + str(bbox_area))
                    # cv2.imshow('bbox', im[int(y1):int(y2), int(x1):int(x2)])
                    # key = cv2.waitKey(0)
                    # if key == ord('q'):
                    #     cv2.destroyAllWindows()
                    # plt.imshow(im)
                    # plt.show()
                    bbox_area = (x2 - x1) * (y2 - y1)
                    logger.debug('bbox area: ' + str(bbox_area))
                    berry_info[sample_id][1][image_id][1][berry_id][1].update({'bbox': bbox})
                    berry_info[sample_id][1][image_id][1][berry_id][1].update({'pred_mask': pred_mask})
                    berry_info[sample_id][1][image_id][1][berry_id][1].update({'bbox_area': bbox_area})
                    berry_info[sample_id][1][image_id][1][berry_id][1].update({'polygon_area':
                                                                                  pred_mask.sum() / bbox_area})

                    avg_depth, rgb_seg, depth_inpaint = get_avg_depth(im, depth_file, bbox, pred_mask)
                    logger.debug('avg depth: ' + str(avg_depth))
                    #
                    berry_info[sample_id][1][image_id][1][berry_id][1].update({'rgb_seg': rgb_seg})
                    berry_info[sample_id][1][image_id][1][berry_id][1].update({'depth_inpaint': depth_inpaint})
            logger.debug('Current sample: ' + str(berry_info[sample_id][1][image_id][1][berry_id][1]))
        # logger.debug('berry info: ' + str(berry_info))
        # with
    dump_file = data_dir.as_posix() + '/dump_array_and_segments'
    logger.info('dump_file: ' + dump_file)
    np.save(dump_file, berry_info)
    # exit()

