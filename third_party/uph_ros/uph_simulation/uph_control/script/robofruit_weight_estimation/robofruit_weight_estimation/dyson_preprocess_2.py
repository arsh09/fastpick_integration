import os
import re

#import cv2
import numpy as np
import pathlib
import logging
import json
from json import JSONEncoder

from inference_method import prediction
from process_depth import process_depth

logger = logging.getLogger('my-app')
logger.setLevel(logging.INFO)
logging.basicConfig()

img_dir = '/home/rick_robofruit/localdrive/robofruit_weightdataset/dataset/'

for data_dir in pathlib.Path(img_dir).glob('*/*'):

    logger.debug('label data_dir: ' + data_dir.as_posix())
    collector = data_dir.as_posix().split('/')[-2]
    logger.debug('collector: ' + collector)
    folder_no = data_dir.as_posix().split('/')[-1]
    logger.debug('folder_no: ' + folder_no)
    sample_id = collector + '_' + folder_no
    logger.debug('sample_id: ' + sample_id)

    # berry_info = {sample_id: ['image_id', {}]}
    berry_json = {'sample': sample_id,
                  'dimensions': [],
                  'annotations': [],
                  }

    for label_file in pathlib.Path(data_dir).glob('*_label.npy'):

        logger.info('label filename: ' + label_file.as_posix())

        weight = None
        berry_x, berry_y = 0, 0

        berries = np.load(label_file.as_posix())
        image_id = re.search('_([0-9])_label', label_file.as_posix())[1]
        logger.debug('image id: ' + image_id)
        rgb_file = re.sub('_label.npy', '_rgb.png', label_file.as_posix())
        r_depth_file = re.sub('_label.npy', '_rdepth.npy', label_file.as_posix())
        p_depth_file = re.sub('_label.npy', '_pdepth.png', label_file.as_posix())
        outputs, im = prediction(rgb_file)

        for berry in berries:
            berry_id = str(berry[0])
            logger.info('berry_id: ' + str(berry))

            if berry.shape[0] == 7:
                weight = berry[1]
                height = berry[2]
                width = berry[3]
                depth = berry[4]
                logger.debug('berry_weight: ' + str(weight))
                berry_json['dimensions'].append({'berry_id': berry_id,
                                                 'weight': weight,
                                                 'height': height,
                                                 'depth': depth
                                                 })
                # print(berry_json['dimensions'])
            berry_x, berry_y = berry[-2:]
            logger.debug('berry_x, berry_y: ' + str(berry_x) + ' ' + str(berry_y))
            for bbox, pred_mask in zip(outputs.pred_boxes.tensor.numpy(), outputs.pred_masks.numpy()):
                logger.debug('bbox: ' + str(bbox))
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                if x1 < berry_x < x2 and y1 < berry_y < y2:
                    rgb_seg, depth_inpaint, rgb_bbox, colourised_depth, raw_depth, raw_depth_mask = \
                        process_depth(im, p_depth_file, r_depth_file, bbox,  pred_mask)
                    # rgb_bbox = cv2.drawMarker(rgb_bbox, (berry_x, berry_y), color=(0, 255, 0))
                    # cv2.imshow('collage', np.vstack((np.hstack((rgb_bbox, rgb_seg)),
                    #                                 np.hstack((colourised_depth, depth_inpaint)))))
                    # key = cv2.waitKey(0)
                    # if key == ord('q'):
                    #     cv2.destroyAllWindows()
                    #     exit()
                    # elif key == ord('n'):
                    #     continue

                    bbox_area = (x2 - x1) * (y2 - y1)
                    logger.debug('bbox area: ' + str(bbox_area))
                    berry_json['annotations'].append({'berry_id': berry_id,
                                                      'image_id': image_id,
                                                      'bbox': bbox,
                                                      'pred_mask': pred_mask,
                                                      'bbox_area': bbox_area,
                                                      'polygon_area': pred_mask.sum() / bbox_area,
                                                      'rgb_seg': rgb_seg,
                                                      'depth_seg_inpainted': depth_inpaint,
                                                      'raw_depth': raw_depth,
                                                      'raw_depth_mask': raw_depth_mask,
                                                      'colourised_depth': colourised_depth,
                                                      })

            logger.debug('Current sample: ' + str(berry_json))
    dump_file = data_dir.as_posix() + '/' + sample_id + '_dump'
    logger.info('dump_file: ' + dump_file)
    logger.debug('Current sample: ' + str(berry_json))
    # json.dumps(berry_json)
    np.save(dump_file, berry_json)
    # exit()

