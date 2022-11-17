import os
import re
import sys

#import cv2
import numpy as np
import pathlib
import logging
import pyrealsense2 as rs
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

from inference_method import prediction

# sys.path.append('/data/lincoln/generative_inpainting/')
# from inpainting_depth import inpainting_depth

logger = logging.getLogger('my-app-2')
logger.setLevel(logging.INFO)
logging.basicConfig()


def process_depth(rgb_image, colourised_depth_image, raw_depth, bbox, pred_mask):

    raw_depth = np.load(raw_depth) * 0.1
    logger.debug('raw_depth shape: ' + str(raw_depth.shape))

    # print(bbox)
    # print(rgb_image.shape)
    # rgb_image_bbox = np.copy(rgb_image)
    # rgb_image_bbox = cv2.rectangle(rgb_image_bbox, (bbox[0], bbox[1]),
    #                                (bbox[2], bbox[3]), (255, 255, 255), 5)
    rgb_image_bbox = 0

    raw_depth_mask = raw_depth * pred_mask
    raw_depth_mask = np.where(raw_depth_mask < 14, 0, raw_depth_mask)
    raw_depth_mask = np.where(raw_depth_mask > 30, 0, raw_depth_mask)
    raw_depth_mask = np.where(raw_depth_mask > 0, True, False)

    raw_depth_mask = np.dstack((raw_depth_mask, raw_depth_mask, raw_depth_mask))
    # raw_depth_mask = np.asarray(raw_depth_mask, dtype=np.float32)
    logger.debug('raw_depth shape: ' + str(raw_depth_mask.shape))

    pred_mask = pred_mask.reshape(480, 640)

    colourised_depth_image = cv2.imread(colourised_depth_image)
    # colourised_depth_image_bbox = np.copy(colourised_depth_image)
    # colourised_depth_image_bbox = cv2.rectangle(colourised_depth_image_bbox, (bbox[0], bbox[1]),
    #                                             (bbox[2], bbox[3]), (255, 255, 255), 5)

    # imshow_1 = np.hstack((rgb_image_bbox, colourised_depth_image_bbox))
    # original colourised depth image
    # cv2.imshow('colourised_depth_image_org', colourised_depth_image)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    pred_mask = np.dstack((pred_mask, pred_mask, pred_mask))
    # colourised_depth_image = colourised_depth_image * pred_mask
    colourised_depth_image = colourised_depth_image * raw_depth_mask
    rgb_image = rgb_image * pred_mask

    # # Display depth with mask
    # cv2.imshow('colourised_depth_image_org_times_mask', colourised_depth_image)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    depth_mask = cv2.cvtColor(colourised_depth_image, cv2.COLOR_RGB2GRAY)
    _, depth_mask = cv2.threshold(depth_mask, 10, 255, cv2.THRESH_BINARY_INV)
    depth_mask = depth_mask * pred_mask[:, :, 0]
    depth_mask =np.dstack((depth_mask, depth_mask, depth_mask))

    # cv2.imshow('depth_mask', np.hstack((colourised_depth_image, depth_mask)))
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    # exit()
    #
    # depth_inpaint = cv2.cvtColor(inpainting_depth(colourised_depth_image, depth_mask), cv2.COLOR_BGR2RGB)
    depth_inpaint = 'None'

    # just display
    # print(colourised_depth_image.shape, depth_mask.shape, depth_inpaint.shape)
    # cv2.imshow('depth_mask', np.hstack((colourised_depth_image, depth_mask,
    #                                     cv2.cvtColor(depth_inpaint, cv2.COLOR_BGR2RGB))))
    # imshow_1 = np.hstack((rgb_image, colourised_depth_image))
    # imshow_2 = np.hstack((depth_mask, depth_inpaint))
    # imshow_2 = np.hstack((colourised_depth_image, depth_inpaint))
    # r = depth_inpaint[:, :, 0]
    # g = depth_inpaint[:, :, 1]
    # b = depth_inpaint[:, :, 2]
    # r = np.average(r[r > 50])
    # g = np.average(g[g > 50])
    # b = np.average(b[b > 50])

    # pca = PCA(n_components=3)

    #  Just to view
    # cv2.imshow('collage', np.vstack((imshow_1, imshow_2)))
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit(0)
    # elif key == ord('w'):
    #     cv2.destroyAllWindows()

    return rgb_image, depth_inpaint, rgb_image_bbox, colourised_depth_image, raw_depth, raw_depth_mask

    # histogram
    # plt.hist(np.asarray(cv2.cvtColor(depth_inpaint, cv2.COLOR_BGR2RGB)).reshape(-1), bins=50)
    # plt.show()

    # exit()


def process_depth_live(rgb_image, raw_depth, bbox, pred_mask):

    # raw_depth = np.load(raw_depth) * 0.1
    raw_depth = raw_depth * 0.1
    logger.debug('raw_depth shape: ' + str(raw_depth.shape))

    # rgb_image_bbox = np.copy(rgb_image)
    # rgb_image_bbox = cv2.rectangle(rgb_image_bbox, (bbox[0], bbox[1]),
    #                                (bbox[2], bbox[3]), (255, 255, 255), 5)

    raw_depth_mask = raw_depth * pred_mask
    raw_depth_mask = np.where(raw_depth_mask < 14, 0, raw_depth_mask)
    raw_depth_mask = np.where(raw_depth_mask > 50, 0, raw_depth_mask)
    raw_depth_mask = np.where(raw_depth_mask > 0, True, False)

    raw_depth_mask = np.dstack((raw_depth_mask, raw_depth_mask, raw_depth_mask))
    # raw_depth_mask = np.asarray(raw_depth_mask, dtype=np.float32)
    logger.debug('raw_depth shape: ' + str(raw_depth_mask.shape))

    pred_mask = pred_mask.reshape(480, 640)

    # colourised_depth_image = cv2.imread(colourised_depth_image)
    # colourised_depth_image_bbox = np.copy(colourised_depth_image)
    # colourised_depth_image_bbox = cv2.rectangle(colourised_depth_image_bbox, (bbox[0], bbox[1]),
    #                                             (bbox[2], bbox[3]), (255, 255, 255), 5)

    # imshow_1 = np.hstack((rgb_image_bbox, colourised_depth_image_bbox))
    # original colourised depth image
    # cv2.imshow('colourised_depth_image_org', colourised_depth_image)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    pred_mask = np.dstack((pred_mask, pred_mask, pred_mask))
    # colourised_depth_image = colourised_depth_image * pred_mask
    # colourised_depth_image = colourised_depth_image * raw_depth_mask
    rgb_image_filtered = rgb_image * pred_mask

    # # Display depth with mask
    # cv2.imshow('colourised_depth_image_org_times_mask', colourised_depth_image)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    # depth_mask = cv2.cvtColor(colourised_depth_image, cv2.COLOR_RGB2GRAY)
    # _, depth_mask = cv2.threshold(depth_mask, 10, 255, cv2.THRESH_BINARY_INV)
    # depth_mask = depth_mask * pred_mask[:, :, 0]
    # depth_mask =np.dstack((depth_mask, depth_mask, depth_mask))

    # cv2.imshow('depth_mask', np.hstack((colourised_depth_image, depth_mask)))
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    # exit()
    #
    # depth_inpaint = cv2.cvtColor(inpainting_depth(colourised_depth_image, depth_mask), cv2.COLOR_BGR2RGB)
    # depth_inpaint = 'None'

    # just display
    # print(colourised_depth_image.shape, depth_mask.shape, depth_inpaint.shape)
    # cv2.imshow('depth_mask', np.hstack((colourised_depth_image, depth_mask,
    #                                     cv2.cvtColor(depth_inpaint, cv2.COLOR_BGR2RGB))))
    # imshow_1 = np.hstack((rgb_image, colourised_depth_image))
    # imshow_2 = np.hstack((depth_mask, depth_inpaint))
    # imshow_2 = np.hstack((colourised_depth_image, depth_inpaint))
    # r = depth_inpaint[:, :, 0]
    # g = depth_inpaint[:, :, 1]
    # b = depth_inpaint[:, :, 2]
    # r = np.average(r[r > 50])
    # g = np.average(g[g > 50])
    # b = np.average(b[b > 50])

    # pca = PCA(n_components=3)

    #  Just to view
    # cv2.imshow('collage', np.vstack((imshow_1, imshow_2)))
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit(0)
    # elif key == ord('w'):
    #     cv2.destroyAllWindows()

    return rgb_image_filtered, raw_depth, raw_depth_mask

    # histogram
    # plt.hist(np.asarray(cv2.cvtColor(depth_inpaint, cv2.COLOR_BGR2RGB)).reshape(-1), bins=50)
    # plt.show()

    # exit()