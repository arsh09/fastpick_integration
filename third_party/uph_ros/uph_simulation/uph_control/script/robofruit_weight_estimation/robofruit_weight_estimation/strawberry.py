# Some basic setup:
import pathlib
import re
from tqdm import tqdm
from matplotlib import pyplot as plt

import detectron2
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode


def get_strawberry_dicts_org(img_dir):

    dataset_dicts = []
    for idx, image_file in tqdm(enumerate(pathlib.Path(img_dir + '/img/').rglob('*.png'))):
        # if idx < 885:
        #     continue
        # if '1906' not in str(image_file):
        #     continue
        # if '1054' not in str(image_file):
        #     continue

        # print('image file: ', image_file.absolute())
        record = {}

        label = cv2.imread(re.sub('img', 'label', str(image_file)))
        img = cv2.imread(str(image_file))
        height, width = img.shape[:2]

        # cv2.imshow('original_file', img)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     break
        # else:
        #     pass

        record["file_name"] = str(image_file)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for instance in np.unique(label)[1:]:
            # print('instance: ', instance)
            bin_img = np.asarray(np.where(label != instance, 0, 255), dtype=np.float32)
            # print('bin img shape: ', bin_img.shape)
            bin_img = cv2.cvtColor(bin_img, cv2.COLOR_RGB2GRAY)
            _, bin_img = cv2.threshold(bin_img, 250, 255, cv2.THRESH_BINARY)
            cv2.imshow('bin_img', bin_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            else:
                pass
            cv2.destroyWindow('bin_img')

            # contours = measure.find_contours(bin_img, 0.8)
            # print((np.max(bin_img) - np.min(bin_img))/2, np.max(bin_img), np.min(bin_img))
            contours = measure.find_contours(bin_img)
            if len(contours) > 2:
                print(contours)
                print('file skipped: ', image_file.absolute())
                # continue

            # print(contours)
            px, py = contours[0][:, 1], contours[0][:, 0]
            # print('px, py: ', px, py)
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # print(poly)
            poly = [p for x in poly for p in x]
            # print(poly)
            # print('poly: ', len(poly))
            # exit()

            # Display the image and plot all contours found

            for contour in contours:

                fig, ax = plt.subplots()
                ax.imshow(bin_img, cmap=plt.cm.gray)

                ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        # if idx > 925:
        #     break
    return dataset_dicts


def get_strawberry_dicts(img_dir):

    dataset_dicts = []
    for idx, image_file in tqdm(enumerate(pathlib.Path(img_dir + '/img/').rglob('*.png'))):
        # if idx < 885:
        #     continue
        # if '1906' not in str(image_file):
        #     continue
        # if '1054' not in str(image_file):
        #     continue
        # 1768 1304
        # if not '1768' in str(image_file):
        #     continue

        # print('image file: ', image_file.absolute())
        record = {}

        label = cv2.imread(re.sub('img', 'label', str(image_file)))
        img = cv2.imread(str(image_file))
        height, width = img.shape[:2]

        cv2.imshow('original_file', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        else:
            pass

        record["file_name"] = str(image_file)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        print('hi')

        objs = []
        for instance in np.unique(label)[1:]:
            print('instance: ', instance)
            bin_img = np.asarray(np.where(label != instance, 0, 255), dtype=np.float32)
            # print('bin img shape: ', bin_img.shape)
            bin_img = cv2.cvtColor(bin_img, cv2.COLOR_RGB2GRAY)
            _, bin_img = cv2.threshold(bin_img, 250, 255, cv2.THRESH_BINARY)

            cv2.imshow('bin_img', bin_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            else:
                pass
            cv2.destroyWindow('bin_img')

            # contours = measure.find_contours(bin_img, 0.8)
            # print((np.max(bin_img) - np.min(bin_img))/2, np.max(bin_img), np.min(bin_img))
            contours = measure.find_contours(bin_img, positive_orientation='low')

            segmentations = []
            polygons = []
            for idx, contour in enumerate(contours):
                # print('contour idx: ', idx)
                # Flip from (row, col) representation to (x, y)
                # and subtract the padding pixel
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Make a polygon and simplify it
                if contour.shape[0] < 3:
                    continue
                poly = Polygon(contour)
                # print('poly: ', poly)
                # poly = poly.simplify(1.0, preserve_topology=False)
                polygons.append(poly)
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)

            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
            bbox = multi_poly.bounds
            # x, y, max_x, max_y = multi_poly.bounds
            # width = max_x - x
            # height = max_y - y
            # bbox = (x, y, width, height)
            # area = multi_poly.area
            # Display the image and plot all contours found

            # for contour in contours:
            #
            #     fig, ax = plt.subplots()
            #     ax.imshow(bin_img, cmap=plt.cm.gray)
            #
            #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
            #
            #     ax.axis('image')
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     plt.show()

            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": segmentations,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        # if idx > 925:
        #     break
    return dataset_dicts


data_dir = "/data/localdrive/StrawDI_Db1/"
working_dir = "/data/localdrive/myprojects/random/"
os.makedirs(working_dir, exist_ok=True)

with open(working_dir + "/val", "w") as write_file:
    print('hi')
    json.dump(get_strawberry_dicts(img_dir=data_dir + "/val"), write_file)
write_file.close()
with open(working_dir + "/train", "w") as write_file:
    json.dump(get_strawberry_dicts(img_dir=data_dir + "/train"), write_file)
write_file.close()

# dataset_dicts = get_strawberry_dicts(data_dir + "/train")
# # for d in random.sample(dataset_dicts, 10):
# for d in dataset_dicts:
#     print(d["file_name"])
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=strawberry_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('annotations', out.get_image()[:, :, ::-1])
#     key = cv2.waitKey(0)
#     if key == ord('q'):
#         break
#     else:
#         continue
#
# exit()


