import math

#import cv2
import numpy as np
import pathlib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger('my-app')
logger.setLevel(logging.INFO)
logging.basicConfig()


def my_python_metric(y_true, y_pred):
    result = np.abs(np.divide(np.subtract(y_true, y_pred), y_true))
    result = np.where(result < 0.2, 1, 0)
    return np.average(result)


img_dir = '/data/localdrive/robofruit_weightdataset/dataset'

berry_data = []
depth_data = []
cnn_training = []
for index, data_file in enumerate(pathlib.Path(img_dir).rglob('*dump_array_and_segments.npy')):
    # print('\r index: ', index, data_file)
    data = np.load(data_file.as_posix(), allow_pickle=True).item()
    # print(data.item()['weight'])
    logger.debug(data)
    for sample_id in data.keys():
        logger.debug('sample_id: ' + str(sample_id))
        for image_id in data[sample_id][1].keys():
            logger.debug('image_id: ' + str(image_id))
            bbox_area = polygon_area = width = height = weight = None
            for berry_id in data[sample_id][1][image_id][1].keys():
                logger.debug('berry_id: ' + str(berry_id))
                if 'bbox_area' in data[sample_id][1][image_id][1][berry_id][1].keys():
                    bbox_area = data[sample_id][1][image_id][1][berry_id][1]['bbox_area']
                    polygon_area = data[sample_id][1][image_id][1][berry_id][1]['polygon_area']
                    width = data[sample_id][1][image_id][1][berry_id][1]['bbox'][2] - \
                            data[sample_id][1][image_id][1][berry_id][1]['bbox'][0]
                    height = data[sample_id][1][image_id][1][berry_id][1]['bbox'][3] - \
                             data[sample_id][1][image_id][1][berry_id][1]['bbox'][1]
                    # print(data[sample_id][1][image_id][1][berry_id][1])
                    # exit()
                    # r = data[sample_id][1][image_id][1][berry_id][1]['avg_depth'][0]
                    # g = data[sample_id][1][image_id][1][berry_id][1]['avg_depth'][1]
                    # b = data[sample_id][1][image_id][1][berry_id][1]['avg_depth'][2]
                    rgb_seg = data[sample_id][1][image_id][1][berry_id][1]['rgb_seg']
                    depth_inpaint = data[sample_id][1][image_id][1][berry_id][1]['depth_inpaint']
                    for img_id in data[sample_id][1].keys():
                        if berry_id in data[sample_id][1][img_id][1].keys():
                            if 'weight' in data[sample_id][1][img_id][1][berry_id][1].keys():
                                weight = data[sample_id][1][img_id][1][berry_id][1]['weight']

                    if bbox_area and polygon_area and width and height and weight is not None:

                        # if not (math.isnan(bbox_area) and math.isnan(polygon_area) and math.isnan(width) and math.isnan(height) and math.isnan(weight)):
                        # if math.isnan(bbox_area) or math.isnan(polygon_area) or math.isnan(
                        #             width) or math.isnan(height) or math.isnan(weight):
                        # print(sample_id, image_id, berry_id)
                        # if math.isnan(r) or math.isnan(g) or math.isnan(b):
                        #     continue
                        berry_data.append((weight, bbox_area, polygon_area, width, height, float(image_id),
                                           float(berry_id)))
                        # cv2.imshow('depth_inpaint: ', depth_inpaint[bb])
                        depth_data.append(depth_inpaint)
                        logger.debug('berry_data: ' + str(berry_data[-1]))

# print(berry_data[0])

berry_data = np.asarray(berry_data, dtype=np.float32)
# berry_data = berry_data[np.where(berry_data[:, -2] == 1)]
# berry_data = berry_data[np.where(berry_data[:, -1] == 1)]
X = berry_data[:, 1:5]
y = berry_data[:, 0]

train_idx = np.random.randint(0, len(berry_data), int(len(berry_data) * 0.8))
train_X = np.take(X, train_idx, axis=0)
train_y = np.take(y, train_idx)
val_X = np.delete(X, train_idx, axis=0)
val_y = np.delete(y, train_idx)
logger.info('train_X val_X shape: ' + str(train_X.shape) + ' ' + str(val_X.shape))

# depth_data = depth_data[np.where(berry_data[:, -2] == 1)]
# depth_data = depth_data[np.where(berry_data[:, -1] == 1)]
train_depth_X = np.take(depth_data, train_idx, axis=0)
val_depth_X = np.delete(depth_data, train_idx, axis=0)
logger.info('train_depth_X val_depth_X shape: ' + str(train_depth_X.shape) + ' ' + str(val_depth_X.shape))

pca = PCA(n_components=3)
train_depth_X = pca.fit_transform(train_depth_X.reshape(train_depth_X.shape[0], -1))
val_depth_X = pca.transform(val_depth_X.reshape(val_depth_X.shape[0], -1))
logger.info('After PCA train_depth_X val_depth_X shape: ' + str(train_depth_X.shape) + ' ' + str(val_depth_X.shape))

train_X = np.hstack((train_X, train_depth_X))
val_X = np.hstack((val_X, val_depth_X))
#
# # regr = RandomForestRegressor(max_depth=2, random_state=0)
# # mm_scaler = preprocessing.RobustScaler()
# # #
# # train_X = mm_scaler.fit_transform(train_X)
# # # train_y = mm_scaler.fit_transform(train_y)
# # val_X = mm_scaler.transform(val_X)
# # # val_y = mm_scaler.transform(val_y)
# #
# # regr.fit(train_X, train_y)
# # pred_y = regr.predict(val_X)
# #
# # print('result: ', my_python_metric(val_y, pred_y))
#
#
# # print(X.shape, y.shape)
# # print(np.take(X, train_idx, axis=0).shape)
for n in range(-0, 5):
    regr = make_pipeline(StandardScaler(), SVR(kernel='linear', C=10**n, epsilon=0.2))

    regr.fit(train_X, train_y)
    pred_y = regr.predict(val_X)

    # print(np.vstack((y_true, y_pred)))

    print('result: ', my_python_metric(val_y, pred_y))

