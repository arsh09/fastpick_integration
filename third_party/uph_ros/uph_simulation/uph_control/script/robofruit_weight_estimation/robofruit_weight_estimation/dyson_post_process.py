import pathlib

#import cv2
from tqdm import tqdm
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
from joblib import dump, load

logger = logging.getLogger('my-app')
logger.setLevel(logging.INFO)
logging.basicConfig()


data_dir = '/home/rick_robofruit/localdrive/robofruit_weightdataset/dataset'
MODEL_FILE = 'multi_distance.joblib'
# MODEL_FILE = 'single_berry_single_distance.joblib'
# accuracy = 0.20
# accuracy = 0.25
# accuracy = 0.30
ACCURACY = [0.1, 0.15, 0.2, 0.25, 0.3]


def my_python_metric(y_true, y_pred, accuracy=0.2):

    result_list = []
    for accuracy in ACCURACY:
        result_calc = np.abs(np.divide(np.subtract(y_true, y_pred), y_true))
        result_calc = np.where(result_calc < accuracy, 1, 0)
        result_list.append(np.average(result_calc))

    return result_list


berry_data = []
raw_depth_data = []
for dump_file in tqdm(pathlib.Path(data_dir).rglob('*_dump.npy')):

    data = np.load(dump_file.as_posix(), allow_pickle=True).item()
    # print('filename: ', dump_file)
    for berry in data['annotations']:
        berry_id = berry['berry_id']
        weight = None
        for dimension in data['dimensions']:
            if dimension['berry_id'] == berry_id:
                weight = dimension['weight']

        if weight is None:
            print('weight none sample id: ', data['sample'])
            continue

        if weight == 0:
            print('weight is zero: ', data['sample'])
            continue

        width = berry['bbox'][2] - berry['bbox'][0]
        height = berry['bbox'][3] - berry['bbox'][1]

        rgb_seg = berry['rgb_seg']
        depth_inpaint = berry['depth_seg_inpainted']
        raw_depth_mask = berry['raw_depth_mask']
        raw_depth = berry['raw_depth']

        if rgb_seg.shape != (480, 640, 3):
            logger.info('rgb shape: ' + str(rgb_seg.shape))

        raw_depth = raw_depth * raw_depth_mask[:, :, 0]
        raw_depth_filtered = raw_depth[raw_depth > 0]

        if raw_depth_filtered.shape[0] == 0:
            print('raw depth is zero: ', data['sample'])
            continue

        min_raw_depth = np.min(raw_depth_filtered)
        max_raw_depth = np.max(raw_depth_filtered)
        average_raw_depth = np.average(raw_depth_filtered)
        std_dev_raw_depth = np.std(raw_depth_filtered)
        # print('raw depth min max avg std', min_raw_depth,
        #       max_raw_depth,
        #       average_raw_depth,
        #       std_dev_raw_depth)
        # # exit()

        # cv2.rectangle(depth_inpaint,
        #               (berry['bbox'][0], berry['bbox'][1]),
        #               (berry['bbox'][2], berry['bbox'][3]),
        #               color=(255, 0, 0))
        # raw_depth_mask = np.asarray(np.where(raw_depth_mask == True, 255, 0), dtype=np.float32)
        # image = np.hstack((rgb_seg, raw_depth_mask, depth_inpaint))
        # cv2.imshow('raw_depth_mask', np.asarray(image, dtype=np.float32))
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        # elif key == ord('n'):
        #     continue

        berry_data.append((weight,
                           berry['bbox_area'],
                           berry['polygon_area'],
                           min_raw_depth,
                           max_raw_depth,
                           average_raw_depth,
                           # std_dev_raw_depth,
                           width,
                           height,
                           berry['image_id'],
                           berry['berry_id']))

# exit()
berry_data = np.asarray(berry_data, dtype=np.float32)

# Select berry id 1 and image id 1
# berry_data = berry_data[np.where(berry_data[:, -2] == 1)]
# berry_data = berry_data[np.where(berry_data[:, -1] == 1)]

# Select berry id 1 and image id 1, 2
berry_data_1 = berry_data[np.where(berry_data[:, -2] == 1)]
berry_data_2 = berry_data[np.where(berry_data[:, -2] == 2)]
berry_data = np.vstack((berry_data_1, berry_data_2))
# berry_data = berry_data[np.where(berry_data[:, -1] == 1)]

X = berry_data[:, 1:-4]
y = berry_data[:, 0]
print('total data size: ', X.shape)

best_model = 0
for i in range(0, 1):

        # train_idx = np.random.randint(0, len(berry_data), int(len(berry_data) * 0.8))
        train_idx = np.random.choice(range(len(berry_data)), int(len(berry_data) * 0.8), replace=False)
        train_X = np.take(X, train_idx, axis=0)
        train_y = np.take(y, train_idx)
        val_X = np.delete(X, train_idx, axis=0)
        val_y = np.delete(y, train_idx)
        logger.info('train idx, val idx: ' + str(train_X.shape) + ' ' + str(val_X.shape))

        result = []
        for n in range(0, 5):
            regr = make_pipeline(RobustScaler(), SVR(kernel='linear', C=10**n, epsilon=0.2))

            regr.fit(train_X, train_y)
            pred_y = regr.predict(val_X)

            # print(np.vstack((y_true, y_pred)))

            # print('result: ', my_python_metric(val_y, pred_y))

            # result.append(my_python_metric(val_y, pred_y))
            result = np.asarray(my_python_metric(val_y, pred_y), dtype=np.float32)[2]
            print('result: ', result * 100)

            if result > best_model:
                best_model = result
                # dump_file = open(MODEL_FILE, 'w')
                # pickle.dump(regr, dump_file)
                # dump_file.close()
                print('best model saved: ', best_model)
                dump(regr, MODEL_FILE)

        # regr = load(MODEL_FILE)
        # pred_y = regr.predict(val_X)
        # print(print(my_python_metric(val_y, pred_y)))
        # for pred, val in zip(pred_y, val_y):
        #     print('pred {:.2f}, val {:.2f}'.format(pred, val))

        # result = np.asarray(result, dtype=np.float32)
        # result = np.around(np.sort(result, axis=1)[-1] * 100, decimals=2)
        # print('result: ', result)

