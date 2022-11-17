import numpy as np
import pathlib
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


def my_python_metric(y_true, y_pred):
    result = np.abs(np.divide(np.subtract(y_true, y_pred), y_true))
    result = np.where(result < 0.2, 1, 0)
    return np.average(result)

img_dir = '/data/localdrive/dyson/raw'

berry_data = []
for index, data_file in enumerate(pathlib.Path(img_dir).rglob('*_1_detectron.npy')):
    # print('\r index: ', index, data_file)
    data = np.load(data_file.as_posix(), allow_pickle=True).item()
    # print(data.item()['weight'])
    berry_data.append((data['weight'],
                       data['bbox_area'],
                       data['bbox'][2] - data['bbox'][0],
                       data['bbox'][3] - data['bbox'][1],
                       data['polygon_area']))


berry_data = np.asarray(berry_data, dtype=np.float32)
X = berry_data[:, 1:]
y = berry_data[:, 0]

train_idx = np.random.randint(0, berry_data.shape[0], int(berry_data.shape[0] * 0.8))
train_X = np.take(X, train_idx, axis=0)
train_y = np.take(y, train_idx)
val_X = np.delete(X, train_idx, axis=0)
val_y = np.delete(y, train_idx)

regr = RandomForestRegressor(max_depth=4, random_state=0)
mm_scaler = preprocessing.MinMaxScaler()

train_X = mm_scaler.fit_transform(train_X)
# train_y = mm_scaler.fit_transform(train_y)
val_X = mm_scaler.transform(val_X)
# val_y = mm_scaler.transform(val_y)

# print(X.shape, y.shape)

# print(np.take(X, train_idx, axis=0).shape)

regr.fit(train_X, train_y)
pred_y = regr.predict(val_X)

# print(np.vstack((y_true, y_pred)))

print('result: ', my_python_metric(val_y, pred_y))

