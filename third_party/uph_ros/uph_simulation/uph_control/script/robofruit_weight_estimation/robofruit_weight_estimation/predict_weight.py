import numpy as np
from inference_method import prediction_live
# from process_depth import process_depth_live
from joblib import dump, load
#import cv2


MODEL_FILE_MULTI = '/home/robofruit/uph_franka_ws/src/franka_uph_project/uph_simulation/uph_control/script/robofruit_weight_estimation/robofruit_weight_estimation/multi_distance.joblib'
MODEL_FILE_SINGLE = '/home/robofruit/uph_franka_ws/src/franka_uph_project/uph_simulation/uph_control/script/robofruit_weight_estimation/robofruit_weight_estimation/single_berry_single_distance.joblib'
regr_multi = load(MODEL_FILE_MULTI)
regr_single = load(MODEL_FILE_SINGLE)

def predict_weight(colour_image, depth_image):

    outputs, bbox_image = prediction_live(colour_image)
    to_panda = []

    for bbox, pred_mask, score in zip(outputs.pred_boxes.tensor.numpy(),
                                       outputs.pred_masks.numpy(),
                                       outputs.scores.numpy()):
        if score < 0.75:
            continue


        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        bbox_area = (x2 - x1) * (y2 - y1)
        polygon_area = pred_mask.sum() / bbox_area

        # print(raw_depth_filtered.shape)

        if raw_depth_filtered.shape[0] == 0:
            print('depth not detected below 50')
            to_panda.append((bbox, 0, 0, 0))
        else:

            # min_raw_depth = np.min(raw_depth_filtered)
            # max_raw_depth = np.max(raw_depth_filtered)
            # average_raw_depth = np.average(raw_depth_filtered)

            # if (average_raw_depth < 22.0) & (min_raw_depth > 17.0):
            #     berry_data = (bbox_area, polygon_area, min_raw_depth, max_raw_depth, average_raw_depth)
            #     berry_data = np.asarray(berry_data, dtype=np.float32)

            #     display_text = 'M: {:.2f} S: {:.2f}'.format(
            #         regr_multi.predict(berry_data.reshape(1, -1))[0],
            #         regr_single.predict(berry_data.reshape(1, -1))[0])

            #     bbox_image = np.ascontiguousarray(bbox_image, dtype=np.uint8)
 
            #     print('Berry weight multi: {:.2f} single: {:.2f}'.format(
            #         regr_multi.predict(berry_data.reshape(1, -1))[0],
            #         regr_single.predict(berry_data.reshape(1, -1))[0]
            #     ))

            to_panda.append((bbox, 0, 0, 0))
 
    return to_panda
 


