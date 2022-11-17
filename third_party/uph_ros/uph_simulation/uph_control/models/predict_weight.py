import numpy as np
# from inference_method import prediction_live
from inference_picking_point import prediction_live
from process_depth import process_depth_live
from joblib import dump, load
import cv2

from rgb_to_pointcloud import pc_from_rgbd

MODEL_FILE_MULTI = 'multi_distance.joblib'
MODEL_FILE_SINGLE = 'single_berry_single_distance.joblib'
regr_multi = load(MODEL_FILE_MULTI)
regr_single = load(MODEL_FILE_SINGLE)


def predict_weight(colour_image, depth_image):

    outputs, bbox_image = prediction_live(colour_image)
    to_panda = []

    for bbox, pred_mask, score, pluck, keypoints, in zip(outputs.pred_boxes.tensor.numpy(),
                                      outputs.pred_masks.numpy(),
                                      outputs.scores.numpy(),
                                      outputs.pred_classes.numpy(),
                                      outputs.pred_keypoints.numpy()):
        if pluck:
            continue
        if score < 0.90:
            continue

        rgb_image_filtered, raw_depth_scaled, raw_depth_mask = \
            process_depth_live(colour_image, depth_image, bbox, pred_mask)

        # pcd_angles = pc_from_rgbd(rgb_image_filtered, raw_depth_scaled, raw_depth_mask, keypoints)

        raw_depth_scaled = raw_depth_scaled * raw_depth_mask[:, :, 0]
        raw_depth_filtered = raw_depth_scaled[raw_depth_scaled > 0]

        # std_dev_raw_depth = np.std(raw_depth_filtered)

        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        bbox_area = (x2 - x1) * (y2 - y1)
        polygon_area = pred_mask.sum() / bbox_area

        # print(raw_depth_filtered.shape)

        if raw_depth_filtered.shape[0] == 0:
            # print('depth not detected below 50')

            keypoints = np.column_stack((keypoints, np.zeros(5)))
            # print(keypoints)
            to_panda.append(keypoints)

        else:
            min_raw_depth = np.min(raw_depth_filtered)
            max_raw_depth = np.max(raw_depth_filtered)
            average_raw_depth = np.average(raw_depth_filtered)

            if (average_raw_depth < 22.0) & (min_raw_depth > 17.0):
                berry_data = (bbox_area, polygon_area, min_raw_depth, max_raw_depth, average_raw_depth)
                berry_data = np.asarray(berry_data, dtype=np.float32)

                display_text = 'M: {:.2f} S: {:.2f}'.format(
                    regr_multi.predict(berry_data.reshape(1, -1))[0],
                    regr_single.predict(berry_data.reshape(1, -1))[0])

                bbox_image = np.ascontiguousarray(bbox_image, dtype=np.uint8)

                cv2.putText(bbox_image, display_text, (int(bbox[0]), int(bbox[1])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, color=(255, 255, 0),
                            thickness=2, lineType=cv2.LINE_AA)

                print('Berry weight multi: {:.2f} single: {:.2f}'.format(
                    regr_multi.predict(berry_data.reshape(1, -1))[0],
                    regr_single.predict(berry_data.reshape(1, -1))[0]
                ))

            average_raw_depth = np.ones(5) * average_raw_depth
            # print(keypoints)
            # keypoints[:, 2] = average_raw_depth
            keypoints = np.column_stack((keypoints, average_raw_depth))
            # to_panda.append((bbox, min_raw_depth, max_raw_depth, average_raw_depth))
            to_panda.append(keypoints)

    cv2.imshow('inference', bbox_image)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    return to_panda

    # if len(to_panda) == 0:
    #     print('Berry not detected')
    #     return 0
    # else:
    #     print('Berry detected: ', to_panda)
    #     return to_panda



