import numpy as np
#import cv2
from skimage import measure
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

dataset_dir = '/data/localdrive/StrawDI_Db1/train'
img_file_path = dataset_dir + '/img'
label_file_path = dataset_dir + '/label'
file_name = '/1.png'
img_file = img_file_path + file_name
label_file = label_file_path + file_name

label = cv2.imread(label_file)

# print(np.where(label == 1)[2])

# print(label[label == 1])

print('label shape: ', label.shape)
print('unique', np.unique(label))


def get_contour():

    for instance in np.unique(label)[1:]:
        print('instance: ', instance)
        bin_img = np.asarray(np.where(label != instance, 0, 255), dtype=np.float32)
        print('bin img shape: ', bin_img.shape)
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_RGB2GRAY)
        _, bin_img = cv2.threshold(bin_img, 250, 255, cv2.THRESH_BINARY)

        # # find the contours from the thresholded image
        # contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # draw all contours
        # image = cv2.drawContours(label, contours, -1, (0, 255, 0), 2)

        # Find contours at a constant value of 0.8
        contours = measure.find_contours(bin_img, 0.8)

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(bin_img, cmap=plt.cm.gray)

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

        # cv2.imshow('bin_img', bin_img)
        # cv2.waitKey(0)


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


if __name__ == '__main__':

    get_contour()
