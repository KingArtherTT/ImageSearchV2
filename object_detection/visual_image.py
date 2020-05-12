from object_detection import visualization
import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('F:/Competition/TaobaoDataset/validation_dataset_part1/image/000375/0.jpg')
    img = img[:, :, [2, 1, 0]]
    rclasses = np.array([15])
    rscores = np.array([0.05])
    rbboxes = np.array([[0.18899465, 0.6800534, 0.46416438, 0.7509859]])
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    print(rclasses, rscores, rbboxes)
