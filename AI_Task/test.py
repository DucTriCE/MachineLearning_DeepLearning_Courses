import sklearn.metrics
import torch
import numpy as np


def get_iou(ground_truth, pred):

    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou
ground_truth_bbox = torch.rand(8,2,4,4)
prediction_bbox = torch.rand(8,2,4,4)
# grund_truth_bbox = ground_truth_bbox.flatten()
# prediction_bbox= prediction_bbox.flatten()
# print(ground_truth_bbox.shape)
# ground_truth_bbox = np.array([1202, 123, 1650, 868], dtype=np.float32)
# prediction_bbox = np.array([1162.0001, 92.0021, 1619.9832, 694.0033], dtype=np.float32)

iou = get_iou(ground_truth_bbox, prediction_bbox)
print('IOU: ', iou)