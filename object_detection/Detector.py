import os
import math
import random
from torch.autograd import Variable
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2

from nets.ssd_detector.ssd import build_ssd

labelmap = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmo#nitor')
MEANS = (104, 117, 123)


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return base_transform(image, self.size, self.mean)


class Detector(object):
    def __init__(self, weights):
        # load net
        num_classes = len(labelmap) + 1  # +1 for background
        self.net = build_ssd('test', 300, num_classes)  # initialize SSD
        self.net.load_weights(weights)
        self.net.eval()
        self.net = self.net.cuda()
        cudnn.benchmark = True

    def get_deal_result(self, imgs, transform=BaseTransform(300, MEANS)):
        height_list = []
        width_list = []
        dealed_img_list = []
        result_bboxes = []
        result_scores = []
        for i in range(len(imgs)):
            # imgs[i] = imgs[i].numpy()
            h, w, c = imgs[i].shape
            height_list.append(h)
            width_list.append(w)
            dealed_img = transform(imgs[i])
            # bgr to rgb
            dealed_img = dealed_img[:, :, (2, 1, 0)]
            dealed_img_list.append(dealed_img)
        # input_imgs =  torch.cat(dealed_img_list, dim=-1)
        input_imgs = torch.from_numpy(np.array(dealed_img_list)).permute(0, 3, 1, 2).cuda()
        detections = self.net(input_imgs)  # .data
        # only need person class info
        dets = detections[:, 15, :, :]
        # 这里要重新写逻辑，此时是一次得到一个批次的box
        for i in range(len(dets)):
            mask = dets[i, :, 0].gt(0.).view(1, dets.size(1), 1).expand(1, dets.size(1), 5)
            det = torch.masked_select(dets[i], mask).view(-1, 5)
            if det.size(0) == 0:
                result_bboxes.append(None)
                continue
            # if i == 2:
            #     result_bboxes.append(None)
            #     continue
            boxes = det[:, 1:].detach().cpu().numpy()
            boxes[:, 0] *= width_list[i]  # x1
            boxes[:, 2] *= width_list[i]  # x2
            boxes[:, 1] *= height_list[i]  # y1
            boxes[:, 3] *= height_list[i]  # y2
            boxes = boxes.astype(np.int)
            boxes[boxes < 0] = 0
            scores = det[:, 0].detach().cpu().numpy()
            scores_index = np.argsort(scores)[::-1]
            select_box_index = [scores_index[0]]
            if len(scores_index) >= 2:
                for j in range(1, len(scores_index)):
                    if scores[scores_index[j]] > scores[scores_index[0]] * 0.75:
                        select_box_index.append(scores_index[j])
            result_bboxes.append(boxes[select_box_index])
        return result_bboxes
        # 之后再进行修改，如何判断套装等；比如分数差异不大，就认为两个box都是目标等

    def get_visual_result(self, imgs, transform=BaseTransform(300, MEANS)):
        height_list = []
        width_list = []
        dealed_img_list = []
        result_bboxes = []
        result_scores = []
        for i in range(len(imgs)):
            # imgs[i] = imgs[i].numpy()
            h, w, c = imgs[i].shape
            height_list.append(h)
            width_list.append(w)
            dealed_img = transform(imgs[i])
            # bgr to rgb
            dealed_img = dealed_img[:, :, (2, 1, 0)]
            dealed_img_list.append(dealed_img)
        # input_imgs =  torch.cat(dealed_img_list, dim=-1)
        input_imgs = torch.from_numpy(np.array(dealed_img_list)).permute(0, 3, 1, 2).cuda()
        detections = self.net(input_imgs)  # .data
        # only need person class info
        dets = detections[:, 15, :, :]
        # 这里要重新写逻辑，此时是一次得到一个批次的box
        for i in range(len(dets)):
            mask = dets[i, :, 0].gt(0.).view(1, dets.size(1), 1).expand(1, dets.size(1), 5)
            det = torch.masked_select(dets[i], mask).view(-1, 5)
            if det.size(0) == 0:
                result_bboxes.append(None)
                continue
            # if i == 2:
            #     result_bboxes.append(None)
            #     continue
            boxes = det[:, 1:].detach().cpu().numpy()
            boxes[:, 0] *= width_list[i]  # x1
            boxes[:, 2] *= width_list[i]  # x2
            boxes[:, 1] *= height_list[i]  # y1
            boxes[:, 3] *= height_list[i]  # y2
            boxes = boxes.astype(np.int)
            boxes[boxes < 0] = 0
            scores = det[:, 0].detach().cpu().numpy()
            scores_index = np.argsort(scores)[::-1]
            select_box_index = [scores_index[0]]
            if len(scores_index) >= 2:
                for j in range(1, len(scores_index)):
                    if scores[scores_index[j]] > scores[scores_index[0]] * 0.75:
                        select_box_index.append(scores_index[j])
            result_bboxes.append(boxes[select_box_index])
            result_scores.append(scores[select_box_index])
        return result_bboxes, result_scores

# if __name__ == '__main__':
#     weights_detect = '../weights/detect.pth'
#     detector = Detector(weights_detect)
#     img_path = '/media/liutao/文档-软件/Competition/TaobaoDataset/validation_dataset_part1/image/000006/5.jpg'
#     img = cv2.imread(img_path)
#     result = detector.get_deal_result(img)
#     print(result)
