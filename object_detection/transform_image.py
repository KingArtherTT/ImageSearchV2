import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TF_CPP_MIN_LOG_LEVEL = 3 //只显示error信息

import sys
import cv2
import time
import threading
import numpy as np
from object_detection.transform_common import ImageFeature
from queue import Queue

sys.path.append('../')


def expand_by_box_count(ids, names, bboxes):
    ids_new = []
    names_new = []
    bboxes_new = []
    box_count = [len(b) if b is not None else 0 for b in bboxes]
    for i in range(len(box_count)):
        if box_count[i] == 0:
            continue
        ids_new.extend([ids[i]] * box_count[i])
        names_new.extend([names[i]] * box_count[i])
    for b in bboxes:
        if b is not None:
            bboxes_new.extend(b)

    return ids_new, names_new, bboxes_new


def deal_image(ids, names, images, detector, feature_extractor, classifier, q):
    """
    一个商品id 对应 一个 Feature对象
    一个商品id 对应多张图片
    一个图片对应多个 目标 box
    处理图片 返回该instance_id下的所有 Feature对象

    """
    # print('SSD 检测 %d：%s' % (i, time.strftime('%H:%M:%S', time.localtime(time.time()))))
    bboxes = detector.get_deal_result(images.numpy().copy())  #
    # print('SSD 检测 %d：%s' % (i, time.strftime('%H:%M:%S', time.localtime(time.time()))))
    if bboxes is not None and len(bboxes) > 0:
        features = feature_extractor.get_feature(images.numpy().copy(), bboxes)
        classifies = classifier.get_class(images.numpy(), bboxes)
        # print('feature_extractor %d：%s' % (i, time.strftime('%H:%M:%S', time.localtime(time.time()))))
        ids, names, bboxes = expand_by_box_count(ids, names, bboxes)
        for f in range(len(features)):
            img_f = ImageFeature(ids[f], names[f], bboxes[f], features[f], classifies[f])
            q.put(img_f)
        # 要注意观察一下 根据box返回值 截图 正确


# thread_nums = 4
# 上传数据时 拟用 20

def build_image_features(image_loader, detector, feature_extractor, classifier):
    """
    根据传入的图片路径 建立商品特征库
    :return:
    """
    # 读取图片路径 商品id 图片id等
    all_images_feature = {}
    _all_feature_count = 0
    q = Queue()
    for batch_idx, (images, ids, names) in enumerate(image_loader):
        # print(ids)  # 查看一下 tensor cpu uint8
        # print(names)
        deal_image(ids, names, images, detector, feature_extractor, classifier, q)
    # 重新组合队列中的数据

    while not q.empty():
        item = q.get()
        if item.classify not in all_images_feature.keys():
            all_images_feature[item.classify] = []
        all_images_feature[item.classify].append(item)
        _all_feature_count += 1
    print("{}建立图片特征库共计：{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _all_feature_count))
    # 单线程 建立 20个商品特征库 大概耗时：14分钟
    # 4线程 建立 5000个商品特征库 大概耗时：12分钟
    # 8线程 建立 5000个商品特征库 大概耗时：12分钟

    return all_images_feature


if __name__ == '__main__':
    pass
