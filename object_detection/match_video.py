import cv2
from object_detection.Abstract import AbstractSimilarity, AbstractDistance
import os
import time
import threading
from queue import Queue
from object_detection.transform_common import VideoFeature
from random import randint
import warnings
from sys import maxsize
import torch

warnings.filterwarnings("ignore")


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


def deal_frame(ids, names, images, detector, feature_extractor, classifier):
    """
    一个video_id 对应 一个 Feature对象
    一个video_id 对应多个视频帧
    一个视频帧对应多个 目标 box

    """
    frames_result = []
    bboxes = detector.get_deal_result(images.numpy().copy())

    if bboxes is not None and len(bboxes) > 0:
        features = feature_extractor.get_feature(images.numpy().copy(), bboxes)
        classifies = classifier.get_class(images.numpy(), bboxes)
        ids, names, bboxes = expand_by_box_count(ids, names, bboxes)
        for f in range(len(features)):
            video_f = VideoFeature(ids[f], names[f], bboxes[f], features[f], classifies[f])
            frames_result.append(video_f)
    return frames_result


def match_single_video(video_features, images_feature, measure_similarity):
    """

    :param video_features:  :type list
    :param images_feature:  :type dic
    :param measure_similarity:  :type class 将相似度映射到0-1之间，1是完全相似，0是完全不相似;或者是度量距离的函数
    :return:
    """
    # if issubclass(type(measure_similarity), AbstractSimilarity):
    #     max_couple = [None, None, 0]
    #     for v_f in video_features:
    #         for i_f_key in images_feature:
    #             if v_f.classify != i_f_key:
    #                 continue
    #             for i_f in images_feature[i_f_key]:
    #                 sim = measure_similarity.get_similarity(v_f.feature, i_f.feature)
    #                 if sim > max_couple[2]:
    #                     max_couple[0] = v_f
    #                     max_couple[1] = i_f
    #                     max_couple[2] = sim
    #     return max_couple
    # 外层尝试引入多线程
    # 内层尝试改进 检索算法 引入 投票模型的思想
    # 每一个Video-box的 feature 都找到一个距离最近的image-feature，看是否有交集重复；若无交集则按照距离最短的来选择
    # 由于暂不考虑套装问题，因此先确定video的主要类别,然后剔除其余的类别的Box
    # class_v_dic = {}
    # for v_f in video_features:
    #     if v_f.classify not in class_v_dic.keys():
    #         class_v_dic[v_f.classify] = []
    #     class_v_dic[v_f.classify].append(v_f)
    # compare_v = []
    # for key, value in enumerate(class_v_dic):
    #     if len(value) > len(compare_v):
    #         compare_v = []
    #         compare_v.extend(value)
    # for key, value in enumerate(class_v_dic):
    #     if len(value) == len(compare_v) and value[0].classify != compare_v[0].classify:
    #         compare_v.extend(value)

    if issubclass(type(measure_similarity), AbstractDistance):
        # 每个video_feature 都要进行比较 获得一个距离最近的box
        min_couple_list = []
        for v_f in video_features:
            min_couple = [None, None, maxsize]
            for i_f_key in images_feature:
                # if v_f.classify != i_f_key:
                #     continue
                for i_f in images_feature[i_f_key]:
                    dis = measure_similarity.get_distance(v_f.feature, i_f.feature)
                    if dis < min_couple[2]:
                        min_couple[0] = v_f
                        min_couple[1] = i_f
                        min_couple[2] = dis
            min_couple_list.append(min_couple)


        # 问题一：怎么找到最能代表Video 当前类别的feature？
        # 问题二：如何进行投票选择
        # 问题三：怎么解决套装的问题
        return min_couple, min_couple_list


def match_video_in_images(all_video_features, all_images_feature, measure_similarity):
    """

    :param video_features:  :type dic
    :param images_feature:  :type dic
    :param measure_similarity:  :type class 将相似度映射到0-1之间，1是完全相似，0是完全不相似;或者是度量距离的函数
    :return:
    """
    all_video_result = {}
    # 逐一处理视频 后续引入多线程处理
    for video_id in all_video_features.keys():
        couple = match_single_video(all_video_features[video_id], all_images_feature, measure_similarity)
        # 暂时不处理套装问题
        if couple[0] is None or couple[1] is None:
            continue
        all_video_result.update(get_single_result(couple[0], couple[1]))
    return all_video_result


def deal_suit_in_video(video_features, images_feature, measure_similarity):
    """
    item_id为商品id，frame_index为匹配到的视频帧编号，result为视频帧和商品图中商品的位置信息
    一个video_id 可对应多个 商品 套装的那种；也就是一个视频帧里 可以有多个商品对应
    """
    pass


def get_single_result(video_feature, image_feature):
    result = {}
    result[video_feature.video_id] = {}
    result[video_feature.video_id]['item_id'] = image_feature.item_id
    result[video_feature.video_id]['frame_index'] = int(video_feature.frame_index)
    result[video_feature.video_id]['result'] = []
    box_result = {}
    box_result['img_name'] = image_feature.img_name
    box_result['item_box'] = image_feature.item_box.tolist()
    box_result['frame_box'] = video_feature.frame_box.tolist()
    result[video_feature.video_id]['result'].append(box_result)
    return result


def deal_videos(ids, names, images, detector, feature_extractor, classifier, q):
    video_features = deal_frame(ids, names, images, detector, feature_extractor, classifier)
    # max_couple = match_video_in_images(video_features, all_images_feature, measure_similarity)
    # 暂时不处理套装问题
    # video_result = precise_match_video_in_images(video_feature, all_images_feature, measure_similarity)
    # if max_couple[0] is None or max_couple[1] is None:
    #     continue
    # get_single_result(max_couple[0], max_couple[1])
    q.put(video_features)


def build_videos_features(video_loader, detector, feature_extractor, classifier):
    """
    获取 视频识别结果
    """
    # print("结束读取文件目录 %s： %s" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), data_set_dir))
    # print('合计读取视频数目：%d' % len(all_video_ids))
    # print('开始处理视频 %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    q = Queue()
    for batch_idx, (images, ids, names) in enumerate(video_loader):
        # print(ids)
        # print(names)
        if torch.is_tensor(ids):
            ids = ids.numpy()
        if torch.is_tensor(names):
            names = names.numpy()
        deal_videos(ids, names, images, detector, feature_extractor, classifier, q)

    # 返回最终结果
    all_video_features = {}
    _all_feature_count = 0
    while not q.empty():
        items = q.get()
        for item in items:
            if item.video_id not in all_video_features.keys():
                all_video_features[item.video_id] = []
            all_video_features[item.video_id].append(item)
            _all_feature_count += 1
    # print('结束处理视频 %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('合计处理视频数目:%d' % len(all_video_result))
    print("{}建立视频特征库共计：{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _all_feature_count))
    return all_video_features


if __name__ == "__main__":
    pass
