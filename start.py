import sys
from object_detection.Detector import Detector
from object_detection.Classifier import Classifier
from object_detection.FeatureExtract import FeatureExtract
from object_detection import transform_image
from object_detection import match_video
from object_detection.Similarity import Euclidean_Distance
import time
import os
from dataset.dataset import ValidDataSet
import torch
import pickle
from Common_Deal import Deal_img, Deal_video
from flask import Flask, render_template, request, jsonify, Response, make_response

app = Flask(__name__)

all_images_feature = None
all_video_feature = None
measure_similarity = None


@app.route('/')
def Demo():
    if all_images_feature is None:
        Build_Feature()
    return render_template('homepage.html')


@app.route('/action/get_result/', methods=['POST'])
def get_search_result():
    global all_images_feature
    global all_video_feature
    global measure_similarity
    if all_images_feature is None:
        try:
            with open('./pickle/all_images_feature.pkl', 'rb') as f:
                all_images_feature = pickle.load(f)
        except Exception as e:
            raise ValueError('all_images_feature is None;{}'.format(e))
    if all_video_feature is None:
        try:
            with open('./pickle/all_video_feature.pkl', 'rb') as f:
                all_video_feature = pickle.load(f)
        except Exception as e:
            raise ValueError('all_video_feature is None;{}'.format(e))

    if measure_similarity is None:
        measure_similarity = Euclidean_Distance()
    result = {}
    request_data = request.get_json()
    video_id = request_data['video_id']
    if video_id not in all_video_feature.keys():
        result['code'] = '-1'
        result['message'] = 'invalid argument of video_id'
        return jsonify(result)
    min_couple, min_couple_list = match_video.match_single_video(all_video_feature[video_id], all_images_feature,
                                                                 measure_similarity)
    result['final_result'] = {}
    result['final_result']['img_path'] = '{}/{}'.format(min_couple[1].item_id, min_couple[1].img_name)
    result['final_result']['distance'] = '{:3f}'.format(min_couple[2])
    result['final_result']['frame_index'] = str(min_couple[0].frame_index)
    result['top_list'] = []
    for couple in min_couple_list:
        t = {}
        t['img_path'] = '{}/{}'.format(couple[1].item_id, couple[1].img_name)
        t['distance'] = '{:3f}'.format(couple[2])
        t['frame_index'] = str(couple[0].frame_index)
        result['top_list'].append(t)
    result['code'] = 1
    return jsonify(result)


def Build_Feature():
    global all_images_feature
    global all_video_feature
    # global detector
    # global classify
    # global feature_extractor
    global measure_similarity

    images_path = ['./static/data/image/']
    videos_path = ['./static/data/']

    batch_size = 5  # 5
    num_workers = 1  # 1

    weights_detect = './weights/detect.pth'
    weights_classify = './weights/classify.pth'
    weights_getFeature = './weights/getFeature.pth'
    thread_nums = 4

    detector = Detector(weights_detect)
    classify = Classifier(weights_classify)
    feature_extractor = FeatureExtract(weights_getFeature)
    measure_similarity = Euclidean_Distance()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    image_dataset = ValidDataSet(images_path, type='image')
    video_dataset = ValidDataSet(videos_path, type='video', thread_nums=thread_nums)
    pin_memory = True
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=pin_memory)
    video_loader = torch.utils.data.DataLoader(video_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=pin_memory)
    # 建立图像特征库
    if os.path.exists('./pickle/all_images_feature.pkl'):
        with open('./pickle/all_images_feature.pkl', 'rb') as f:
            all_images_feature = pickle.load(f)
    else:
        # 因为有了分类信息 所有 all_images_feature 的数据格式可以进行更改，修改成 key：类别；value 是image-feature
        all_images_feature = transform_image.build_image_features(image_loader, detector, feature_extractor, classify)
        with open('./pickle/all_images_feature.pkl', 'wb') as f:
            pickle.dump(all_images_feature, f)

    # 建立视频特征库
    if os.path.exists('./pickle/all_video_feature.pkl'):
        with open('./pickle/all_video_feature.pkl', 'rb') as f:
            all_video_feature = pickle.load(f)
    else:
        # 原有的多线程处理 太占用GPU内存了，因此切换成 使用dataset 按照batchsize进行处理
        all_video_feature = match_video.build_videos_features(video_loader, detector, feature_extractor, classify)
        with open('./pickle/all_video_feature.pkl', 'wb') as f:
            pickle.dump(all_video_feature, f)
    # 释放显存占用
    torch.cuda.empty_cache()
    # all_video_result = match_video.match_video_in_images(all_video_feature, all_images_feature, measure_similarity)

    # 处理图像
    if os.path.exists('./static/data/dealed_image/') and len(os.listdir('./static/data/dealed_image/')) > 0:
        print('already exist dealed_image')
    else:
        Deal_img(images_path[0], detector, classify)

    # 处理视频
    if os.path.exists('./static/data/dealed_video/') and len(os.listdir('./static/data/dealed_video/')) > 0:
        print('already exist dealed_video')
    else:
        Deal_video(videos_path[0], detector, classify)
