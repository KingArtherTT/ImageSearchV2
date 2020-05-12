import numpy as np


class ImageFeature(object):
    def __init__(self, item_id, img_name, item_box, feature, classify):
        self._item_id = item_id
        self._img_name = img_name
        self._item_box = item_box
        self._feature = feature
        self._classify = classify

    @property
    def item_id(self):
        """
        商品id,也就是文件夹名称
        """
        return self._item_id

    @item_id.setter
    def item_id(self, value):
        self._item_id = str(value)

    @property
    def img_name(self):
        """
         商品图文件名，无.jpg后缀
        """
        return self._img_name

    @img_name.setter
    def img_name(self, value):
        self._img_name = str(value)

    @property
    def item_box(self):
        """
        :return: 该商品图中商品检测框
        """
        return self._item_box

    @item_box.setter
    def item_box(self, value):
        if not isinstance(value, list):
            value = list(value)
        self._item_box = value

    @property
    def feature(self):
        """
        检测框中物品的特征向量表示
        :return:
        """
        return self._feature

    @feature.setter
    def feature(self, value):
        # if not isinstance(value, np.ndarray):
        #     value = np.array(value, shape=(-1,), dtype=np.float)
        self._feature = value

    @property
    def classify(self):
        return self._classify

    @classify.setter
    def classify(self, value):
        try:
            self._classify = int(value)
        except:
            self._classify = -1


class VideoFeature(object):
    def __init__(self, video_id, frame_index, frame_box, feature, classify):
        self._video_id = video_id
        self._frame_index = frame_index
        self._frame_box = frame_box
        self._feature = feature
        self._classify = classify

    @property
    def video_id(self):
        """
        商品id,也就是文件夹名称
        """
        return self._video_id

    @video_id.setter
    def video_id(self, value):
        self._video_id = str(value)

    @property
    def frame_index(self):
        """
         商品图文件名，无.jpg后缀
        """
        return self._frame_index

    @frame_index.setter
    def frame_index(self, value):
        self._frame_index = int(value)

    @property
    def frame_box(self):
        """
        :return: 该商品图中商品检测框
        """
        return self._frame_box

    @frame_box.setter
    def frame_box(self, value):
        if not isinstance(value, list):
            value = list(value)
        self._frame_box = value

    @property
    def feature(self):
        """
        检测框中物品的特征向量表示
        :return:
        """
        return self._feature

    @feature.setter
    def feature(self, value):
        # if not isinstance(value, np.ndarray):
        #     value = np.array(value, shape=(-1,), dtype=np.float)
        self._feature = value

    @property
    def classify(self):
        return self._classify

    @classify.setter
    def classify(self, value):
        try:
            self._classify = int(value)
        except:
            self._classify = -1