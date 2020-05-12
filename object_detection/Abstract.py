from abc import ABCMeta, abstractmethod


# 获取特征向量表示的抽象基类
class AbstractFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_feature(self, img, bboxes):
        """
        获取图片中，标注框内的特征向量表示
        :param img:
        :param bboxes:
        :return:
        """
        pass


# 用于比较相似度的抽象基类
class AbstractSimilarity(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def get_similarity(feature1, feature2):
        """
        用于比较相似度的抽象基类,将两个特征向量的相似度映射到0-1之间，1是完全相似，0是完全不相似
        :param feature1:
        :param feature2:
        :return:
        """
        pass


# 用于获取向量距离的基类
class AbstractDistance(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def get_distance(feature1, feature2, flags=''):
        """
        用于比较相似度的抽象基类,将两个特征向量的相似度映射到0-1之间，1是完全相似，0是完全不相似
        :param feature1:
        :param feature2:
        :return:
        """
        pass
