import numpy as np
from object_detection.Abstract import AbstractSimilarity, AbstractDistance
from scipy.spatial.distance import pdist
import cv2
from sys import maxsize


class CosSimilarity(AbstractSimilarity):

    @staticmethod
    def get_similarity(f1, f2):
        if isinstance(f1, list):
            f1 = np.array(f1).reshape((-1,))
        if isinstance(f2, list):
            f2 = np.array(f2).reshape((-1,))
        return CosSimilarity.get_cos(f1, f2)

    @staticmethod
    def get_cos(vector_a, vector_b):
        vector_a = np.reshape(vector_a, (-1))
        vector_b = np.reshape(vector_b, (-1))
        num = np.dot(vector_a, vector_b.T)  # 若为行向量则 A * B.T
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom  # 余弦值
        sim = 0.5 + 0.5 * cos  # 归一化
        return sim


class Manhattan_Distance(AbstractDistance):
    """曼哈顿距离"""

    @staticmethod
    def get_distance(feature1, feature2):
        X = np.vstack([feature1, feature2])
        d2 = pdist(X, 'cityblock')
        return d2


class Chebyshev_Distance(AbstractDistance):
    """ 切比雪夫距离"""

    @staticmethod
    def get_distance(feature1, feature2, flags=''):
        X = np.vstack([feature1, feature2])
        d2 = pdist(X, 'chebyshev')
        return d2


class Mahalanobis_Distance(AbstractDistance):
    """马氏距离"""

    @staticmethod
    def get_distance(feature1, feature2, flags=''):
        # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
        # 此处进行转置，表示10个样本，每个样本2维
        X = np.vstack([feature1, feature2])
        XT = X.T
        d2 = pdist(XT, 'mahalanobis')
        return d2


class Hanming_Distance(AbstractDistance):
    """汉明距离"""

    @staticmethod
    def get_distance(feature1, feature2, flags=''):
        feature1 = feature1.astype(np.int)
        feature2 = feature2.astype(np.int)
        f1 = ''.join([bin(i) for i in feature1])
        f1 = f1.replace('0b', '')
        f2 = ''.join([bin(i) for i in feature2])
        f2 = f2.replace('0b', '')
        dst = 0
        for l, r in zip(f1, f2):
            if l != r:
                dst += 1
        return dst


class Surf_Distance(AbstractDistance):
    bf = None
    flags = None

    def __init__(self, flags=0.75):
        Surf_Distance.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # surf的normType应该使用NORM_L2或者NORM_L1
        Surf_Distance.flags = flags

    @staticmethod
    def get_distance(feature1, feature2, flags=0.75):
        if Surf_Distance.flags != None:
            flags = Surf_Distance.flags
        matches = Surf_Distance.bf.match(feature1[1], feature2[1])
        matches = sorted(matches, key=lambda x: x.distance)
        for m in matches:
            for n in matches:
                if (m != n and m.distance >= n.distance * flags):
                    matches.remove(m)
                    break
        # 算所有distance的向量长度，越小越相似
        if len(matches) <= 2:
            return maxsize
        else:
            return 1 / np.log2(len(matches))
        # vector = []
        # for m in matches:
        #     vector.append(m.distance)
        # if len(vector) > 0:
        #     vector = np.array(vector)
        #     return np.dot(vector, vector.T)
        # else:
        #     return maxsize


class Euclidean_Distance(AbstractDistance):
    """欧氏距离"""

    @staticmethod
    def get_distance(feature1, feature2, flags=''):
        distance = np.power(feature1 - feature2, 2).sum()
        return distance
