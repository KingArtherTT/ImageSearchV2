from nets.Embedding.InceptionV4 import Inceptionv4
from nets.Embedding.TripletNet import TripletNet
import torch
import cv2
import numpy as np
from object_detection.Abstract import AbstractFeature


def base_transform(image, size):
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return base_transform(image, self.size)


class FeatureExtract(AbstractFeature):
    def __init__(self, weights):
        self.net = Inceptionv4()
        self.net = TripletNet(self.net)
        self.net.load_state_dict(torch.load(weights))
        self.net.eval()
        print('Finished loading model!', weights)
        self.net = self.net.cuda()

    def get_feature(self, imgs, bboxes, transform=BaseTransform(512)):
        dealed_img_list = []
        # 这里输入的时候 是以 box作为最小单元的
        # 记录每张图片的box数目 便于还原  其实不用还原，有多少个box 就有多少个feature
        for i in range(len(imgs)):
            if bboxes[i] is None:
                continue
            for box in bboxes[i]:
                if box is None:
                    continue
                dealed_img = imgs[i][box[1]:box[3], box[0]:box[2], :]
                dealed_img = transform(dealed_img)
                # bgr to rgb
                dealed_img = dealed_img[:, :, (2, 1, 0)]
                dealed_img_list.append(dealed_img)
        input_imgs = torch.from_numpy(np.array(dealed_img_list)).permute(0, 3, 1, 2).cuda()
        features = self.net.get_embedding(input_imgs).data.cpu().numpy()  # tensor 转numpy
        return features

    # def get_feature(self, imgs, bboxes):
    #
    #     for box in bboxes[i]:
    #         # cv2.imwrite('test.jpg', img[box[1]:box[3], box[0]:box[2], :]) # 要做一下测试看看会不会出问题
    #         f = self._get_feature(img[box[1]:box[3], box[0]:box[2], :])
    #         f_result.append(f)
    #     return f_result


if __name__ == '__main__':
    pass
