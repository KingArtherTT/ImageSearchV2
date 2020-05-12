from nets.classify.classifyNet import buildClassifyNet
import torch
import cv2
import numpy as np
from torch.autograd import Variable

# 为了完善分类效果，将观察后 经常被错分类的商品归于同一类
# class_map = {0: 21, 1: 22, 2: 22, 3: 22, 4: 21, 5: 22, 6: 21, 7: 20, 8: 22, 9: 22, 10: 19, 11: 12, 12: 12, 13: 21, 14: 21, 15: 21,
#              16: 19, 17: 21, 18: 21, 19: 19, 20: 20, 21: 21, 22: 22}
class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 6, 7: 20, 8: 3, 9: 22, 10: 10, 11: 12, 12: 12, 13: 13, 14: 14, 15: 15,
             16: 19, 17: 21, 18: 21, 19: 19, 20: 20, 21: 21, 22: 22}

def base_transform(image, size):
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return base_transform(image, self.size)


class Classifier(object):
    def __init__(self, weights):
        self.net = buildClassifyNet(None, train=False)
        self.net.load_state_dict(torch.load(weights))
        self.net.eval()
        print('Finished loading model!', weights)
        self.net = self.net.cuda()

    def get_class(self, imgs, bboxes, transform=BaseTransform(300)):
        c_result = []
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
        outputs = self.net(input_imgs).data.cpu().numpy()  # tensor 转numpy
        for o in outputs:
            c = np.argmax(o)  # 得到最大元素值的下标 代表着类别
            c = class_map[c]
            c_result.append(c)
        return c_result

    # def get_classes(self, imgs, bboxes):
    #     c_result = []
    #     for box in bboxes:
    #         # cv2.imwrite('test.jpg', img[box[1]:box[3], box[0]:box[2], :]) # 要做一下测试看看会不会出问题
    #         c = self.get_class(img[box[1]:box[3], box[0]:box[2], :])
    #         c_result.append(c)
    #     return c_result


if __name__ == '__main__':
    pass
