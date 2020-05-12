from nets.Embedding.classifyNet import buildClassifyNet
import torch.nn as nn
import torch


class Embedding_Net(nn.Module):
    def __init__(self, resume=None):
        super(Embedding_Net, self).__init__()
        classifyNet = buildClassifyNet(None, is_resume=True)
        if resume is not None:
            classifyNet.load_state_dict(
                torch.load(resume, map_location=lambda storage, loc: storage))  # 在CPU上加载模型参数

        self.vgg = classifyNet.vgg
        # 全局 fine-tune
        # for p in self.parameters():
        #     p.requires_grad = False
        # print(p[0])

        self.conv_1_1 = classifyNet.conv_1_1

        self.features = classifyNet.features
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(1536, self.num_classes)

    def forward(self, x):
        # 不提取多个纬度的特征了 暂时用单个纬度的
        # feature-1
        for k in range(len(self.vgg)):  # 23
            x = self.vgg[k](x)
        # print('after vgg:{}'.format(x.size()))
        x = self.conv_1_1(x)
        # print('conv_1_1:{}'.format(x.size()))
        x = self.features(x)
        # print('after 3 Inception-C:{}'.format(x.size()))
        x = self.global_average_pooling(x)
        # print('after AdaptiveAvgPool2d:{}'.format(x.size()))
        x = x.view(x.size(0), -1)
        # print('after view(x.size(0), -1):{}'.format(x.size()))
        # x = self.linear(x)

        return x
