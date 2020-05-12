import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class ClassifyNet(nn.Module):
    def __init__(self, ssd_net, weights_path, num_classes, resume, train=True):
        super(ClassifyNet, self).__init__()

        self.for_train = train
        # 拆分SSD 加载完参数后 丢弃 边缘框预测；并固定vgg-16 and extras
        if not resume:
            ssd_net.load_weights(weights_path)
        self.num_classes = num_classes
        if ssd_net != None:
            self.vgg = ssd_net.vgg  # 参数固定
        else:
            self.vgg = nn.ModuleList(get_vgg(3))
        # self.L2Norm = ssd_net.L2Norm
        # self.extras = ssd_net.extras
        for p in self.parameters():
            p.requires_grad = False
            # print(p[0])

        self.conv_1_1 = Conv2d(1024, 1536, 1, stride=1, padding=0, bias=True)
        blocks = []
        # 分类预测
        for i in range(3):
            blocks.append(Inception_C(1536))
        self.features = nn.Sequential(*blocks)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, self.num_classes)
        if not self.for_train:  # 想想softmax怎么输出
            self.softmax = nn.Softmax()

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
        return x
        x = self.linear(x)

        if self.for_train:
            return x
        else:
            return self.softmax(x)



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def get_vgg(i, batch_norm=False):
    cfg =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        self.branch_1 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv2d(384, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_1_2 = Conv2d(384, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False),
            Conv2d(384, 448, (3, 1), stride=1, padding=(1, 0), bias=False),
            Conv2d(448, 512, (1, 3), stride=1, padding=(0, 1), bias=False),
        )
        self.branch_2_1 = Conv2d(512, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_2_2 = Conv2d(512, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        # print('x0.size():{}'.format(x0.size()))
        x1 = self.branch_1(x)
        # print('x1.size():{}'.format(x1.size()))
        x1_1 = self.branch_1_1(x1)
        # print('x1_1.size():{}'.format(x1_1.size()))
        x1_2 = self.branch_1_2(x1)
        # print('x1_2.size():{}'.format(x1_2.size()))
        x1 = torch.cat((x1_1, x1_2), 1)
        # print('x1.size():{}'.format(x1.size()))
        x2 = self.branch_2(x)
        # print('x2.size():{}'.format(x2.size()))
        x2_1 = self.branch_2_1(x2)
        # print('x2_1.size():{}'.format(x2_1.size()))
        x2_2 = self.branch_2_2(x2)
        # print('x2_2.size():{}'.format(x2_2.size()))
        x2 = torch.cat((x2_1, x2_2), dim=1)
        # print('x2.size():{}'.format(x2.size()))
        x3 = self.branch_3(x)
        # print('x3.size():{}'.format(x3.size()))
        return torch.cat((x0, x1, x2, x3), dim=1)  # 19 x 19 x 1536


def buildClassifyNet(weights, is_resume=True, train=True):
    # cfg = taobao_classify
    classify_net = ClassifyNet(None, weights, 23, is_resume, train=train)
    return classify_net


if __name__ == '__main__':
    weights = '/home/liutao/PycharmProjects/ssd-classification/weights/ssd300_TaoBao_52800.pth'
    buildClassifyNet(weights, is_resume=False)
