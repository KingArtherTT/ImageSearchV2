import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont
from object_detection.Detector import Detector
from object_detection.Classifier import Classifier

colors = [(random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _ in range(10)]
TaoBbo_CLASSES_MAP = {
    0: '古风', 1: '长马甲', 2: '短马甲', 3: '背心上衣', 4: '背带裤', 5: '吊带上衣', 6: '连体衣', 7: '中裤',
    8: '无袖上衣', 9: '短袖衬衫', 10: '长袖衬衫', 11: '中等半身裙', 12: '长半身裙', 13: '短裙', 14: '长外套',
    15: '短裤', 16: '短外套', 17: '无袖连衣裙', 18: '长袖连衣裙', 19: '长袖上衣', 20: '长裤',
    21: '短袖连衣裙', 22: '短袖上衣'}




def Draw_box(img, rclasses, rscores, rbboxes):
    if img is None:
        return None
    for i in range(len(rclasses)):
        bbox = rbboxes[i]
        # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[i % 10])
        # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
        img = cv2ImgAddText(img, '{:s}|{:.3f}'.format(TaoBbo_CLASSES_MAP[rclasses[i]], rscores[i]), bbox[0],
                            bbox[1] - 25,
                            textColor=colors[i % 10], textSize=25)
        # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        # 不能显示中文 很无奈
        # cv2.putText(img, '{:s}|{:.3f}'.format(TaoBbo_CLASSES_MAP[rclasses[i]], rscores[i]), (bbox[0], bbox[1]),
        #             fontFace=font,
        #             fontScale=1.5,
        #             color=colors[i % 10])
    return img


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("./static/fonts/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def Deal_img(path, detector, classifier):
    output_dir = './static/data/dealed_image/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image_ids = os.listdir(path)
    for i_id in image_ids:
        filenames = os.listdir(os.path.join(path, i_id))
        for f in filenames:
            img = cv2.imread(os.path.join(path, i_id, f))
            rbboxs, rscores = detector.get_visual_result([img.copy()])
            rclasses = classifier.get_class([img], rbboxs)
            img = Draw_box(img, rclasses, rscores[0], rbboxs[0])
            if not os.path.exists(os.path.join(output_dir, i_id)):
                os.mkdir(os.path.join(output_dir, i_id))

            cv2.imwrite(os.path.join(output_dir, i_id, f), img)


def Deal_video(path, detector, classifier):
    output_dir = './static/data/dealed_video/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(path)
    # fourcc1 = cv2.VideoWriter_fourcc(*'LMP4')  # 'FMP4'
    for f in filenames:
        cap = cv2.VideoCapture(os.path.join(path, f))
        fourcc = int(cap.get(6))  # CV_CAP_PROP_FOURCC
        width = int(cap.get(3))  # CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
        height = int(cap.get(4))  # CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
        fps = int(cap.get(5))  # CV_CAP_PROP_FPS Frame rate.
        out = cv2.VideoWriter(os.path.join(output_dir, f), fourcc, fps, (width, height), True)
        # 读取一帧图片
        # i = 0
        while True:
            ret, frame = cap.read()
            if ret:
                rbboxs, rscores = detector.get_visual_result([frame.copy()])
                rclasses = classifier.get_class([frame], rbboxs)
                frame = Draw_box(frame, rclasses, rscores[0], rbboxs[0])
                out.write(frame)
                # print('write:{}'.format(i))
                # i += 1
            else:
                break

        cap.release()
        out.release()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    weights_detect = './weights/detect.pth'
    weights_classify = './weights/classify.pth'
    detector = Detector(weights_detect)
    classify = Classifier(weights_classify)
    Deal_video('./static/data/video/', detector, classify)
