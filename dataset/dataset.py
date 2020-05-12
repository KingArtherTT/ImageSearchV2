from torch.utils.data import Dataset
import cv2
import os
import time
import threading
from queue import Queue

# 关键帧序号
KEYS_FRAMES = [i for i in range(0, 361, 40)]
IMAGE_SIZE = (800, 800)
VIDEO_SIZE = (540, 960)


class ValidDataSet(Dataset):
    """需要匹配的数据集合"""

    def __init__(self, path_list, type='image', thread_nums=4):
        self.image_count = 0
        self.root = []
        self.data = []
        self.type = type
        if self.type == 'image':
            for data_set_dir in path_list:
                self.root.append(data_set_dir)
                image_ids = os.listdir(data_set_dir)
                image_filenames = []
                for image_id in image_ids:
                    filenames = os.listdir(data_set_dir + image_id + '/')
                    image_filenames.extend([image_id + '/' + f for f in filenames])
                self.data.append(image_filenames)
            self.image_count = sum([len(d) for d in self.data])
            print(time.strftime('%H:%M:%S', time.localtime(time.time())))
            print('读取完毕图片文件目录')
            print('共计图片：%d' % self.image_count)
        elif self.type == 'video':
            # 先利用多线程 进行视频 to 图片的切分
            self.image_count = 0
            self.root = path_list
            # self.data = self.video2Image(path_list, thread_nums)
            # roots = []
            self.all_video_ids = []
            for data_set_dir in path_list:
                # roots.append(data_set_dir)
                video_filenames = os.listdir(os.path.join(data_set_dir, 'video'))
                video_ids = []
                for filename in video_filenames:
                    video_ids.append(filename.replace('.mp4', ''))
                self.all_video_ids.append(video_ids)
            self.image_count = sum([len(d) * len(KEYS_FRAMES) for d in self.all_video_ids])
            print(time.strftime('%H:%M:%S', time.localtime(time.time())))
            print('读取完毕视频文件目录')
            print('共计视频：%d' % sum([len(d) for d in self.all_video_ids]))

    def video2Image(self, path_list, thread_nums=4):
        roots = []
        all_video_ids = []
        for data_set_dir in path_list:
            roots.append(data_set_dir)
            video_filenames = os.listdir(os.path.join(data_set_dir, 'video'))
            video_ids = []
            for filename in video_filenames:
                video_ids.append(filename.replace('.mp4', ''))
            all_video_ids.append(video_ids)
        print(time.strftime('%H:%M:%S', time.localtime(time.time())))
        print('读取完毕视频文件目录，进行视频帧提取')
        print('共计视频：%d' % sum([len(i) for i in all_video_ids]))
        q = Queue()
        all_image_names = []
        for i in range(len(roots)):
            root = roots[i]
            image_names = []
            output_dir = root + 'video_to_image/'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            video_ids = all_video_ids[i]
            # 使用多线程处理
            # 4线程 需要21分钟
            # 8线程 需要20分钟
            boundary = int(len(video_ids) / thread_nums)
            threads = []
            for _num in range(thread_nums - 1):
                _get_image = threading.Thread(target=deal_video, args=(
                    root, output_dir,
                    video_ids[boundary * _num:boundary * (_num + 1)], q))
                threads.append(_get_image)
            get_image_last = threading.Thread(target=deal_video, args=(
                root, output_dir,
                video_ids[boundary * (thread_nums - 1):], q))
            threads.append(get_image_last)
            # 开启线程运行
            for t in threads:
                t.start()

            # join()方法等待线程完成
            for t in threads:
                t.join()
            while not q.empty():
                image_names.append(q.get())
            all_image_names.append(image_names)
        print(time.strftime('%H:%M:%S', time.localtime(time.time())))
        print('视频帧提取完毕，共计图片：%d' % sum([len(i) for i in all_image_names]))
        return all_image_names

    def __getitem__(self, index):
        if self.type == 'image':
            which = 0
            while True:
                sum_which = sum([len(self.data[j]) for j in range(which + 1)])
                if index < sum_which:
                    index = index - sum([len(self.data[j]) for j in range(which)])
                    break
                else:
                    which += 1
            img = cv2.imread(self.root[which] + self.data[which][index])
            if img is None:
                print(self.root[which] + self.data[which][index], which, index)
            if self.type == 'image':
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            else:
                img = cv2.resize(img, VIDEO_SIZE, interpolation=cv2.INTER_CUBIC)
        elif self.type == 'video':
            which = 0
            while True:
                sum_which = sum([len(self.all_video_ids[j]) * len(KEYS_FRAMES) for j in range(which + 1)])
                if index < sum_which:
                    index = index - sum([len(self.all_video_ids[j]) * len(KEYS_FRAMES) for j in range(which)])
                    break
                else:
                    which += 1
            remainder = index % len(KEYS_FRAMES)
            integer = int(index / len(KEYS_FRAMES))
            img = None
            cap = cv2.VideoCapture(self.root[which] + 'video/' + self.all_video_ids[which][integer] + '.mp4')
            while img is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, KEYS_FRAMES[remainder])
                # 读取一帧图片
                ret, frame = cap.read()
                if ret:
                    img = frame
                else:
                    print('出现异常,读取视频帧', self.root[which] + 'video/' + self.all_video_ids[which][integer] + '.mp4')
                    if remainder > 0:
                        remainder -= 1
                    elif integer > 0:
                        integer -= 1
                    else:
                        which = 0
                    cap = cv2.VideoCapture(self.root[which] + 'video/' + self.all_video_ids[which][integer] + '.mp4')
            cap.release()

        if self.type == 'image':
            info = self.data[which][index].split('/')
            id = info[0]
            name = info[1]
        else:
            id = self.all_video_ids[which][integer]
            name = KEYS_FRAMES[remainder]
        return img, id, name

    def __len__(self):
        return self.image_count


def deal_video(root, output_dir, video_ids, q):
    for v_id in video_ids:
        if not os.path.exists(output_dir + v_id):
            os.mkdir(output_dir + v_id)
        cap = cv2.VideoCapture(root + 'video/' + v_id + '.mp4')
        for i in range(len(KEYS_FRAMES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, KEYS_FRAMES[i])
            # 读取一帧图片
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(output_dir + v_id + '/' + str(KEYS_FRAMES[i]) + '.jpg', frame)
                q.put('video_to_image/' + v_id + '/' + str(KEYS_FRAMES[i]) + '.jpg')
        cap.release()

        # for i in range(len(KEYS_FRAMES)):
        #     q.put('video_to_image/' + v_id + '/' + str(KEYS_FRAMES[i]) + '.jpg')


if __name__ == '__main__':
    path_list = ['/media/liutao/文档-软件/Competition/TaobaoDataset/validation_dataset_part1/']
    output_dir = '/media/liutao/文档-软件/Competition/TaobaoDataset/validation_dataset_part1/video_to_image/'
    dataset = ValidDataSet(path_list, type='video')
    dataset.video2Image(path_list)
