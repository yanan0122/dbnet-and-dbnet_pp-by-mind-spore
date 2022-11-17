import yaml
import numpy as np
import glob
import cv2
import os
import math

from mindspore import dataset as ds
from mindspore.dataset.vision import RandomColorAdjust, ToTensor, ToPIL, Normalize
import sys

sys.path.append('.')
from datasets import *
from utils.coco_text import COCO_Text


def get_img(img_path):
    assert os.path.isfile(img_path), img_path + ", Wrong img path!"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float32')
    return img


def get_bboxes(gt_path, config):
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    polys = []
    dontcare = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            dontcare.append(True)
        else:
            dontcare.append(False)
        if config['dataset']['is_icdar2015']:
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), np.array(dontcare)


# def resize(img, polys=None, denominator=32, isTrain=True):
#     # if  polys is not None:
#     #     print(type(polys))
#     #     print(polys)
#     w_scale = math.ceil(img.shape[1] / denominator) * denominator / img.shape[1]
#     h_scale = math.ceil(img.shape[0] / denominator) * denominator / img.shape[0]
#     img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
#     if polys is None:
#         return img
#     if isTrain:

#         new_polys = []
#         for poly in polys:
#             poly[:, 0] = poly[:, 0] * w_scale
#             poly[:, 1] = poly[:, 1] * h_scale
#             new_polys.append(poly)
#         polys = new_polys
#         # print(polys)
#     else:

#         # polys[:, :, 0] = polys[:, :, 0] * w_scale
#         # polys[:, :, 1] = polys[:, :, 1] * h_scale

#         new_polys = []
#         for poly in polys:
#             poly[:, 0] = poly[:, 0] * w_scale
#             poly[:, 1] = poly[:, 1] * h_scale
#             new_polys.append(poly)
#         polys = new_polys
#     return img, polys

def resize(img, polys=None, denominator=32, isTrain=True):
    # if  polys is not None:
    #     print(type(polys))
    #     print(polys)
    w_scale = math.ceil(img.shape[1] / denominator) * denominator / img.shape[1]
    h_scale = math.ceil(img.shape[0] / denominator) * denominator / img.shape[0]
    img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
    if polys is None:
        return img
    else:

        new_polys = []
        for poly in polys:
            poly[:, 0] = poly[:, 0] * w_scale
            poly[:, 1] = poly[:, 1] * h_scale
            new_polys.append(poly)
        polys = new_polys
    return img, polys


class DataLoader():
    """
    This class must be inherited by other class which is used to deal with different datasets.
    The method should be rewrited according to the ground truth in the particular dataset.
    """

    def __init__(self, config, isTrain=True):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.config = config
        self.isTrain = isTrain

        self.ra = RandomAugment(**config['dataset']['random_crop'])
        self.ms = MakeSegDetectionData(config['train']['min_text_size'],
                                       config['train']['shrink_ratio'])
        self.mb = MakeBorderMap(config['train']['shrink_ratio'],
                                config['train']['thresh_min'], config['train']['thresh_max'])


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        raise NotImplementedError

    def get_bboxes(self, gt_path):
        raise NotImplementedError


class IC2015Dataloader(DataLoader):

    def __init__(self, config, isTrain=True):
        super().__init__(config, isTrain=isTrain)

        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['img_dir'],
                                               '*' + config['train']['img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['eval']['img_dir'],
                                               '*' + config['eval']['img_format']))
        self.img_paths = img_paths

        if self.isTrain:
            img_dir = config['train']['gt_dir']
            gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.jpg.txt')
                        for img_path in self.img_paths]

        else:
            img_dir = config['eval']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'gt_' + img_path.split('/')[-1].split('.')[0] + '.txt')
                        for img_path in self.img_paths]

        # self.img_paths = img_paths
        self.gt_paths = gt_paths

    def get_bboxes(self, gt_path):
        # print(gt_path)
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        polys = []
        dontcare = []
        for line in lines:
            line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
            gt = line.split(',')

            # print("======================= gt =====================")
            # print(gt)

            if "#" in gt[-1]:
                dontcare.append(True)
            else:
                dontcare.append(False)

            box = [int(gt[i]) for i in range(8)]

            polys.append(box)
        # print(type(polys))
        # print(dontcare)
        # print(np.array(polys).shape)
        return np.array(polys), np.array(dontcare)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # Getting
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = self.get_bboxes(gt_path)
        # print(polys.shape)

        # Random Augment
        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, self.config['dataset']['short_side'])
            img, polys = self.ra.random_rotate(img, polys, self.config['dataset']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))
        # print(type(polys))
        # print(polys)
        img, polys = resize(img, polys, isTrain=self.isTrain)

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)

        # Show Images
        if self.config['dataset']['is_show']:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0] * 255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask * 255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map * 255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask * 255)

        # Random Colorize
        if self.isTrain and self.config['train']['is_transform']:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)
        # img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        else:
            return original, img, polys, dontcare


class TotalTextDataloader(DataLoader):
    def __init__(self, config, isTrain=True):

        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['img_dir'],
                                               '*' + config['train']['img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['eval']['img_dir'],
                                               '*' + config['eval']['img_format']))
        self.img_paths = img_paths

        super().__init__(config, isTrain=isTrain)

        if self.isTrain:
            img_dir = config['train']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'poly_gt_' + img_path.split('/')[-1].split('.')[0] + '.mat')
                            for img_path in self.img_paths]

        else:
            img_dir = config['eval']['gt_dir']
            gt_paths = [os.path.join(img_dir, 'poly_gt_' + img_path.split('/')[-1].split('.')[0] + '.mat')
                            for img_path in self.img_paths]

        # self.img_paths = img_paths
        self.gt_paths = gt_paths


    def get_bboxes(self, gt_path):
        """
        处理一个mat，即处理一张图片的polygons
        :param anno_file: mat文件路径
        :return:
        """
        # print(gt_path)
        from scipy.io import loadmat
        dict = loadmat(gt_path)

        arr = dict['polygt']

        # print(arr)
        # print(arr.shape)

        # print(arr.shape[0])

        polys = []
        dontcare = []
        for line in range(0, arr.shape[0]):
            # 一个循环处理一个polygon
            # print(line)

            x_arr = arr[line][1]
            # print(type(x_str))
            # print(x_str)
            y_arr = arr[line][3]
            content = arr[line][4]

            # print(arr[line])

            if content.shape[0] == 0:  # 有空的标签
                content_str = '#'
            else:
                content_str = content.item()
            poly = []
            for i in range(0, x_arr.shape[1]):
                poly.append(x_arr[0][i])
                poly.append(y_arr[0][i])
            polys.append(poly)

            if "#" in content_str:
                dontcare.append(True)
            else:
                dontcare.append(False)
        # print(polys)
        # print(dontcare)
        return polys, np.array(dontcare)
        # return polys, dontcare

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # Getting
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = self.get_bboxes(gt_path)
        # 由于TotalText中polys是多边形，所以无法用array全部存储（多边形顶点数不同），因此此时polys是list[list,list,...]
        # print(type(polys), type(dontcare))
        # print(polys)

        # Random Augment
        # 对于TotalText数据集，这个if结束后，polys是list[array, array,...]
        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, self.config['dataset']['short_side'])
            img, polys = self.ra.random_rotate(img, polys, self.config['dataset']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = self.polys_convert(polys)
            # polys = polys

        # print(type(polys))
        # print(polys)
        img, polys = resize(img, polys, isTrain=self.isTrain)

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            # polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)

        # print(polys)
        # print(type(gt))
        # Show Images
        if self.config['dataset']['is_show']:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0] * 255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask * 255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map * 255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask * 255)

        # Random Colorize
        if self.isTrain and self.config['train']['is_transform']:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)
        # img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        else:
            return original, img, polys, dontcare

    def polys_convert(self, polys):
        """
        将list[list, list, ....]转换为list[array, array, ....]
        :param anno_file: list[list, list, ....]形式的polys
        :return: list[array, array, ....]的polys
        """
        new_polys = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = np.array(poly)
            poly = np.reshape(poly, (poly.shape[0] // 2, 2))
            # poly = poly.tolist()
            new_polys.append(poly)
        # new_polys = np.array(new_polys)
        return new_polys


class CocoTextDataloader(DataLoader):
    def __init__(self, config, isTrain=True):
        super().__init__(config, isTrain=isTrain)
        self.Coco_api = COCO_Text(config["dataset"]["annotations_path"])
        if self.isTrain:
            img_dir = config['train']['img_dir']
            img_ids = self.Coco_api.get_ids_in_TargetSet("train")
            # 删除没有标签的图片
            real_img_ids = []
            for img_id in img_ids:
                if len(self.Coco_api.getAnnIds(img_id)) == 0:
                    continue
                else:
                    real_img_ids.append(img_id)
            img_names = self.Coco_api.loadImgs_name(real_img_ids)
            img_path  = [os.path.join(img_dir, img_name) for img_name in img_names]

        else:
            img_dir = config['eval']['img_dir']
            img_ids = self.Coco_api.get_ids_in_TargetSet("val")
            # 删除没有标签的图片
            real_img_ids = []
            for img_id in img_ids:
                if len(self.Coco_api.getAnnIds(img_id)) == 0:
                    continue
                else:
                    real_img_ids.append(img_id)
            img_names = self.Coco_api.loadImgs_name(real_img_ids)
            img_path  = [os.path.join(img_dir, img_name) for img_name in img_names]


        self.img_ids = real_img_ids
        self.img_paths = img_path
        # print(len(self.img_ids))

    def get_bboxes(self, img_id):
        annids = self.Coco_api.getAnnIds(img_id)
        anns  = self.Coco_api.loadAnns(annids)
        # ann_polygons = [ann["polygon"] for ann in anns]
        polys = []
        dontcare = []
        
        for ann in anns:
            polys.append(ann["polygon"])
            dontcare.append(True if ann["legibility"] == "legible" else False)

        return np.array(polys), np.array(dontcare)
        

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_id = self.img_ids[index]

         # Getting
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = self.get_bboxes(img_id)
        # print(polys.shape)

        # Random Augment
        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, self.config['dataset']['short_side'])
            img, polys = self.ra.random_rotate(img, polys, self.config['dataset']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))
        # print(type(polys))
        # print(polys)
        img, polys = resize(img, polys, isTrain=self.isTrain)

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)

        if self.config['dataset']['is_show']:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0] * 255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask * 255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map * 255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask * 255)

        if self.isTrain and self.config['train']['is_transform']:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)
        # img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        else:
            return original, img, polys, dontcare

        

def create_dataset(config, is_train):
    ds.config.set_prefetch_size(config["dataset"]["prefetch_size"])
    data_loader = eval(config['dataset']['class'])(config, isTrain=is_train)

    # print(data_loader[1])
    # print(data_loader[1][2])
    # print(data_loader[1][2].dtype)

    if "device_num" not in config:
        config["device_num"] = 1
    if "rank_id" not in config:
        config["rank_id"] = 0
    if is_train:
        dataset = ds.GeneratorDataset(data_loader,
                                      ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
                                      num_parallel_workers=config['dataset']['num_workers'],
                                      num_shards=config["device_num"], shard_id=config["rank_id"],
                                      shuffle=True, max_rowsize=64)
    else:
        dataset = ds.GeneratorDataset(data_loader, ['original', 'img', 'polys', 'dontcare'])
    batch_size = config['train']['batch_size'] if is_train else 1
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    steps_pre_epoch = dataset.get_dataset_size()
    return dataset, steps_pre_epoch


def TotalText_eval_dic_iter(config):
    from mindspore import Tensor
    data_loader = eval(config['dataset']['class'])(config, isTrain=False)
    for i in range(len(data_loader)):
        dic = {}
        dic['original'] = Tensor(np.expand_dims(data_loader[i][0], axis=0))
        dic["img"] = Tensor(np.expand_dims(data_loader[i][1], axis=0))
        dic["polys"] = np.expand_dims(data_loader[i][2], axis=0)
        dic['dontcare'] = np.expand_dims(data_loader[i][3], axis=0)
        yield dic


if __name__ == '__main__':
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(2)
    stream = open('./config/icdar2015/dbnet/config_resnet18_1p.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    iters = TotalText_eval_dic_iter(config)

    count = 0
    avg = 0
    epochs = 1

    # dataset, _ = create_dataset(config, False)
    
    # iters = dataset.create_dict_iterator(num_epochs=epochs)
    # next(iters)

    import time
    start = time.time()
    for it in iters:
        if count > 1:
            cost = time.time() - start
            avg += cost
            print(f"time cost: {cost}, avg: {avg / (count - 1)}, count: {count}", flush=True)
        count += 1
        start = time.time()

