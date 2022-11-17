import os
import time
import numpy as np
import cv2
from tqdm.auto import tqdm

from utils.metric import QuadMetric
from utils.post_process import SegDetectorRepresenter
from mindspore import Tensor


class WithEvalCell:
    def __init__(self, model, config):
        super(WithEvalCell, self).__init__()
        self.model = model
        self.config = config
        self.metric = QuadMetric(config['eval']['polygon'])
        self.post_process = SegDetectorRepresenter(config['eval']['thresh'], config['eval']['box_thresh'],
                                                   config['eval']['max_candidates'],
                                                   config['eval']['unclip_ratio'],
                                                   config['eval']['polygon'],
                                                   config['eval']['dest'])

    def once_eval(self, batch):
        start = time.time()

        preds = self.model(batch['img'])
        # print(preds)
        boxes, scores = self.post_process(preds)
        cur_time = time.time() - start
        raw_metric = self.metric.validate_measure(batch, (boxes, scores))

        cur_frame = batch['img'].shape[0]

        return raw_metric, (cur_frame, cur_time)

    def eval(self, dataset, show_imgs=True):
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []
        count = 0

        for batch in tqdm(dataset):
            # for batch in dataset:
            print(f"start compute time: {time.asctime()}")
            raw_metric, (cur_frame, cur_time) = self.once_eval(batch)
            print(
                f"Recall: {raw_metric['recall']}\n"
                f"Precision: {raw_metric['precision']}")
            print(f"end compute time: {time.asctime()}\n")
            raw_metrics.append(raw_metric)

            total_frame += cur_frame
            total_time += cur_time

            count += 1
            if show_imgs:
                img = batch['original'].asnumpy().squeeze().astype('uint8')
                # gt
                for idx, poly in enumerate(raw_metric['gtPolys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['gtDontCare']:
                        cv2.polylines(img, [poly], True, (255, 160, 160), 4) # 浅绿
                    else:
                        cv2.polylines(img, [poly], True, (255, 0, 0), 4) # 深绿
                # pred
                for idx, poly in enumerate(raw_metric['detPolys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['detDontCare']:
                        cv2.polylines(img, [poly], True, (200, 255, 200), 4) # 浅蓝
                    else:
                        cv2.polylines(img, [poly], True, (0, 255, 0), 4) # 深蓝
                if not os.path.exists(self.config['eval']['image_dir']):
                    os.makedirs(self.config['eval']['image_dir'])
                cv2.imwrite(self.config['eval']['image_dir'] + f'eval_{count}.jpg', img)

        metrics = self.metric.gather_measure(raw_metrics)
        fps = total_frame / total_time
        return metrics, fps


class Evaluate_310:
    def __init__(self, config):
        super(Evaluate_310, self).__init__()
        self.config = config
        self.metric = QuadMetric(config['eval']['polygon'])
        self.post_process = SegDetectorRepresenter(config['eval']['thresh'], config['eval']['box_thresh'],
                                                   config['eval']['max_candidates'],
                                                   config['eval']['unclip_ratio'],
                                                   config['eval']['polygon'],
                                                   config['eval']['dest'])
        self.gt_path = config["gt_path"]
        self.pred_path = config["pred_path"]

    def once_eval(self, batch):
        start = time.time()

        preds = batch['pred']

        boxes, scores = self.post_process(preds)
        cur_time = time.time() - start
        raw_metric = self.metric.validate_measure(batch, (boxes, scores))

        cur_frame = batch['img'].shape[0]

        return raw_metric, (cur_frame, cur_time)

    def eval(self, show_imgs=False):
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []
        count = 0

        for batch in tqdm(self.get_batch()):
            raw_metric, (cur_frame, cur_time) = self.once_eval(batch)
            raw_metrics.append(raw_metric)

            total_frame += cur_frame
            total_time += cur_time

            count += 1
            if show_imgs:
                img = batch['original'].asnumpy().squeeze().astype('uint8')
                # gt
                for idx, poly in enumerate(raw_metric['gtPolys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['gtDontCare']:
                        cv2.polylines(img, [poly], True, (255, 160, 160), 4)
                    else:
                        cv2.polylines(img, [poly], True, (255, 0, 0), 4)
                # pred
                for idx, poly in enumerate(raw_metric['detPolys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['detDontCare']:
                        cv2.polylines(img, [poly], True, (200, 255, 200), 4)
                    else:
                        cv2.polylines(img, [poly], True, (0, 255, 0), 4)
                if not os.path.exists(self.config['eval']['image_dir']):
                    os.makedirs(self.config['eval']['image_dir'])
                cv2.imwrite(self.config['eval']['image_dir'] + f'eval_{count}.jpg', img)

        metrics = self.metric.gather_measure(raw_metrics)
        fps = total_frame / total_time
        return metrics, fps

    def get_batch(self):
        polys_dir = os.path.join(self.gt_path, "eval_polys_bin")
        polys_list = os.listdir(polys_dir)
        polys_list.sort(key=lambda x: int(x[:-4].split('_')[-1]))

        dontcare_dir = os.path.join(self.gt_path, "eval_dontcare_bin")
        dontcare_list = os.listdir(dontcare_dir)
        dontcare_list.sort(key=lambda x: int(x[:-4].split('_')[-1]))

        img_dir = os.path.join(self.gt_path, "eval_input_bin")
        img_list = os.listdir(img_dir)
        img_list.sort(key=lambda x: int(x[:-4].split('_')[-1]))

        pred_list = os.listdir(self.pred_path)
        pred_list.sort(key=lambda x: int(x[:-4].split('_')[2]))
        # print(pred_list)

        polys_shape = open(os.path.join(self.gt_path, "polys_shape"), "r")
        dontcare_shape = open(os.path.join(self.gt_path, "dontcare_shape"), "r")

        for i in range(len(polys_list)):
            batch = {}

            polys = np.fromfile(os.path.join(polys_dir, polys_list[i]), dtype=np.int64)
            # print(polys_shape.readline())
            # print(polys_list[i])
            polys.shape = eval(polys_shape.readline())
            # batch['polys'] = Tensor(polys)
            batch['polys'] = polys

            dontcare = np.fromfile(os.path.join(dontcare_dir, dontcare_list[i]), dtype=np.bool8)
            dontcare.shape = eval(dontcare_shape.readline())
            # batch['dontcare'] = Tensor(dontcare)
            batch['dontcare'] = dontcare

            img = np.fromfile(os.path.join(img_dir, img_list[i]), dtype=np.float32)
            img.shape = (1, 3, 736, 1280)
            # batch['img'] = Tensor(img)
            batch['img'] = img

            pred = np.fromfile(os.path.join(self.pred_path, pred_list[i]), dtype=np.float32)
            pred.shape = (1, 1, 736, 1280)
            # batch["pred"] = (Tensor(pred),)
            batch["pred"] = (pred,)

            yield batch

        polys_shape.close()
        dontcare_shape.close()






