import time
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore.train.callback import Callback

from utils.metric import AverageMeter
from utils.eval_utils import WithEvalCell
from datasets.load import create_dataset, TotalText_eval_dic_iter
from modules.model import DBnet, DBnetPP

class DBNetMonitor(Callback):
    """
    Monitor the result of DBNet.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
    """

    def __init__(self, config, train_net, per_print_times=1):
        super(DBNetMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.config = config

        self.loss_avg = AverageMeter()
        self.rank_id = config["rank_id"]
        self.run_eval = config["run_eval"]
        self.eval_iter = config["eval_iter"]
        if self.run_eval:
            eval_net = eval(config['net'])(config, isTrain=False)
            self.eval_net = WithEvalCell(eval_net, config)
            # val_dataset, _ = create_dataset(config, False)
            # self.val_dataset = val_dataset.create_dict_iterator()
            if "TotalText" in config["dataset"]["class"]:
                self.val_dataset = TotalText_eval_dic_iter(config)
                # print("new loader")
            else:
                val_dataset, _ = create_dataset(config, False)
                self.val_dataset = val_dataset.create_dict_iterator()
            self.max_f = 0.0
        self.train_net = train_net
        self.epoch_start_time = time.time()

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if cb_params.net_outputs is not None:
            if isinstance(loss, tuple):
                if loss[1]:
                    print("==========overflow!==========")
                loss = loss[0]
            loss = loss.asnumpy()
        else:
            print("custom loss callback class loss is None.")
            return

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if cur_step_in_epoch == 1:
            self.loss_avg = AverageMeter()
        self.loss_avg.update(loss)

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            loss_log = "[%s] epoch: %d step: %2d, loss is %.6f" % \
                       (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                       cb_params.cur_epoch_num, cur_step_in_epoch, np.mean(self.loss_avg.avg))
            print(loss_log, flush=True)

    def epoch_begin(self, run_context):
        self.epoch_start_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_time = (time.time() - self.epoch_start_time) * 1000
        time_log = "[%s] epoch: %d cast %2f ms, per tep time: %2f ms" % \
                   (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    cur_epoch, epoch_time, epoch_time / cb_params.batch_num)
        print(time_log, flush=True)
        train_net = self.train_net
        if self.run_eval and cur_epoch % self.eval_iter == 0:
            ms.save_checkpoint(train_net, f"cur_epoch.ckpt")
            ms.load_checkpoint("cur_epoch.ckpt", self.eval_net.model)

            self.eval_net.model.set_train(False)

            # 自己用yield写的迭代器好像不能重置，只能重新生成了
            if "TotalText" in self.config["dataset"]["class"]:
                self.val_dataset = TotalText_eval_dic_iter(self.config)

            metrics, fps = self.eval_net.eval(self.val_dataset, show_imgs=self.config['eval']['show_images'])

            cur_f = metrics['fmeasure'].avg
            print(f"\ncurrent epoch is {cur_epoch}\nFPS: {fps}\nRecall: {metrics['recall'].avg}\n"
                  f"Precision: {metrics['precision'].avg}\nFmeasure: {metrics['fmeasure'].avg}\n")
            if cur_f >= self.max_f:
                print(f"update best ckpt at epoch {cur_epoch}, best fmeasure is {cur_f}\n")
                ms.save_checkpoint(self.eval_net.model, f"best_epoch.ckpt")
                self.max_f = cur_f

    def end(self, run_context):
        print(f" the best fmeasure is {self.max_f}")
