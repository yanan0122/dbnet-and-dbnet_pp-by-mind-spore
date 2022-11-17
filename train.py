import os
import yaml
import argparse

import mindspore as ms
import mindspore.nn as nn
import modules.loss as loss
from modules.model import get_dbnet, WithLossCell
from utils.callback import DBNetMonitor
from utils.learning_rate import warmup_polydecay
from utils.env import init_env
from datasets.load import create_dataset


def init_group_params(net, weight_decay):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


def train():
    ## Config
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, "config/config.yaml"),
                        help="Config file path")
    parser.add_argument("--device_num", type=int, default=1, help="Device numbers")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    path_args, _ = parser.parse_known_args()

    stream = open(path_args.config_path, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    if config['train']['start_epoch_num'] >= config['train']['total_epochs']:
        print('Training cancelled due to invalid config.')
        return
    config["device_num"] = path_args.device_num
    config["device_id"] = path_args.device_id
    init_env(config)  ## Dataset
    print(config)

    train_dataset, steps_pre_epoch = create_dataset(config, True)

    ## Model & Loss & Optimizer
    net = get_dbnet(config['net'], config, isTrain=True)

    if config["train"]["is_continue_training"]:
        ms.load_checkpoint(config["train"]["last_ckpt_path"], net)

    lr = ms.Tensor(warmup_polydecay(base_lr=config['optimizer']['lr']['base_lr'],
                                    target_lr=config['optimizer']['lr']['target_lr'],
                                    warmup_epoch=config['optimizer']['lr']['warmup_epoch'],
                                    total_epoch=config['train']['total_epochs'],
                                    start_epoch=config['train']['start_epoch_num'],
                                    steps_pre_epoch=steps_pre_epoch,
                                    factor=config['optimizer']['lr']['factor']))
    if config['optimizer']['type'] == "sgd":
        print("Use Momentum")
        opt = nn.Momentum(params=init_group_params(net, config['optimizer']['weight_decay']),
                          learning_rate=lr,
                          momentum=config['optimizer']['momentum'])
    elif config['optimizer']['type'] == "adam":
        if hasattr(nn.Adam, "use_amsgrad"):
            print("Use amsgrad Adam")
            opt = nn.Adam(net.trainable_params(), learning_rate=lr, use_amsgrad=True)
        else:
            print("Use Adam")
            opt = nn.Adam(net.trainable_params(), learning_rate=lr)
    else:
        raise ValueError(f"Not support optimizer: {config['optimizer']['type']}")
    criterion = loss.L1BalanceCELoss(**config['loss'])
    if config["mix_precision"]:
        net.to_float(ms.float32)
        net.backbone.to_float(ms.float16)
    net_with_loss = WithLossCell(net, criterion)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss,
                                                 optimizer=opt,
                                                 scale_sense=nn.FixedLossScaleUpdateCell(1024.))
    model = ms.Model(train_net)
    model.train(config['train']['total_epochs'] - config['train']['start_epoch_num'], train_dataset,
                callbacks=[DBNetMonitor(config, train_net=train_net)],
                dataset_sink_mode=config['train']['dataset_sink_mode'])


if __name__ == '__main__':
    train()
    print("Train has completed.")
