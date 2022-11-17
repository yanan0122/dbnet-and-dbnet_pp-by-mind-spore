import os
import argparse
import yaml
import sys

sys.path.insert(0, '.')
import mindspore as ms

from datasets.load import create_dataset, TotalText_eval_dic_iter
from utils.eval_utils import WithEvalCell
from utils.env import init_env
from modules.model import get_dbnet


def evaluate(config, path):
    ## Dataset
    eval_net = get_dbnet(config['net'], config, isTrain=True)
    eval_net = WithEvalCell(eval_net, config)

    if "TotalText" in config["dataset"]["class"]:
        val_dataset = TotalText_eval_dic_iter(config)
        # print("new loader")
    else:
        val_dataset, _ = create_dataset(config, False)
        val_dataset = val_dataset.create_dict_iterator()
    # print(val_dataset)
    ms.load_checkpoint(path, eval_net.model)
    eval_net.model.set_train(False)
    metrics, fps = eval_net.eval(val_dataset, show_imgs=config['eval']['show_images'])
    print(f"FPS: {fps}\n"
          f"Recall: {metrics['recall'].avg}\n"
          f"Precision: {metrics['precision'].avg}\n"
          f"Fmeasure: {metrics['fmeasure'].avg}\n")
    return metrics


if __name__ == '__main__':
    ## Config
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, "config/config.yaml"),
                        help="Config file path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--device_num", type=int, default=1, help="Device numbers")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    path_args, _ = parser.parse_known_args()

    stream = open(path_args.config_path, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    config["device_num"] = path_args.device_num
    config["device_id"] = path_args.device_id
    config["ckpt_path"] = path_args.ckpt_path
    init_env(config)  ## Dataset
    print(config)
    evaluate(config, config["ckpt_path"])
