from importlib.resources import path
from utils.eval_utils import *
import argparse
import yaml
from mindspore import context


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=str, help="config file path")
    parser.add_argument("--device_id", type=int, help="device id", default=0)
    parser.add_argument("--pred_path", type=str, help="output dir")
    parser.add_argument("--gt_path", type=str, help="eval_bin")
    path_args, _ = parser.parse_known_args()

    stream = open(path_args.config_path, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    config['pred_path'] = path_args.pred_path
    config["gt_path"] = path_args.gt_path

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=path_args.device_id)

    evaluater = Evaluate_310(config=config)
    # for batch in evaluater.get_batch():
    #     print(batch)
    metrics, fps = evaluater.eval()
    print(f"FPS: {fps}\n"
          f"Recall: {metrics['recall'].avg}\n"
          f"Precision: {metrics['precision'].avg}\n"
          f"Fmeasure: {metrics['fmeasure'].avg}\n")
    