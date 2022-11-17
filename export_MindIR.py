import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net,context
import os
import yaml
import argparse
from datasets.load import DataLoader
import mindspore.dataset as ds
from modules.model import DBnet, DBnetPP



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint file path")
    parser.add_argument("--device_id", type=int, help="device id")
    parser.add_argument("--config_path", type=str, help="config file path")
    path_args, _ = parser.parse_known_args()

    stream = open(path_args.config_path, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=path_args.device_id)
    net = eval(config['net'])(config, isTrain=False)
    
    # load the parameter into net
    model_dict = load_checkpoint(path_args.checkpoint_path)

    load_param_into_net(net, model_dict)

    input = np.random.uniform(0.0, 1.0, size=[1, 3, 736, 1280]).astype(np.float32)
    file_name = config['net'] + '_' + config['backbone']['initializer']
    export(net, Tensor(input), file_name=file_name, file_format='MINDIR')
    print("Finished.")