from datasets.load import *
import mindspore
from mindspore import context
import yaml
import numpy as np
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=str, help="config file path")
    parser.add_argument("--device_id", type=int, help="device id", default=0)
    path_args, _ = parser.parse_known_args()

    stream = open(path_args.config_path, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=path_args.device_id)

    import mindspore.dataset as ds
    data_loader = eval(config['dataset']['class'])(config, isTrain=False)
    dataset = ds.GeneratorDataset(data_loader, ['original', 'img', 'polys', 'dontcare'])
    dataset = dataset.batch(1)
    it = dataset.create_dict_iterator(output_numpy=True)

    # print(next(it)["polys"].dtype)
    # print(next(it)["dontcare"].dtype)

    data_path = "./eval_bin"
    os.makedirs(data_path)

    input_dir = "./eval_bin/eval_input_bin"
    os.makedirs(input_dir)

    polys_dir = "./eval_bin/eval_polys_bin"
    os.makedirs(polys_dir)

    dontcare_dir = "./eval_bin/eval_dontcare_bin"
    os.makedirs(dontcare_dir)

    # Record the shape, because when numpy read binary file(np.fromfile()), the shape should be given.
    # Otherwise, the data would be wrong
    polys_shape_recorder = open(data_path+"/polys_shape", 'w')
    dontcare_shape_recorder = open(data_path+"/dontcare_shape", 'w')

    for i,data in tqdm(enumerate(it)):
        input_name = "eval_input_" + str(i+1) + ".bin"
        polys_name = "eval_polys_" + str(i+1) + ".bin"
        dontcare_name = "eval_dontcare_" + str(i+1) + ".bin"

        input_path = os.path.join(input_dir, input_name)
        polys_path = os.path.join(polys_dir, polys_name)
        dontcare_path = os.path.join(dontcare_dir, dontcare_name)

        data['img'].tofile(input_path)
        # print(type(data['img']))
        data['polys'].tofile(polys_path)
        data['dontcare'].tofile(dontcare_path)

        polys_shape_recorder.write(str(data['polys'].shape)+"\n")
        dontcare_shape_recorder.write(str(data['dontcare'].shape)+"\n")
    
    polys_shape_recorder.close()
    dontcare_shape_recorder.close()

    # polys_shape_recorder = open(polys_dir+"/polys_shape", 'w')
    # for i,data in tqdm(enumerate(it)):
    #     polys_shape_recorder.write(str(data['polys'].shape)+"\n")
    # polys_shape_recorder.close()
        
    # polys_shape = open(polys_dir + "/polys_shape", "r",)
    # for i in range(100):
    #     print(type(eval(next(polys_shape))))
    # polys_shape.close()
    

    print("finished")

if __name__ == '__main__':
    # print(np.fromfile("./eval_bin/eval_input_bin/eval_input_1.bin", dtype=np.float32).shape)
    main()