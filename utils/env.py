import os
import cv2
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size

def init_env(cfg):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(2)
    ms.set_seed(cfg["seed"])

    if cfg["device_target"] != "None":
        if cfg["device_target"] not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg['device_target']}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg["device_target"])

    if cfg["context_mode"] not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg['context_mode']}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg["context_mode"] == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg["device_target"] = ms.get_context("device_target")
    if cfg["device_target"] == "CPU":
        cfg["device_id"] = 0
        cfg["device_num"] = 1
        cfg["rank_id"] = 0

    if cfg["device_num"] > 1:
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg["device_num"] != group_size:
            raise ValueError(f"the setting device_num: {cfg['device_num']} not equal to the real group_size: {group_size}")
        cfg["rank_id"] = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if "all_reduce_fusion_config" in cfg:
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg["all_reduce_fusion_config"])
    else:
        if "device_id" in cfg and isinstance(cfg["device_id"], int):
            ms.set_context(device_id=cfg["device_id"])
        cfg["device_num"] = 1
        cfg["rank_id"] = 0
        print("run standalone!", flush=True)
