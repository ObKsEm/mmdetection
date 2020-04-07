import argparse
from collections import OrderedDict

import mmcv
import torchvision
import torch
from mmdet.apis import init_detector
from torch.autograd import Variable
import onnx
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--image', help='detect image file')
    args = parser.parse_args()
    return args


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    config_file = args.config
    checkpoint_file = args.checkpoint

    print(torch.__version__)
    input_name = ['input']
    output_name = ['output']
    dev = torch.device("cuda:0")
    input = torch.from_numpy((np.random.rand(1, 3, 720, 1280).astype('float32'))).to(dev)
    model = init_detector(config_file, checkpoint_file, device=dev)
    # model.load_state_dict(copyStateDict(torch.load("/data/lichengzhi/ocr_font/models/craft")))
    model.cuda()
    torch.onnx.export(model,
                      input,
                      'faster-rcnn.onnx',
                      input_names=input_name,
                      output_names=output_name,
                      verbose=True,
                      opset_version=10
                      )


if __name__ == "__main__":
    main()
