import os

import numpy as np
import onnxruntime
import cv2
import mmcv
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import torch

test_img_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/JPEGImages"
test_xml_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/Annotations"
test_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/ImageSets/Main/test.txt"
xml_file_name = "rzx_statistics_6.22.xlsx"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_cfg = [{'type': 'MultiScaleFlipAug',
             'img_scale': (720, 1280),
             'flip': False,
             'transforms': [{'type': 'Resize', 'keep_ratio': True},
                            {'type': 'RandomFlip'},
                            {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53],
                             'std': [58.395, 57.12, 57.375], 'to_rgb': True},
                            {'type': 'Pad', 'size_divisor': 32},
                            {'type': 'ImageToTensor', 'keys': ['img']},
                            {'type': 'Collect', 'keys': ['img']}]}]


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def run_with_onnx_runtime(model_path, w, h):

    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = [LoadImage()] + test_cfg
    test_pipeline = Compose(test_pipeline)
    device = torch.device(0)
    with open(test_path, "r") as f:
        filenames = f.readlines()
        for filename in filenames:
            img_file = filename.strip() + ".jpg"
            xml_file = filename.strip() + ".xml"
            img = cv2.imread(os.path.join(test_img_path, img_file))
            if img is not None:
                # prepare data
                data = dict(img=img)
                data = test_pipeline(data)
                data = scatter(collate([data], samples_per_gpu=1), [device])[0]
                result = session.run([], {input_name: data})
                print(f'Output y.shape: {result.shape}')
                break


if __name__ == '__main__':
    run_with_onnx_runtime('rzx.onnx', w=720, h=1280)
