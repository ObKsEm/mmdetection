import argparse

import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image, ImageFont, ImageDraw
from mmcv import color_val, imwrite
from mmdet.ops import nms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.shell import ShellDataset
from mmdet.datasets.MidChineseDescription import MidChineseDescriptionDataset
from mmdet.datasets.sku import SkuDataset
from mmdet.datasets.uav import UavDataset
from mmdet.datasets.rosegold import RoseGoldDataset, RoseGoldMidDataset

font = ImageFont.truetype('fzqh.ttf', 20)


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


def write_text_to_image(img_OpenCV, label, bbox, text_color):
    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    draw.text(bbox, label, font=font, fill=text_color)
    ret = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return ret


def show_result_in_Chinese(img, result, class_names, score_thr=0.3, out_file=None, thickness=1, bbox_color='green',
                           text_color='green'):
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    test_bboxes = nms(bboxes, 0.5)
    new_bboxes = [bboxes[i] for i in test_bboxes[1]]
    new_labels = [labels[i] for i in test_bboxes[1]]

    for bbox, label in zip(new_bboxes, new_labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        img = write_text_to_image(img, label_text, (bbox_int[0], bbox_int[1] - 2), text_color)

    if out_file is not None:
        imwrite(img, out_file)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    config_file = args.config
    checkpoint_file = args.checkpoint

    # build the model from a config file and a checkpoint file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_detector(config_file, checkpoint_file, device=device)
    model.CLASSES = RoseGoldDataset.CLASSES
    # test a single image and show the results

    # img = mmcv.imread(args.image)

    # result = inference_detector(model, img)
    # show_result(img, result, model.CLASSES, score_thr=0.5, out_file=args.out, show=False)
    # show_result_in_Chinese(img, result, model.CLASSES, score_thr=0.5, out_file=args.out)

    # test a list of images and write the results to image files
    imgs = ['/home/lichengzhi/mmdetection/demo/test25.jpg', '/home/lichengzhi/mmdetection/demo/test26.jpg',
            '/home/lichengzhi/mmdetection/demo/test27.jpg', '/home/lichengzhi/mmdetection/demo/test28.jpg',
            '/home/lichengzhi/mmdetection/demo/test29.jpg', '/home/lichengzhi/mmdetection/demo/test30.jpg']
    for img in imgs:
        pos = img.rfind('/')
        imgname = img[pos + 1:]
        result = inference_detector(model, img)
        show_result_in_Chinese(img, result, model.CLASSES, score_thr=0.5, out_file='result_{}'.format(imgname))


if __name__ == "__main__":
    main()
