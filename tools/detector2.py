import argparse
import os
from pprint import pprint

import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image, ImageFont, ImageDraw
from mmcv import color_val, imwrite
from mmdet.ops import nms
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.rgcoco import RGCocoDataset
from mmdet.datasets.rosegold import RoseGoldDataset
from mmdet.datasets.UltraAB import UltraABDataset
from util.py_nms import py_cpu_nms

font = ImageFont.truetype('fzqh.ttf', 20)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('configrg', help='test config file path')
    parser.add_argument('checkpointrg', help='checkpoint file')
    parser.add_argument('configab', help='test config file path')
    parser.add_argument('checkpointab', help='checkpoint file')
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


def show_result_in_Chinese(img, resultrg, resultab, rg_class_names, ab_class_names, score_thr=0.3, out_file=None, thickness=1, bbox_color='green',
                           text_color='green'):
    assert isinstance(rg_class_names, (tuple, list))
    assert isinstance(ab_class_names, (tuple, list))
    img = mmcv.imread(img)

    if isinstance(resultrg, tuple):
        rg_bbox_result, segm_result = resultrg
    else:
        rg_bbox_result, segm_result = resultrg, None

    if isinstance(resultab, tuple):
        ab_bbox_result, segm_result = resultab
    else:
        ab_bbox_result, segm_result = resultab, None

    print(rg_bbox_result)
    print(ab_bbox_result)

    # bbox_result = np.vstack(ab_bbox_result)
    bboxes_rg = np.vstack(rg_bbox_result)
    bboxes_ab = np.vstack(ab_bbox_result)
    # draw segmentation masks
    # if segm_result is not None:
    #     segms = mmcv.concat_list(segm_result)
    #     inds = np.where(bboxes[:, -1] > score_thr)[0]
    #     for i in inds:
    #         color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #         mask = maskUtils.decode(segms[i]).astype(np.bool)
    #         img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes

    labels_rg = [
        np.full(bbox.shape[0], rg_class_names[i])
        for i, bbox in enumerate(rg_bbox_result)
    ]
    labels_rg = np.concatenate(labels_rg)

    labels_ab = [
        np.full(bbox.shape[0], ab_class_names[i])
        for i, bbox in enumerate(ab_bbox_result)
    ]
    labels_ab = np.concatenate(labels_ab)

    assert bboxes_rg.ndim == 2
    assert labels_rg.ndim == 1
    assert bboxes_rg.shape[0] == labels_rg.shape[0]
    assert bboxes_rg.shape[1] == 4 or bboxes_rg.shape[1] == 5

    assert bboxes_ab.ndim == 2
    assert labels_ab.ndim == 1
    assert bboxes_ab.shape[0] == labels_ab.shape[0]
    assert bboxes_ab.shape[1] == 4 or bboxes_ab.shape[1] == 5

    bboxes = np.vstack((rg_bbox_result + ab_bbox_result))
    labels = np.hstack((labels_rg, labels_ab))

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    test_bboxes = py_cpu_nms(bboxes, labels, 0.5)
    new_bboxes = [bboxes[i] for i in test_bboxes]
    new_labels = [labels[i] for i in test_bboxes]
    # test_bboxes = nms(bboxes, 0.5)
    # new_bboxes = [bboxes[i] for i in test_bboxes[1]]
    # new_labels = [labels[i] for i in test_bboxes[1]]

    for bbox, label in zip(new_bboxes, new_labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = label
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        img = write_text_to_image(img, label_text, (bbox_int[0], bbox_int[1] - 2), text_color)

    if out_file is not None:
        imwrite(img, out_file)


def main():
    args = parse_args()

    rgcfg = mmcv.Config.fromfile(args.configrg)
    rgcfg.data.test.test_mode = True

    abcfg = mmcv.Config.fromfile(args.configab)
    abcfg.data.test.test_mode = True

    rg_config_file = args.configrg
    rg_checkpoint_file = args.checkpointrg

    ab_config_file = args.configab
    ab_checkpoint_file = args.checkpointab

    # build the model from a config file and a checkpoint file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelrg = init_detector(rg_config_file, rg_checkpoint_file, device=device)
    modelrg.CLASSES = RoseGoldDataset.CLASSES

    modelab = init_detector(ab_config_file, ab_checkpoint_file, device=device)
    modelab.CLASSES = UltraABDataset.CLASSES

    # test a single image and show the results

    img = mmcv.imread(args.image)
    #
    resultrg = inference_detector(modelrg, img)
    resultab = inference_detector(modelab, img)
    # show_result(img, result, model.CLASSES, score_thr=0.5, out_file=args.out, show=False)
    show_result_in_Chinese(img, resultrg, resultab, modelrg.CLASSES, modelab.CLASSES, score_thr=0.5, out_file=args.out)
    #
    # test a list of images and write the results to image files

    # imgs = []
    # for r, _, files in os.walk("./demo/shell/demo"):
    #     for file in files:
    #         imgs.append(os.path.join(r, file))
    # for img in imgs:
    #     pos = img.rfind('/')
    #     imgname = img[pos + 1:]
    #     result = inference_detector(model, img)
    #     show_result_in_Chinese(img, result, model.CLASSES, score_thr=0.5, out_file='result_{}'.format(imgname))


if __name__ == "__main__":
    main()
