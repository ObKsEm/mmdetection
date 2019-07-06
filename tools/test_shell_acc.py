import argparse
import os

import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image, ImageFont, ImageDraw
from mmcv import color_val, imwrite
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmdet.ops import nms

from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.shell import ShellDataset
from mmdet.datasets.sku import SkuDataset
import xml.etree.ElementTree as ET

test_img_path = "/home/lichengzhi/mmdetection/data/test/JPEGImages"
test_xml_path = "/home/lichengzhi/mmdetection/data/test/Annotations"


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = float(box[0].text)
        y_min = float(box[1].text)
        x_max = float(box[2].text)
        y_max = float(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords


def get_result(result, score_thr=0.5):
    bbox_result = result

    bboxes = np.vstack(bbox_result)
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

    test_bboxes = nms(bboxes, 0.5)
    new_bboxes = [bboxes[i] for i in test_bboxes[1]]
    new_labels = [labels[i] for i in test_bboxes[1]]

    # for bbox, label in zip(new_bboxes, new_labels):
    #     bbox_int = bbox.astype(np.int32)
    #     left_top = (bbox_int[0], bbox_int[1])
    #     right_bottom = (bbox_int[2], bbox_int[3])
    #     cv2.rectangle(
    #         img, left_top, right_bottom, bbox_color, thickness=thickness)
    #     label_text = class_names[
    #         label] if class_names is not None else 'cls {}'.format(label)
    #     if len(bbox) > 4:
    #         label_text += '|{:.02f}'.format(bbox[-1])
    #     img = write_text_to_image(img, label_text, (bbox_int[0], bbox_int[1] - 2), text_color)
    #
    # # if show:
    # #     imshow(img, win_name, wait_time)
    # if out_file is not None:
    #     imwrite(img, out_file)
    return new_bboxes, new_labels


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    config_file = args.config
    checkpoint_file = args.checkpoint

    # build the model from a config file and a checkpoint file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_detector(config_file, checkpoint_file, device=device)
    model.CLASSES = ShellDataset.CLASSES
    # test a single image and show the results
    acc = 0.0
    tot = 0.0
    for r, dirs, files in os.walk(test_img_path):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                xml_path = os.path.join(test_xml_path, file[:-4] + ".xml")
                coords = read_xml(xml_path)
                gt_bboxes = [coord[:4] for coord in coords]
                gt_labels = [coord[4] for coord in coords]
                tot += len(gt_bboxes)
                result = inference_detector(model, img)
                det_bboxes, det_labels = get_result(result, score_thr=0.5)
                ious = bbox_overlaps(np.array(det_bboxes), np.array(gt_bboxes))
                ious_max = ious.max(axis=1)
                ious_argmax = ious.argmax(axis=1)
                for i in range(0, len(det_bboxes)):
                    matched_gt = ious_argmax[i]
                    if model.CLASSES[det_labels[i]] == gt_labels[matched_gt]:
                        acc += 1
    print("tot boxes: %d, acc boxes: %d\n" % (tot, acc))
    print("accuracy: %f" % (acc / tot))


if __name__ == "__main__":
    main()
