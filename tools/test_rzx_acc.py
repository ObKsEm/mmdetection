import argparse
import os

import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch

import openpyxl
from PIL import Image, ImageFont, ImageDraw
from mmcv import color_val, imwrite
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmdet.ops import nms

from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.rzxcoco import RZXCocoDataset
import xml.etree.ElementTree as ET

TABLE_HEAD = ["名称", "样本个数", "tp", "fp", "fn", "precision", "recall"]

test_img_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.02.29/JPEGImages"
test_xml_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.02.29/Annotations"
test_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.02.29/ImageSets/Main/test.txt"


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
        scores = scores[inds]

    test_bboxes = nms(bboxes, 0.5)
    new_bboxes = [bboxes[i] for i in test_bboxes[1]]
    new_labels = [labels[i] for i in test_bboxes[1]]
    new_scores = [scores[i] for i in test_bboxes[1]]

    return new_bboxes, new_labels, new_scores


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    config_file = args.config
    checkpoint_file = args.checkpoint

    # build the model from a config file and a checkpoint file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_detector(config_file, checkpoint_file, device=device)
    model.CLASSES = RZXCocoDataset.CLASSES
    cls2id = dict(zip(model.CLASSES, range(0, len(model.CLASSES))))
    # test a single image and show the results
    gt_cls_num = np.zeros((len(model.CLASSES)))
    tp = np.zeros((len(model.CLASSES)))
    fp = np.zeros((len(model.CLASSES)))
    fn = np.zeros((len(model.CLASSES)))
    tn = np.zeros((len(model.CLASSES)))
    acc = 0.0
    tot = 0.0
    with open(test_path, "r") as f:
        filenames = f.readlines()
        for filename in filenames:
            img_file = filename.strip() + ".jpg"
            xml_file = filename.strip() + ".xml"
            img = cv2.imread(os.path.join(test_img_path, img_file))
            if img is not None:
                xml_path = os.path.join(test_xml_path, xml_file)
                coords = read_xml(xml_path)
                if len(coords) is 0:
                    print("No annotations\n")
                    continue
                gt_bboxes = [coord[:4] for coord in coords]
                gt_labels = [coord[4] for coord in coords]
                for label in gt_labels:
                    gt_cls_num[cls2id[label]] += 1
                    tot += 1
                result = inference_detector(model, img)
                det_bboxes, det_labels, det_scores = get_result(result, score_thr=0.5)
                ious = bbox_overlaps(np.array(det_bboxes), np.array(gt_bboxes))
                ious_max = ious.max(axis=1)
                ious_argmax = ious.argmax(axis=1)
                gt_matched_det = np.ones((len(gt_bboxes))) * -1
                det_matched_gt = np.ones((len(det_bboxes))) * -1
                gt_matched_scores = np.zeros((len(gt_bboxes)))
                for i in range(0, len(det_bboxes)):
                    if ious_max[i] > 0.5:
                        target_gt = ious_argmax[i]
                        if gt_matched_scores[target_gt] < det_scores[i]:
                            gt_matched_scores[target_gt] = det_scores[i]
                            gt_matched_det[target_gt] = i
                            det_matched_gt[i] = target_gt
                    else:
                        fp[det_labels[i]] += 1

                for i in range(0, len(det_matched_gt)):
                    gt = int(det_matched_gt[i])
                    if gt > -1:
                        if model.CLASSES[det_labels[i]] == gt_labels[gt]:
                            tp[det_labels[i]] += 1
                            acc += 1
                        else:
                            fp[det_labels[i]] += 1
    for i in range(0, len(model.CLASSES)):
        fn[i] = gt_cls_num[i] - tp[i]
        tn[i] = gt_cls_num.sum() - fn[i] - tp[i] - fp[i]
    print("accuracy: %f" % (acc / tot))
    mat = np.zeros((len(model.CLASSES), len(TABLE_HEAD)))
    for i in range(0, len(model.CLASSES)):
        mat[i][0] = i
        mat[i][1] = gt_cls_num[i]
        mat[i][2] = tp[i]
        mat[i][3] = fp[i]
        mat[i][4] = fn[i]
        mat[i][5] = tp[i] / (tp[i] + fp[i])
        mat[i][6] = tp[i] / (tp[i] + fn[i])
        print("%s: %.0f gt, %.0f det, %.0f tp, precision: %.6f, recall: %.6f" %
              (model.CLASSES[i], gt_cls_num[i], tp[i] + fp[i], tp[i], tp[i] / (tp[i] + fp[i]), tp[i] / (tp[i] + fn[i])))

    if os.path.exists("rzx_statistics.xlsx"):
        os.remove("rzx_statistics.xlsx")
    workbook = openpyxl.Workbook("rzx_statistics.xlsx")
    sheet = workbook.create_sheet("sheet")
    sheet.append(TABLE_HEAD)
    for i in range(0, len(model.CLASSES)):
        label = model.CLASSES[i]
        sheet.append([label, "%.0f" % gt_cls_num[i], "%.0f" % tp[i], "%.0f" % fp[i], "%.0f" % fn[i],
                      "%.6f" % (tp[i] / (tp[i] + fp[i])), "%.6f" % (tp[i] / (tp[i] + fn[i]))])

    workbook.save("rzx_statistics.xlsx")


if __name__ == "__main__":
    main()
