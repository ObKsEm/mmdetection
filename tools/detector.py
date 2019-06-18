import argparse

import cv2
import mmcv

from mmdet.apis import init_detector, inference_detector, show_result


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


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    config_file = args.config
    checkpoint_file = args.checkpoint

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file)

    # test a single image and show the results
    img = cv2.imread(args.image)
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, score_thr=0.5, out_file=args.out)

    # test a list of images and write the results to image files
    # imgs = ['test1.jpg', 'test2.jpg']
    # for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
    #     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))


if __name__ == "__main__":
    main()
