import os
import sys
import argparse
import json
import time
import cv2
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets


from test_config import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from tqdm import tqdm

# import matplotlib
# matplotlib.use('')

n_joints = 17
target_size = [target_h, target_w] = [384, 288]
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_parse():
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')

    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='CPN384x288', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')

    return parser.parse_args()


def load_model(args):
    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)
    model = torch.nn.DataParallel(model).cuda()

    # load trainning weights
    checkpoint_file = os.path.join(args.checkpoint, args.test+'.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    print()
    return model


def load_anno(anno_name):
    anno_dir = '/home/yandanqi/0_code/pose_estimation/pytorch-cpn/data/COCO2017/annotations'
    anno_path = os.path.join(anno_dir, anno_name)

    if not os.path.isfile(anno_path):
        raise Exception(f"{anno_path} does not exist!")
    else:
        with open(anno_path) as f:
            anno = json.load(f)

    print(f"=> annotation has been load from {anno_path}!")
    return anno


######################################################################################################
# image process
#####################################################################################################
def crop_image(img, bbox):
    """
    img: [C, H, W]
    """
    # check
    if img.shape[0] != 3:
        raise Exception(f"input img should be [H, W, C]!")
    if img.__class__.__name__ != 'Tensor':
        raise Exception(f"input img should be numpy format, rather than {img.__class__.__name__}!")
    if bbox.__class__.__name__ != 'Tensor':
        bbox = torch.Tensor(bbox)

    bbox_rescaler = 1.1
    input_h, input_w = img.shape[1], img.shape[2]
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]     # torch.Size([1])

    # calculate crop bbox
    x_center, y_center = (x1+x2)/2, (y1+y2)/2
    if x2-x1 > y2-y1:   # x_length is specified, calculate y_length
        x_length = x2 - x1
        y_length = (target_h / target_w) * x_length
    else:
        y_length = y2 - y1
        x_length = (target_w / target_h) * y_length
    x1_crop = (x_center - bbox_rescaler * x_length/2).round().to(torch.int)
    x2_crop = (x_center + bbox_rescaler * x_length/2).round().to(torch.int)
    y1_crop = (y_center - bbox_rescaler * y_length/2).round().to(torch.int)
    y2_crop = (y_center + bbox_rescaler * y_length/2).round().to(torch.int)

    # make sure rescale value in a meaningful range, use zero padding
    x1_crop_ = torch.clamp(input=x1_crop, min=0, max=input_w)
    x2_crop_ = torch.clamp(input=x2_crop, min=0, max=input_w)
    y1_crop_ = torch.clamp(input=y1_crop, min=0, max=input_h)
    y2_crop_ = torch.clamp(input=y2_crop, min=0, max=input_h)

    # crop it
    img_crop = img[:, y1_crop_:y2_crop_, x1_crop_:x2_crop_]

    # add zero padding
    padding = [
        abs(x1_crop - x1_crop_),
        abs(y1_crop - y1_crop_),
        abs(x2_crop - x2_crop_),
        abs(y2_crop - y2_crop_),
    ]
    img_crop = F.pad(img_crop, padding, 0, 'constant')

    return img_crop


def preprocess_image(img_path, bbox=None):
    """
    bbox: [x0, y0, x1, y1]
    """
    with_vis = True
    img_np = plt.imread(img_path)

    # transform
    img_ts = transform(img_np)

    # here it will use gt bbox
    if bbox is not None:
        img_ts = crop_image(img_ts, bbox)

    # vis
    if with_vis:
        fig = plt.figure()
        rows, cols = 1, 2

        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(img_np)
        ax2 = fig.add_subplot(rows, cols, 2)
        img_ts = img_ts.permute(1, 2, 0)
        img_ts_np = img_ts.detach().cpu().numpy()
        ax2.imshow(img_ts_np)

        plt.show()
        plt.close()

    print(f"=> imaged is processed!")
    return img_ts


def prepare_data(anno):
    img_dir = '/home/yandanqi/0_code/pose_estimation/pytorch-cpn/data/COCO2017/val2017'

    # load data
    image_name = anno['imgInfo']['img_paths']
    img_path = os.path.join(img_dir, image_name)

    gt_bbox = anno['unit']['GT_bbox']
    # np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]])

    # preprocess img
    img = preprocess_image(img_path, gt_bbox)

    return img


def main():
    print(f"=> begin...")
    # set params
    idx = 5
    anno_name = 'COCO_2017_val.json'

    args = get_parse()
    print(f"=> argues have been parsed...")

    # load data annotation file and model
    anno_all = load_anno(anno_name)
    model = load_model(args)

    # load img and paired anno
    img = prepare_data(anno_all[idx])[None, :, :, :]   # [C, H, W] ---> [N, C, H, W]

    # change to evaluation mode
    model.eval()
    with torch.no_grad():
        # compute output
        global_outputs, refine_output = model(img)
        score_map = refine_output.data.cpu()
        score_map = score_map.numpy()

        for b in range(img.shape[0]):   # each img in a batch
            single_result_dict = {}
            single_result = []

            single_map = score_map[b]
            r0 = single_map.copy()
            r0 /= 255
            r0 += 0.5
            v_score = np.zeros(n_joints)
            for p in range(n_joints):
                single_map[p] /= np.amax(single_map[p])
                border = 10
                dr = np.zeros((cfg.output_shape[0] + 2*border, cfg.output_shape[1]+2*border))
                dr[border:-border, border:-border] = single_map[p].copy()
                dr = cv2.GaussianBlur(dr, (21, 21), 0)
                lb = dr.argmax()
                y, x = np.unravel_index(lb, dr.shape)
                dr[y, x] = 0
                lb = dr.argmax()
                py, px = np.unravel_index(lb, dr.shape)
                y -= border
                x -= border
                py -= border + y
                px -= border + x
                ln = (px ** 2 + py ** 2) ** 0.5
                delta = 0.25
                if ln > 1e-3:
                    x += delta * px / ln
                    y += delta * py / ln
                x = max(0, min(x, cfg.output_shape[1] - 1))
                y = max(0, min(y, cfg.output_shape[0] - 1))
                resy = float((4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                resx = float((4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                single_result.append(resx)
                single_result.append(resy)
                single_result.append(1)

            if len(single_result) != 0:
                single_result_dict['image_id'] = int(ids[b])
                single_result_dict['category_id'] = 1
                single_result_dict['keypoints'] = single_result
                single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                full_result.append(single_result_dict)

    result_path = args.result
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file,'w') as wf:
        json.dump(full_result, wf)

    # evaluate on COCO
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()

