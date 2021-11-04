import argparse
import os
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as T

from models import get_instance_segmentation_model


import warnings
warnings.filterwarnings(action='ignore')

def main(args):
    # get device (GPU or CPU)
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model (MASKRCNN)
    print("... loading model")
    model = get_instance_segmentation_model(num_classes=2)
    model.to(device)
    model.load_state_dict(torch.load(args.ckp_path, map_location=torch.device('cpu')))
    model.eval()
    thres = float(args.thres)

    # load transform 
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                        ])
    print("... loading", end=' ')

    # load camera
    cam_id = args.cam_id
    cam = cv2.VideoCapture(cam_id)
    assert cam.isOpened(), 'Cannot capture source'

    # inference 
    print("+++ Start inference !")
    while cam.isOpened():
        start_time = time.time()
        # load and transform image
        start_time = time.time()
        ret, img_arr = cam.read()
        IMG_H, IMG_W, IMG_C = img_arr.shape
        img_data = Image.fromarray(img_arr).convert("RGB")
        img_tensor = transform(img_data)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # forward and post-process results
        pred_result = model(img_tensor, None)[0]
        pred_mask = pred_result['masks'].cpu().detach().numpy().transpose(0, 2, 3, 1)
        pred_mask[pred_mask >= 0.5] = 1
        pred_mask[pred_mask < 0.5] = 0
        pred_mask = np.repeat(pred_mask, 3, 3)
        pred_scores = pred_result['scores'].cpu().detach().numpy()
        pred_boxes = pred_result['boxes'].cpu().detach().numpy()
        # pred_labels = pred_result['labels']

        # draw predictions
        ids = np.where(pred_scores > thres)[0]
        colors = np.random.randint(0, 255, (len(ids), 3))
        # set colors considering location and size of bbox 
        colors = []
        for (x1, y1, x2, y2) in pred_boxes: 
            w = max(x1, x2) - min(x1, x2)
            h = max(y1, y2) - min(y1, y2)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            ratio_x, ratio_y = x / IMG_W, y / IMG_H
            ratio_s = min(w, h) / max(w, h)
            ratio_s = 1 + ratio_s if ratio_s < 0 else ratio_s
            ratio_x, ratio_y, ratio_s = int(ratio_x*255), int(ratio_y*255), int(ratio_s*255)
            colors.append([ratio_x, ratio_y, ratio_s])

        for color_i, pred_i in enumerate(ids):
            color = tuple(map(int, colors[color_i]))
            # draw segmentation
            mask = pred_mask[pred_i] 
            mask = mask * color
            img_arr = cv2.addWeighted(img_arr, 1, mask.astype(np.uint8), 0.5, 0)
            # draw bbox 
            x1, y1, x2, y2 = map(int, pred_boxes[pred_i])
            cv2.rectangle(img_arr, (x1, y1), (x2, y2), color, 2)
            # put text
            vis_text = "FOOD({:.2f})".format(pred_scores[pred_i])
            cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
            cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('frame', img_arr)
        key = cv2.waitKey(1)
        if key == 27:
            break

        print("FPS is {:.2f} | Image Size:{}\t\t".format(
              1/(time.time()-start_time), img_arr.shape), end='\r')
    cam.release()
    cv2.destroyAllWindows()

def get_args_parser():
    parser = argparse.ArgumentParser('Set Visualization of unseen food segmentation', add_help=False)
    parser.add_argument("--gpu", type=str, default="0", help="gpu ID number to use.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ckp_path", type=str, default="ckps/MASKRCNN_SIM.tar", help='path/to/trained/weight')
    parser.add_argument("--cam_id", type=int, default=0, help="CAMERA ID to use.")
    parser.add_argument("--thres", type=float, default=0.5, help='threshold for instance segmentation')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualizing Food Image Recognition', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
