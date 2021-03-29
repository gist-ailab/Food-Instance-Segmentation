import argparse
import os
import cv2
import numpy as np
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
    model.load_state_dict(torch.load(args.ckp_path))
    model.eval()

    # load images and transform 
    print("... loading", end=' ')
    img_list = sorted(os.listdir(args.input_dir)) 
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                        ])
    print("{} images".format(len(img_list)))

    # visualization setting
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    thres = float(args.thres)

    # inference 
    print("+++ Start inference !")
    for i, img_name in enumerate(img_list):
        print("... inference ({}/{}) _ {}".format(i+1, len(img_list), img_name))
        # load and transform image
        img_file = os.path.join(args.input_dir, img_name)
        img_data = Image.open(img_file).convert("RGB")
        # img_tensor = img_data.resize(img_resize, Img.BICUBIC)
        img_tensor = transform(img_data)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_arr = np.array(img_data).astype(np.uint8)

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
        # print("[{} Scores]:".format(pred_scores.shape[0]), list(pred_scores))
        ids = np.where(pred_scores > thres)[0]
        colors = np.random.randint(0, 255, (len(ids), 3))
        for color_i, pred_i in enumerate(ids):
            color = tuple(map(int, colors[color_i]))
            # draw segmentation
            mask = pred_mask[pred_i] 
            mask = mask * color
            img_arr = cv2.addWeighted(img_arr, 1, mask.astype(np.uint8), 0.5, 0)
            # draw bbox and text
            x1, y1, x2, y2 = map(int, pred_boxes[pred_i])
            cv2.rectangle(img_arr, (x1, y1), (x2, y2), color, 2)
            vis_text = "FOOD({:.2f})".format(pred_scores[pred_i])
            cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
            cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # # save for debugging
            # cv2.imwrite("tmp_{}.png".format(color_i), img_arr)
        # save visualized image
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        save_name = os.path.join(args.output_dir, img_name)
        cv2.imwrite(save_name, img_arr)  


def get_args_parser():
    parser = argparse.ArgumentParser('Set Visualization of unseen food segmentation', add_help=False)
    parser.add_argument('--input_dir', type=str, help='path/to/images/for/inference')
    parser.add_argument('--output_dir', type=str, help='path/to/save/visualized/images')
    parser.add_argument("--gpu", type=str, default="0", help="gpu ID number to use.")
    parser.add_argument("--ckp_path", type=str, default="ckps/MASKRCNN_SIM.tar", help='path/to/trained/weight')
    parser.add_argument("--thres", type=float, default=0.5, help='threshold for instance segmentation')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualizing Food Image Recognition', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
