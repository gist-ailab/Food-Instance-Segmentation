import os
import numpy as np
import cv2
import glob
from PIL import Image
import json
import skimage


import torch
from torch.utils.data import Dataset

from .transforms import get_transform

class UNIMIB2016Dataset(Dataset):
    def __init__(self, config, mode):
        assert mode in ['train', 'test']
        self.mode = mode # train mode or validation mode
        self.label_type = config["label_type"]
        # load datas from annotation json
        self.data_root = config["dataset_path"]
        self.ann = os.path.join(self.data_root, "annotations", "{}.json".format(mode))
        self.ann = json.load(open(self.ann))
        self.rgb_list = list(self.ann.keys())
        # load rgb transform
        self.transform_ver = config["transform"] if "transform" in config else "torchvision"
        self.rgb_transform, self.transform_ver = get_transform(self.transform_ver, self.mode)
        self.width = config["img_resize_w"] if "img_resize_w" in config else 640
        self.height = config["img_resize_h"] if "img_resize_h" in config else 480

        # # TODO: delete after debugging
        # self.rgb_list = self.rgb_list[:10]
        # self.seg_list = self.seg_list[:10]
        # self.rgb_list = list(reversed(self.rgb_list))

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        # load rgb image
        img_name = self.rgb_list[idx]
        img_file = os.path.join(self.data_root, "images", img_name+'.jpg')
        rgb = Image.open(img_file).convert("RGB")
        img_w, img_h = rgb.size
        rgb = rgb.resize((self.width, self.height), Image.BICUBIC)
        # extract mask from annotation
        ann = self.ann[img_name]
        num_instance = len(ann)
        masks = []
        boxes = []
        labels = []
        for inst_idx, poly_pts in enumerate(ann):    
            # # extract mask
            # poly_dict = list(ann_inst.values())[0]
            # poly_pts = poly_dict["BR"]
            rr, cc = skimage.draw.polygon(poly_pts[1::2], poly_pts[::2])
            # rr = (img_h-1) - rr
            # cc = (img_w-1) - cc
            # fill mask
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask[rr, cc] = 1
            mask = cv2.resize(mask, (self.width, self.height), cv2.INTER_AREA)
            # extract bbox
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # skip small boxes
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1: continue
            # same label for unseen food segmentation
            label = 1
            labels.append(label)
            masks.append(mask)
            boxes.append([xmin, ymin, xmax, ymax])   

        labels = np.array(labels)
        masks = np.array(masks)
        boxes = np.array(boxes)

        # tensor format data
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["labels"] = labels

        # RGB transform
        if self.transform_ver == "torchvision":
            rgb = self.rgb_transform(rgb)
        elif self.transform_ver == "albumentation":
            rgb = self.rgb_transform(image=np.array(rgb))['image']
        else:
            raise ValueError("Wrong transform version {}".format(self.transform_ver))

        return rgb, target


class UNIMIB2016DatasetFake(Dataset):
    def __init__(self, config, mode):
        assert mode in ['train', 'val', 'test']
        self.mode = mode # train mode or validation mode
        self.label_type = config["label_type"]
        # load datas from annotation json
        self.data_root = config["dataset_path"]
        self.ann = os.path.join(self.data_root, "annotations", "{}.json".format(mode))
        self.ann = json.load(open(self.ann))
        self.rgb_list = list(self.ann.keys())
        # load rgb transform
        self.transform_ver = config["transform"] if "transform" in config else "torchvision"
        self.rgb_transform, self.transform_ver = get_transform(self.transform_ver, self.mode)
        self.width = config["img_resize_w"] if "img_resize_w" in config else 640
        self.height = config["img_resize_h"] if "img_resize_h" in config else 480
        
        # # TODO: delete after debugging
        # self.rgb_list = self.rgb_list[:10]
        # self.seg_list = self.seg_list[:10]
        # self.rgb_list = list(reversed(self.rgb_list))

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        # load rgb image
        img_name = self.rgb_list[idx]
        img_file = os.path.join(self.data_root, "images", img_name+'.jpg')
        rgb = Image.open(img_file).convert("RGB")
        img_w, img_h = rgb.size
        rgb = rgb.resize((self.width, self.height), Image.BICUBIC)
        # extract mask from annotation
        ann = self.ann[img_name]
        num_instance = len(ann)
        masks = []
        boxes = []
        labels = []
        for inst_idx, ann_inst in enumerate(ann):    
            # extract mask
            poly_dict = list(ann_inst.values())[0]
            poly_pts = poly_dict["BR"]
            rr, cc = skimage.draw.polygon(poly_pts[1::2], poly_pts[::2])
            # rr = (img_h-1) - rr
            # cc = (img_w-1) - cc
            # fill mask
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask[rr, cc] = 1
            mask = cv2.resize(mask, (self.width, self.height), cv2.INTER_AREA)
            # extract bbox
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # skip small boxes
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1: continue
            # same label for unseen food segmentation
            label = 1
            labels.append(label)
            masks.append(mask)
            boxes.append([xmin, ymin, xmax, ymax])   

        labels = np.array(labels)
        masks = np.array(masks)
        boxes = np.array(boxes)

        # tensor format data
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["labels"] = labels

        # RGB transform
        if self.transform_ver == "torchvision":
            rgb = self.rgb_transform(rgb)
        elif self.transform_ver == "albumentation":
            rgb = self.rgb_transform(image=np.array(rgb))['image']
        else:
            raise ValueError("Wrong transform version {}".format(self.transform_ver))

        return rgb, target

