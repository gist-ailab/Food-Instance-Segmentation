import os
import numpy as np
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset

from .transforms import get_transform

class SyntheticDataset(Dataset):
    def __init__(self, config, mode):
        assert mode in ['train', 'val']
        self.mode = mode # train mode or validation mode
        self.label_type = config["label_type"]
        # load rgb and segmentation mask images
        self.rgb_list = []
        self.seg_list = []
        data_roots = config["dataset_path"]
        if isinstance(data_roots, str): data_roots = [data_roots]
        for data_root in data_roots:
            self.rgb_path = os.path.join(data_root, "image")
            self.seg_path = os.path.join(data_root, "mask_obj")
            rgb_list = list(sorted(glob.glob(os.path.join(self.rgb_path, '*'))))
            seg_list = list(sorted(glob.glob(os.path.join(self.seg_path, '*'))))
            if 'Thumbs.db' in rgb_list: rgb_list.remove("Thumbs.db")
            if 'Thumbs.db' in seg_list: seg_list.remove("Thumbs.db")

            if self.mode == 'train': 
                rgb_list = rgb_list[config["num_test"]:]
                seg_list = seg_list[config["num_test"]:]
            else:
                rgb_list = rgb_list[:config["num_test"]]
                seg_list = seg_list[:config["num_test"]]
            self.rgb_list += rgb_list
            self.seg_list += seg_list
        assert len(self.rgb_list) > 0
        assert len(self.rgb_list) == len(self.seg_list)

        # load rgb transform
        self.transform_ver = config["transform"] if "transform" in config else "torchvision"
        self.rgb_transform, self.transform_ver = get_transform(self.transform_ver, self.mode)
        self.width = config["img_resize_w"] if "img_resize_w" in config else 640
        self.height = config["img_resize_h"] if "img_resize_h" in config else 480

        # colors of label in mask images
        self.id2color = {1: (255, 0, 0),
                         2: (0, 255, 0),
                         3: (0, 0, 255),
                         4: (255, 0, 255),
                         5: (0, 255, 255)}
        
        # # TODO: delete after debugging
        # self.rgb_list = self.rgb_list[:10]
        # self.seg_list = self.seg_list[:10]
        
    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        # load rgb image
        rgb = Image.open(self.rgb_list[idx]).convert("RGB")
        rgb = rgb.resize((self.width, self.height), Image.BICUBIC)
        # load mask image
        make_arr = Image.open(self.seg_list[idx]).convert("RGB")
        make_arr = make_arr.resize((self.width, self.height), Image.NEAREST)
        make_arr = np.array(make_arr)
        # extrack masks
        labels, masks, boxes = [], [], []
        for slot_id, (r,g,b) in self.id2color.items():
            cnd_r = make_arr[:,:, 0] == r
            cnd_g = make_arr[:,:, 1] == g
            cnd_b = make_arr[:,:, 2] == b
            mask = cnd_r*cnd_g*cnd_b
            if len(np.unique(mask)) == 1: continue
            # extract bbox
            pos = np.where(mask)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            # skip small boxes
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1: continue
            if self.label_type == "slot_food": label = slot_id
            elif self.label_type == "unseen_food": label = 1
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
