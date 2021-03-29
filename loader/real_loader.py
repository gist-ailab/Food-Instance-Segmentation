import os
import numpy as np
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset

from .transforms import get_transform

class RealTrayDataset(Dataset):
    def __init__(self, config, mode):
        assert mode in ['train', 'val']
        self.mode = mode # train mode or validation mode
        self.label_type = config["label_type"]
        # load rgb and segmentation mask images
        self.rgb_list = []
        self.seg_list = []
        self.slot_label_list = []
        data_roots = config["dataset_path"]
        if isinstance(data_roots, str): data_roots = [data_roots]
        for data_root in data_roots:
            self.rgb_path = os.path.join(data_root, "RGBImages")
            self.seg_path = os.path.join(data_root, "Annotations", "Annotations_all")
            rgb_list = list(sorted(glob.glob(os.path.join(self.rgb_path, '*'))))
            if 'Thumbs.db' in rgb_list: rgb_list.remove("Thumbs.db")
            seg_list = list(sorted(glob.glob(os.path.join(self.seg_path, '*.mask'))))
            slot_label_list = list(sorted(glob.glob(os.path.join(self.seg_path, '*.txt'))))

            if config["num_test"] != "all":
                if self.mode == 'train': 
                    rgb_list = rgb_list[config["num_test"]:]
                    seg_list = seg_list[config["num_test"]:]
                    slot_label_list = slot_label_list[config["num_test"]:]
                else:
                    rgb_list = rgb_list[:config["num_test"]]
                    seg_list = seg_list[:config["num_test"]]
                    slot_label_list = slot_label_list[:config["num_test"]]
            self.rgb_list += rgb_list
            self.seg_list += seg_list
            self.slot_label_list += slot_label_list
        assert len(self.rgb_list) > 0
        assert len(self.rgb_list) == len(self.seg_list)
        assert len(self.rgb_list) == len(self.slot_label_list)

        # load rgb transform
        self.transform_ver = config["transform"] if "transform" in config else "torchvision"
        self.rgb_transform, self.transform_ver = get_transform(self.transform_ver, self.mode)
        self.width = config["img_resize_w"] if "img_resize_w" in config else 640
        self.height = config["img_resize_h"] if "img_resize_h" in config else 480

        # labels in mask images
        self.slot_cat2label = {
                            '1_slot': 1,
                            '2_slot': 2,
                            '3_slot': 3,
                            '4_slot': 4,
                            '5_slot': 5
                        }
        
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
        make_arr = Image.open(self.seg_list[idx]).convert("L")
        make_arr = make_arr.resize((self.width, self.height), Image.NEAREST)
        make_arr = np.array(make_arr)
                
        # extrack masks
        slot_labels = open(self.slot_label_list[idx]).readlines()
        slot_id2category, slot_valid_ids = {}, []
        for slot_label in slot_labels:
            label, category = slot_label.split(' ') 
            category = category[:-1] if '\n' in category else category
            slot_id2category[int(label)] = category
            if category in self.slot_cat2label: slot_valid_ids.append(int(label))
        
        obj_ids = np.array([mask_id for mask_id in np.unique(make_arr) if mask_id in slot_valid_ids])
        masks = make_arr == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        temp_masks = []
        boxes = []
        labels = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1:
                continue
            temp_masks.append(masks[i])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  
        masks = np.array(temp_masks)
        labels = np.array(labels)
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
