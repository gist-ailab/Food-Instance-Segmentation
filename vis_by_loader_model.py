import cv2
import numpy as np
import os 
import yaml
from tqdm import tqdm

from loader import get_dataset
import torch
from models import get_instance_segmentation_model


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # The normalize code -> t.sub_(m).div_(s)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


thres = 0.5
# device = torch.device("cpu")
device = torch.device("cuda")

# config for dataset
cfg_name = "./config/eval_unimib"
cfg_name = "./config/eval_real_data_MediHard"
mode = 'test'

# syn only
ckp_path = "/data/joo/food/maskrcnn/1108_seg_tray_SIM_all/epoch_1.tar"
save_dir = "./tmp/Inference/OurReal_MediHard_SynOnly"
save_dir = "./tmp/Inference/UNIMIB_Testset_SynOnly"

# fine-tune
ckp_path = "/data/joo/food/maskrcnn/210327_real_easy_finetuning/epoch_18.tar"
save_dir = "./tmp/Inference/OurReal_MediHard_FineTune"
# ckp_path = "/data/joo/food/maskrcnn/210326_UNIMIB_FineTuning/epoch_49.tar"
# save_dir = "./tmp/Inference/UNIMIB_Testset_FineTune"

# # # from-scratch
# ckp_path = "/data/joo/food/maskrcnn/210327_real_easy/epoch_22.tar"
# save_dir = "./tmp/Inference/OurReal_MediHard_FromScr"
# ckp_path = "/data/joo/food/maskrcnn/210326_UNIMIB_FronScratch/epoch_44.tar"
# save_dir = "./tmp/Inference/UNIMIB_Testset_FromScr"

os.makedirs(save_dir, exist_ok=True)

# load trained model 
print("... loading model")
model = get_instance_segmentation_model(num_classes=2)
model.to(device)
model.load_state_dict(torch.load(ckp_path))
model.eval()

if cfg_name[-4:] != '.yaml': cfg_name += '.yaml'
with open(cfg_name) as cfg_file:
    config = yaml.safe_load(cfg_file)
    print(config)

dataset = get_dataset(config, mode=mode)
dataloader = torch.utils.data.DataLoader(
                    dataset=dataset, num_workers=config["num_workers"], 
                    batch_size=1, shuffle=False)
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

print("+++ Visualize {} data".format(len(dataloader)))
for idx, (image, target) in tqdm(enumerate(dataloader)):
    # forward and post-process results
    pred_result = model(image.to(device), None)[0]
    pred_mask = pred_result['masks'].cpu().detach().numpy().transpose(0, 2, 3, 1)
    pred_mask[pred_mask >= 0.5] = 1
    pred_mask[pred_mask < 0.5] = 0
    pred_mask = np.repeat(pred_mask, 3, 3)
    pred_scores = pred_result['scores'].cpu().detach().numpy()
    pred_boxes = pred_result['boxes'].cpu().detach().numpy()

    # de-normalize image tensor
    image = unorm(image[0]).cpu().detach().numpy().transpose(1,2,0) 
    image = np.uint8(image * 255)

    ids = np.where(pred_scores > thres)[0]
    colors = np.random.randint(0, 255, (len(ids), 3))
    for color_i, pred_i in enumerate(ids):
        color = tuple(map(int, colors[color_i]))
        # draw segmentation
        mask = pred_mask[pred_i] 
        mask = mask * color
        image = cv2.addWeighted(image, 1, mask.astype(np.uint8), 0.5, 0)
        # draw bbox and text
        x1, y1, x2, y2 = map(int, pred_boxes[pred_i])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_name = os.path.join(save_dir, "{}.jpg".format(idx))
    cv2.imwrite(save_name, image)
