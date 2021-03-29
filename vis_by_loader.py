import cv2
import numpy as np
import os 
import yaml
from tqdm import tqdm

from loader import get_dataset
import torch


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

draw_box = True

# config for dataset
cfg_name = "./config/eval_real_data_MediHard"
mode = 'test'
save_dir = "./tmp/UNIMIB_vis/OurReal_MediHard"

cfg_name = "./config/eval_unimib"
mode = 'test'
save_dir = "./tmp/Inference/UNIMIB_Testset_GroundTruth"
os.makedirs(save_dir, exist_ok=True)


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
    # de-normalize image tensor
    image = unorm(image[0]).cpu().detach().numpy().transpose(1,2,0) 
    image = np.uint8(image * 255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract mask 
    masks = target["masks"][0].cpu().detach().numpy()
    boxes = target["boxes"][0].cpu().detach().numpy()
    colors = np.random.randint(0, 255, (len(masks), 3))
    for i, mask in enumerate(masks):
        color = tuple(map(int, colors[i]))
        mask = np.expand_dims(mask, -1)
        mask = np.repeat(mask, 3, 2)
        mask = mask * color
        image = cv2.addWeighted(image, 1, mask.astype(np.uint8), 0.5, 0)

        if draw_box:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    save_name = os.path.join(save_dir, "{}.jpg".format(idx))
    cv2.imwrite(save_name, image)
