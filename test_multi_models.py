import argparse
import os
import glob
import yaml

import warnings
warnings.filterwarnings('ignore')

import torch
from models import get_instance_segmentation_model
from loader import get_dataset
from coco_utils import get_coco_api_from_dataset, coco_to_excel
from engine import evaluate


def collate_fn(batch):
    return tuple(zip(*batch))

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

    # fix seed for reproducibility
    torch.manual_seed(7777)

    # load config
    if args.config[-4:] != '.yaml': args.config += '.yaml'
    with open(args.config) as cfg_file:
        config = yaml.safe_load(cfg_file)
        print(config)

    # load dataset
    val_dataset = get_dataset(config, mode=args.mode)
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset, num_workers=config["num_workers"], 
                    batch_size=1, shuffle=False, collate_fn=collate_fn)
    print("... Get COCO Dataloader for evaluation")
    coco = get_coco_api_from_dataset(val_loader.dataset)

    ckp_paths = glob.glob(os.path.join(args.ckp_dir, "*.tar"))
    for ckp_idx, ckp_path in enumerate(ckp_paths):
        print("[CKP {} / {}]".format(ckp_idx, len(ckp_paths)), "-----" * 10)
        # load model
        model = get_instance_segmentation_model(num_classes=2)
        model.load_state_dict(torch.load(ckp_path))
        model.to(device)

        coco_evaluator = evaluate(coco, model, val_loader, device)
        
        if args.write_excel:
            os.makedirs(args.excel_save_dir, exist_ok=True)
            epoch = int(os.path.basename(ckp_path)[6:-4])
            coco_to_excel(
                coco_evaluator, epoch, args.excel_save_dir, 
                "{}_{}".format(config["dataset"], config["label_type"]))

def get_args_parser():
    parser = argparse.ArgumentParser('Set training of unseen food segmentation', add_help=False)
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--mode", type=str, default="test", help="test, val, train")
    parser.add_argument("--config", type=str, help="path/to/configfile/.yaml")
    parser.add_argument("--ckp_dir", type=str, default=None, help="path/to/trained/weight/directory")
    parser.add_argument("--write_excel", action="store_true", help='write COCO results into excel file')
    parser.add_argument("--excel_save_dir", type=str, help="path/to/save/result/excel")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Unseen Food Segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
