import argparse
import os
import yaml

import warnings
warnings.filterwarnings('ignore')

import torch
from models import get_instance_segmentation_model
from loader import get_dataset
from coco_utils import get_coco_api_from_dataset, coco_to_excel
from engine import train_one_epoch, evaluate


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
    train_dataset = get_dataset(config)
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset, num_workers=config["num_workers"], 
                    batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dataset = get_dataset(config, mode="val")
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset, num_workers=config["num_workers"], 
                    batch_size=1, shuffle=False, collate_fn=collate_fn)
    print("... Get COCO Dataloader for evaluation")
    coco = get_coco_api_from_dataset(val_loader.dataset)

    # load model
    model = get_instance_segmentation_model(num_classes=2)
    if args.resume:
        if args.resume_ckp:
            resume_ckp = args.resume_ckp
        elif "resume_ckp" in config:
            resume_ckp = config["resume_ckp"]
        else:
            raise ValueError("Wrong resume setting, there's no trainied weight in config and args")
        model.load_state_dict(torch.load(resume_ckp))
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["wd"])
    lr_update = config["save_interval"] if "save_interval" in config else None

    # set training epoch    
    start_epoch = args.resume_epoch if args.resume_epoch else 0
    if args.max_epoch:
        max_epoch = args.max_epoch
    else:
        max_epoch = config['max_epoch'] if "max_epoch" in config else 100
    assert start_epoch < max_epoch
    save_interval = config["save_interval"] if "save_interval" in config else 1

    # logging
    output_folder = config["save_dir"]
    os.makedirs(output_folder, exist_ok=True)

    print("+++ Start Training  @start:{} @max: {}".format(start_epoch, max_epoch))
    for epoch in range(start_epoch, max_epoch):
        # train
        train_one_epoch(epoch, model, train_loader, optimizer, device, lr_update)
        # validate and write results
        coco_evaluator = evaluate(coco, model, val_loader, device)
        # save weight
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch))
            if args.write_excel:
                coco_to_excel(
                    coco_evaluator, epoch, output_folder, 
                    "{}_{}".format(config["dataset"], config["label_type"]))

def get_args_parser():
    parser = argparse.ArgumentParser('Set training of unseen food segmentation', add_help=False)
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--config", type=str, help="path/to/configfile/.yaml")
    parser.add_argument("--max_epoch", type=int, default=None, help="maximun epoch for training")
    parser.add_argument("--resume", action="store_true", help='resume training if true')
    parser.add_argument("--resume_ckp", type=str, default=None, help="path/to/trained/weight")
    parser.add_argument("--resume_epoch", type=int, default=None, help="epoch when resuming")
    parser.add_argument("--write_excel", action="store_true", help='write COCO results into excel file')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Unseen Food Segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
