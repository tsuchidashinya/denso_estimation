import argparse
import os
from pathlib import Path
import torch
import torch.utils.data
import yaml
from model import YOLO, model_util
from data import train_dataset
import pandas as pd
from util import util


def print_info(info):
    print(f"epoch: {info['epoch']}", end=" ")
    print(f"image size: {info['img_size']}", end=" ")
    print(f"[Loss] total: {info['loss_total']:.2f}", end=" ")
    print(f"xy: {info['loss_xy']:.2f}", end=" ")
    print(f"wh: {info['loss_wh']:.2f}", end=" ")
    print(f"object: {info['loss_obj']:.2f}", end=" ")
    print(f"class: {info['loss_cls']:.2f}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True,
        help="directory path to custom dataset",
    )
    parser.add_argument(
        "--weights", type=Path, default="weights/darknet53.conv.74",
        help="path to darknet weights file (.weights) or checkpoint file (.pth)",
    )
    parser.add_argument(
        "--config_path", type=Path, default="config/yolov3_denso.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id to use")
    parser.add_argument(
        "--checkpoints", type=Path, default="train_output",
        help="directory where checkpoint files are saved",
    )
    parser.add_argument(
        "--save_epoch_freq", type=int, default=100,
        help="interval between saving checkpoints",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=100,
        help="interval between saving checkpoints",
    )
    parser.add_argument(
        "--print_freq", type=int, default=1,
        help="interval between saving checkpoints",
    )
    args = parser.parse_args()
    time_str = util.get_time_str()
    with open(args.config_path) as f:
        config_object = yaml.safe_load(f)
    img_size = config_object["train"]["img_size"]
    batch_size = config_object["train"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(os.path.dirname(args.config_path), config_object["model"]["class_names"])
    with open(path) as f:
        class_names = [x.strip() for x in f.read().splitlines()]
    net = YOLO.YOLOv3(config_object["model"])
    net = net.to(device).train()

    state = torch.load(args.weights) if args.weights.suffix == ".ckpt" else None
    if state:
        net.load_state_dict(state["model"])
        print(f"Checkpoint file {args.weights} loaded.")
    else:
        model_util.parse_yolo_weights(net, args.weights)
        print(f"Darknet weights file {args.weights} loaded.")
    
    train_dataset = train_dataset.CustomDataset(
        args.dataset_dir, class_names, train=True, img_size=img_size, 
        bbox_format="pascal_voc", augmentation=config_object["augmentation"],
    )
    train_size = len(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config_object["train"]["lr"] / train_size,
        momentum=config_object["train"]["momentum"]
    )
    history = []
    total_steps = 0
    for epoch in range(args.num_epoch):
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            total_steps += batch_size
            imgs, targets, _ = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = net(imgs, targets)
            loss.backward()
            info = {
                "epoch": epoch + 1,
                "img_size": train_dataset.img_size,
                "loss_total": float(loss),
                "loss_xy": float(net.loss_dict["xy"]),
                "loss_wh": float(net.loss_dict["wh"]),
                "loss_obj": float(net.loss_dict["obj"]),
                "loss_cls": float(net.loss_dict["cls"]),
            }
            optimizer.step()
            history.append(info)
            if i % args.print_freq == 0:
                print_info(info)
        if epoch % args.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d, iter %d" % (epoch, total_steps))
            save_dir = os.path.join(args.checkpoints, time_str)
            util.make_dir(save_dir)
            save_file = os.path.join(save_dir, str(epoch) + ".pth")
            save_file_latest = os.path.join(save_dir, "latest.pth")
            history_path = os.path.join(save_dir, f"history_{epoch:06d}.csv")
            torch.save(net.state_dict(), save_file)
            torch.save(net.state_dict(), save_file_latest)
            pd.DataFrame(history).to_csv(history_path, index=False)
            print(
                f"Training state saved. checkpoints: {save_file}, loss history: {history_path}."
            )
    save_dir = os.path.join(args.checkpoints, time_str)
    save_file = os.path.join(save_dir, "latest.pth")
    history_path = os.path.join(save_dir, "history_latest.csv")
    torch.save(net.state_dict(), save_file)


            
