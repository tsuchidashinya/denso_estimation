#! /usr/bin/env python3
import os
from data import pose_dataset
from network_common import network_util
import torch
import torch.utils.data
import torch.nn as nn
from model import POINTNET
import matplotlib.pyplot as plt
import argparse
import numpy as np

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_path', help='Dataset root directory path')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
parser.add_argument('--num_epoch', type=int, default=150)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
args = parser.parse_args()

print("------------------current main directory------------------")
print(__file__)

dataset_all = pose_dataset.PoseDataset(args.dataset_path, args.start_index)
train_dataset,_ = network_util.DivideTrainValDataset(dataset_all)
train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=network_util.collate_fn)

train_dataset_size = len(train_dataloader)
# val_dataset_size = len(val_dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("#training data = %d" % train_dataset_size)
# print("#val data = %d" % val_dataset_size)

net = POINTNET.PointNetPose(3, 9)
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion = nn.MSELoss()
criterion = criterion.to(device)
total_steps = 0
count = 0
loss_plot_y = []
plot_x = []

for epoch in range(args.num_epoch):
    train_loss = 0.0
    for i, data in enumerate(train_dataloader):
        x_data = torch.from_numpy(data["x_data"].astype(np.float32))
        y_data = torch.from_numpy(data["y_data"].astype(np.float32))
        x_data = x_data.transpose(2, 1)
        x_data, y_data = x_data.to(device), y_data.to(device)
        net.train()
        optimizer.zero_grad()
        pred = net(x_data)
        loss = criterion(pred, y_data)
        loss.backward()
        optimizer.step()
        loss_plot_y.append(loss)
        count = count + 1
        plot_x.append(count)
        message = 'epoch: %d loss: %.3f ' %(epoch + 1, loss)
        print(message)
    if epoch % args.save_epoch_freq == 0:
        print("saving the model at the end of epoch %d, iter %d" % (epoch, total_steps))
        save_file = os.path.join(args.checkpoints, str(epoch) + ".pth")
        torch.save(net.state_dict(), save_file)

save_file = os.path.join(args.checkpoints, "latest.pth")
torch.save(net.cpu().state_dict(), save_file)
plt.plot(plot_x, loss_plot_y)
plt.grid()
plot_file = args.checkpoints_dir + "/" + args.dataset_mode + "/" + args.checkpoints_human_swich + "/" + args.arch + "/" + args.dataset_model + "/loss_plot.png"
plt.savefig(plot_file)
print(plot_file)
