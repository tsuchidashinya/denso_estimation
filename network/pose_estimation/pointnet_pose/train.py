#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from data import TrainValDataset
from options.train_options import TrainOptions_raugh_recognition
from options.test_options import TestOptions_raugh_recognition
from data import *
from pose_estimation.PointNetPose import create_model
from pointnet_exec import *
from pose_estimate import *
import matplotlib.pyplot as plt
from pose_estimation.PointNetPose.pointnet_exec import *
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_root', help='Dataset root directory path')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
args = parser.parse_args()

if __name__ == '__main__':
    print("------------------current main directory------------------")
    print(__file__)

    train_dataset = TrainDataLoader(args)

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("#training data = %d" % train_dataset_size)
    print("#val data = %d" % val_dataset_size)

    model = create_model(opt)
    net = PointNetPose(3, 9)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    total_steps = 0
    count = 0
    loss_plot_y = []
    plot_x = []
   
    for epoch in range(opt.epoch_count, opt.num_epoch + 1):
        epoch_iter = 0
        train_loss = 0.0

        for i, data in enumerate(train_dataset):
            epoch_iter += opt.batch_size * opt.gpu_num

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
            message = 'epoch: %d loss: %.3f ' %(epoch, loss)

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d, iter %d" % (epoch, total_steps))
            torch.save(net.cpu().state_dict(), opt.save_file)

        writer.close()
    plt.plot(plot_x, loss_plot_y)
    plt.grid()
    plot_file = opt.checkpoints_dir + "/" + opt.dataset_mode + "/" + opt.checkpoints_human_swich + "/" + opt.arch + "/" + opt.dataset_model + "/loss_plot.png"
    plt.savefig(plot_file)
    print(plot_file)
