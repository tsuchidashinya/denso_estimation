#! /usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
from data import segmentation_dataset
from model import POINTNET_SEMANTIC
from network_common import network_util
from denso_estimation.network.semantic_segmentation.pointnet_semantic.data import segmentation_dataset
from semantic_segmentation.pointnet_semantic import create_model
from denso_estimation.network.semantic_segmentation.pointnet_semantic.test import *
from pointnet_semantic import *
import argparse
import matplotlib.pyplot as plt
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    print("------------------current main directory------------------")
    print(__file__)
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
    
    dataset_all = segmentation_dataset.SegmentationDataset(args.dataset_path, args.start_index)
    train_dataset, _ = network_util.DivideTrainValDataset(dataset_all)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=network_util.collate_fn
    )
    

    train_dataset_size = len(train_dataloader)
    # val_dataset_size = len(val_dataset)

    print("#training data = %d" % train_dataset_size)
    # print("#val data = %d" % val_dataset_size)

    net = POINTNET_SEMANTIC
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    count = 0
    loss_plot_y = []
    plot_x = []
   
    for epoch in range(opt.epoch_count, opt.num_epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        train_loss = 0.0

        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_steps += opt.batch_size * opt.gpu_num
            epoch_iter += opt.batch_size * opt.gpu_num

            # print("***data****")
            # print(data)
            model.set_input_segmentation(data)
            t_loss = model.train_step()
            train_loss += t_loss
            loss_plot_y.append(t_loss)
            count = count + 1
            plot_x.append(count)
            # print("********************GOAL*************************")
            # break

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time / opt.batch_size)
                writer.print_current_losses("train", epoch, epoch_iter, t_loss, t, t_data)

            if i % opt.save_latest_freq == 0:
                print("saving the latest model (epoch %d, total_steps %d)" % (epoch, total_steps))
                model.save_network("latest")

            iter_data_time = time.time()
        # break
        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d, iter %d" % (epoch, total_steps))
            model.save_network("latest")
            model.save_network(epoch)


        if epoch % opt.run_test_freq == 0:
            val_loss = run_segmentation_test(opt_v, val_dataset)
            writer.print_current_losses("val", epoch, epoch_iter, t_loss, t, t_data)

        if epoch % opt.run_test_freq == 0:
            train_loss /= train_dataset_size
            val_loss /= val_dataset_size
            print("epoch: {}, train_loss: {:.3}" ", val_loss: {:.3}".format(epoch, train_loss, val_loss))
            writer.plot_loss(epoch, train_loss, val_loss)
        writer.close()
    plt.plot(plot_x, loss_plot_y)
    plt.grid()
    plot_file = opt.checkpoints_dir + "/" + opt.dataset_mode + opt.checkpoints_human_swich + "/" + opt.arch + "/" + opt.dataset_model + "/loss_plot.png"
    plt.savefig(plot_file)
    print(plot_file)
    

