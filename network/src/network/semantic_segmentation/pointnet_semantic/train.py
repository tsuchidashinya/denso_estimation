#! /usr/bin/env python3
import os
import numpy as np
import torch
import torch.utils.data
from network.semantic_segmentation.pointnet_semantic.data import segmentation_dataset
from network.semantic_segmentation.pointnet_semantic.model import POINTNET_SEMANTIC, semantic_loss
from network.network_common import network_util
import argparse
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    print("------------------current main directory------------------")
    print(__file__)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights')
    parser.add_argument('--dataset_path', help='Dataset root directory path')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--checkpoints', default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--num_instance_classes', default=2, type=int)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = POINTNET_SEMANTIC.PointNetSemantic(args.num_instance_classes)
    net = net.to(device)
    state = torch.load(args.weights) if args.weights else None
    if state:
        net.load_state_dict(state)
    
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr)
    criterion = semantic_loss.SemanticLoss()
    criterion = criterion.to(device)
    total_steps = 0
    count = 0
    loss_plot_y = []
    plot_x = []
   
    for epoch in range(args.num_epoch):
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # total_steps += args.batch_size
            x_data = torch.from_numpy(data["x_data"].astype(np.float32))
            y_data = torch.from_numpy(data["y_data"].astype(np.float32))
            x_data = x_data.transpose(2, 1)
            x_data, y_data = x_data.to(device), y_data.to(device)
            net.train()
            optimizer.zero_grad()
            # print(y_data.shape)
            pred, trans_feat = net(x_data)
            loss = criterion(pred, y_data, trans_feat)
            loss.backward()
            optimizer.step()
            # train_loss += loss
            # loss_plot_y.append(loss)
            # count = count + 1
            # plot_x.append(count)
            network_util.print_current_losses("train", epoch, loss)

        if epoch % args.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch))
            if not os.path.exists(args.checkpoints):
                os.mkdir(args.checkpoints)
            save_file = os.path.join(args.checkpoints, str(epoch) + ".pth")
            torch.save(net.state_dict(), save_file)
            save_file_latest = os.path.join(args.checkpoints, "latest.pth")
            torch.save(net.state_dict(), save_file_latest)
    save_file = os.path.join(args.checkpoints, str(epoch) + ".pth")
    torch.save(net.state_dict(), save_file)
    save_file_latest = os.path.join(args.checkpoints, "latest.pth")
    torch.save(net.state_dict(), save_file_latest)
    

