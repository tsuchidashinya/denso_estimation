#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn.parallel
from network_common import network_util
# from .semantic_loss import Semantic_Loss
import numpy as np


class PointNetGlobalFeatureSemantic(nn.Module):
    def __init__(self, global_feat = False, feature_transform = True):
        super(PointNetGlobalFeatureSemantic, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2] #getting number of point_cloud (dataset structure: batch_size ch point_cloud)
        trans = self.stn(x) #T-Net
        x = x.transpose(2, 1) #transpose for matrix multiplication
        x = torch.bmm(x, trans) #matrix multiplication 
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0] #getting max data of each ch  (※:[0] is role for getting maxed data([1] is index))
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts) #set number of point_cloud for cat (********)
            return torch.cat([x, pointfeat], 1), trans, trans_feat #convoluted point_cloud(concat final_encoded_data and data_passed_T-net (purpose:add ch_data)) and first T-net second T-Net


class PointNetSemantic(nn.Module):
    def __init__(self, num_class):
        super(PointNetSemantic, self).__init__()
        self.num_class=num_class
        
        self.PointNetGlobalFeature = PointNetGlobalFeatureSemantic(global_feat = False, feature_transform = True)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.last_conv = nn.Conv1d(128, self.num_class, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batchsize = x.size()[0]
        pc_pts = x.size()[2]
        x, trans, trans_feat = self.PointNetGlobalFeature(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.last_conv(x)
        x = x.transpose(2,1).contiguous() #memory clean for view
        x = F.log_softmax(x.view(-1,self.num_class), dim=-1)
        x = x.view(batchsize, pc_pts, self.num_class)
        return x, trans_feat



class STN3d(nn.Module):
    def __init__(self, quaternion=False):
        super(STN3d, self).__init__()
        self.quaternion = quaternion
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        if self.quaternion:
            self.fc3 = nn.Linear(256, 4)
        else:
            self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.quaternion:
            iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            if x.is_cuda:
                trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
            else:
                trans = Variable(torch.FloatTensor(batchsize, 3, 3))
            x = network_util.batch_quat_to_rotmat(x, trans)
        else:
            x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x