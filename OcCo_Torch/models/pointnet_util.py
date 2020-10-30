#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/fxia22/pointnet.pytorch/pointnet/model.py

import torch, torch.nn as nn, numpy as np, torch.nn.functional as F
from torch.autograd import Variable


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


# STN -> Spatial Transformer Network
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)  # in-channel, out-channel, kernel size
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]  # global descriptors

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32))).view(1, 9).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(
            1, self.k ** 2).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False,
                 channel=3, detailed=False):
        # when input include normals, it
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)  # Batch * 3 * 3
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.detailed = detailed

    def forward(self, x):

        _, D, N = x.size()  # Batch Size, Dimension of Point Features, Num of Points
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            # pdb.set_trace()
            x, feature = x.split([3, D-3], dim=2)
        x = torch.bmm(x, trans)
        # feature = torch.bmm(feature, trans)  # feature -> normals

        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        out1 = self.bn1(self.conv1(x))
        x = F.relu(out1)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        out2 = self.bn2(self.conv2(x))
        x = F.relu(out2)

        out3 = self.bn3(self.conv3(x))
        # x = self.bn3(self.conv3(x))
        x = torch.max(out3, 2, keepdim=False)[0]
        if self.global_feat:
            return x, trans, trans_feat
        elif self.detailed:
            return out1, out2, out3, x
        else:  # concatenate global and local feature together
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetPartSegEncoder(nn.Module):
    def __init__(self, feature_transform=True, channel=3):
        super(PointNetPartSegEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)
        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        if self.feature_transform:
            trans_feat = self.fstn(out3)
            net_transformed = torch.bmm(out3.transpose(2, 1), trans_feat)
            out3 = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(out3)))
        out5 = self.bn5(self.conv5(out4))

        out_max = torch.max(out5, 2, keepdim=False)[0]
        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        if self.feature_transform:
            return concat, trans_feat
        return concat


class encoder(nn.Module):
    def __init__(self, num_channel=3, **kwargs):
        super(encoder, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, channel=num_channel)

    def forward(self, x):
        feat, _, _ = self.feat(x)
        return feat


class detailed_encoder(nn.Module):
    def __init__(self, num_channel=3, **kwargs):
        super(detailed_encoder, self).__init__()
        self.feat = PointNetEncoder(global_feat=False,
                                    channel=num_channel,
                                    detailed=True)

    def forward(self, x):
        out1, out2, out3, x = self.feat(x)
        return out1, out2, out3, x