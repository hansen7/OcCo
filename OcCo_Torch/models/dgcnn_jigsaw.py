#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py

import torch, torch.nn as nn, torch.nn.functional as F
from dgcnn_util import get_graph_feature


class get_model(nn.Module):
    def __init__(self, args, num_class, **kwargs):
        super(get_model, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, num_class, kernel_size=1, bias=False)

    def forward(self, x):

        batch_size, _, num_points = x.size()

        x = get_graph_feature(x, self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        # x = F.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)
        '''add softmax: 
            https://towardsdatascience.com/cuda-error-device-side-assert-triggered-c6ae1c8fa4c3
            https://github.com/pytorch/pytorch/issues/1204
        '''
        return x


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    @staticmethod
    def cal_loss(pred, gold, smoothing=False):
        """Calculate cross entropy loss, apply label smoothing if needed."""

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

    def forward(self, pred, target):
        return self.cal_loss(pred, target, smoothing=False)
