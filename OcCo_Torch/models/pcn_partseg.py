#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch, torch.nn as nn, torch.nn.functional as F
from pcn_util import PCNPartSegEncoder


class get_model(nn.Module):
    def __init__(self, part_num=50, num_channel=3, **kwargs):
        super(get_model, self).__init__()
        self.part_num = part_num
        self.feat = PCNPartSegEncoder(channel=num_channel)

        self.convs1 = nn.Conv1d(5264, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, _, N = point_cloud.size()
        x = self.feat(point_cloud, label)
        x = F.relu(self.bns1(self.convs1(x)))
        x = F.relu(self.bns2(self.convs2(x)))
        x = F.relu(self.bns3(self.convs3(x)))
        x = self.convs4(x).transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.part_num), dim=-1)
        x = x.view(B, N, self.part_num)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss


if __name__ == '__main__':
    model = get_model(part_num=50, num_channel=3)
    xyz = torch.rand(16, 3, 4096)
    label = torch.randint(low=0, high=20, size=(16, 1, 16)).float()
    model(xyz, label)
