#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch, torch.nn as nn, torch.nn.functional as F
from pcn_util import PCNEncoder


class get_model(nn.Module):
    def __init__(self, num_class, num_channel=9, **kwargs):
        super(get_model, self).__init__()
        self.num_class = num_class
        self.feat = PCNEncoder(global_feat=False, channel=num_channel)
        self.conv1 = nn.Conv1d(1280, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.num_class, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_class), dim=-1)
        x = x.view(batch_size, num_points, self.num_class)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss


if __name__ == '__main__':
    model = get_model(num_class=13, num_channel=3)
    xyz = torch.rand(12, 3, 2048)
    model(xyz)
