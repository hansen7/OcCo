#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py
#  Sanity Check: https://github.com/vinits5/learning3d/blob/master/models/pcn.py

import torch, torch.nn as nn, torch.nn.functional as F

class PCNEncoder(nn.Module):
    def __init__(self, global_feat=False, channel=3):
        super(PCNEncoder, self).__init__()

        self.conv1 = nn.Conv1d(channel, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)  no bn in PCN encoders
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.global_feat = global_feat

    def forward(self, x):
        _, D, N = x.size()  # (batch size, dimension of features, num of points)

        x = F.relu(self.conv1(x))
        pointfeat = self.conv2(x)

        # 'encoder_0'
        feat = torch.max(pointfeat, 2, keepdim=True)[0]
        feat = feat.view(-1, 256, 1).repeat(1, 1, N)
        x = torch.cat([pointfeat, feat], 1)

        # 'encoder_1'
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:  # used in completion and classification tasks
            return x
        else:  # concatenate global and local features, for segmentation tasks
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)


if __name__ == "__main__":
    model = PCNEncoder(global_feat=False)
    xyz = torch.rand(12, 3, 100)
    x = model(xyz)
    print(x.size())
