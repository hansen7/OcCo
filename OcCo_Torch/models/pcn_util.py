#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch, torch.nn as nn, torch.nn.functional as F

class PCNEncoder(nn.Module):
    def __init__(self, global_feat=False, channel=3):
        super(PCNEncoder, self).__init__()

        self.conv1 = nn.Conv1d(channel, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)  no bn in PCN
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.global_feat = global_feat

    def forward(self, x):
        _, D, N = x.size()
        x = F.relu(self.conv1(x))
        pointfeat = self.conv2(x)

        # 'encoder_0'
        feat = torch.max(pointfeat, 2, keepdim=True)[0]
        feat = feat.view(-1, 256, 1).repeat(1, 1, N)
        x = torch.cat([pointfeat, feat], 1)

        # 'encoder_1'
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=False)[0]

        if self.global_feat:  # used in completion and classification tasks
            return x
        else:  # concatenate global and local features, for segmentation tasks
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)


class PCNPartSegEncoder(nn.Module):
    def __init__(self, channel=3):
        super(PCNPartSegEncoder, self).__init__()

        self.conv1 = nn.Conv1d(channel, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 2048, 1)

    def forward(self, x, label):
        _, D, N = x.size()
        out1 = F.relu(self.conv1(x))
        out2 = self.conv2(out1)

        # 'encoder_0'
        feat = torch.max(out2, 2, keepdim=True)[0]
        feat = feat.repeat(1, 1, N)
        out3 = torch.cat([out2, feat], 1)

        # 'encoder_1'
        out4 = F.relu(self.conv3(out3))
        out5 = self.conv4(out4)

        out_max = torch.max(out5, 2, keepdim=False)[0]
        out_max = torch.cat([out_max, label.squeeze(1)], 1)

        expand = out_max.view(-1, 2064, 1).repeat(1, 1, N)  # (batch, 2064, num_point)
        concat = torch.cat([expand, out1, out3, out4, out5], 1)

        return concat


class encoder(nn.Module):
    def __init__(self, num_channel=3, **kwargs):
        super(encoder, self).__init__()
        self.feat = PCNEncoder(global_feat=True, channel=num_channel)

    def forward(self, x):
        return self.feat(x)


if __name__ == "__main__":
    # model = PCNEncoder()
    model = PCNPartSegEncoder()
    xyz = torch.rand(16, 3, 100)  # batch, channel, num_point
    label = torch.randint(low=0, high=20, size=(16, 1, 12)).float()
    x = model(xyz, label)
    print(x.size())
