#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
import torch, os, random, numpy as np


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU Usage
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def copy_parameters(model, pretrained, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    model_dict = model.state_dict()
    pretrained_dict = pretrained['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def weights_init(m):
    """
    Xavier normal initialisation for weights and zero bias,
    find especially useful for completion and segmentation Tasks
    """
    classname = m.__class__.__name__
    if (classname.find('Conv1d') != -1) or (classname.find('Conv2d') != -1) or (classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
