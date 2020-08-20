#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  ref: https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_semseg.py
#  ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_semseg.py

import os, sys, torch, argparse, importlib, shutil
sys.path.append('models')
sys.path.append('utils')
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from Torch_Utility import weights_init, bn_momentum_adjust
from ModelNetDataLoader import ModelNetJigsawDataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from TrainLogger import TrainLogger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('3D Point Cloud Jigsaw Puzzles')

    ''' === Training === '''
    parser.add_argument('--log_dir', type=str, help='log folder [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size [default: 32]')
    parser.add_argument('--epoch', default=200, type=int, help='training epochs [default: 200]')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimiser [default: Adam]')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum [default: 0.9]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='lr decay rate [default: 0.7]')
    parser.add_argument('--bn_decay', action='store_true', help='BN Momentum Decay [default: False]')
    parser.add_argument('--xavier_init', action='store_true', help='Xavier weight init [default: False]')
    parser.add_argument('--scheduler', type=str, default='step', help='lr decay scheduler [default: step]')
    parser.add_argument('--model', type=str, default='pointnet_jigsaw', help='model [default: pointnet_jigsaw]')
    parser.add_argument('--step_size', type=int, default=20, help='decay steps for lr [default: every 20 epochs]')

    ''' === Model === '''
    parser.add_argument('--k', type=int, default=20, help='num of nearest neighbors to use [default: 20]')
    parser.add_argument('--emb_dims', type=int, default=1024, help='dimension of embeddings [default: 1024]')
    parser.add_argument('--num_point', type=int, default=1024, help='number of points per object [default: 1024]')

    return parser.parse_args()


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    NUM_CLASSES = 3 ** 3
    DATA_PATH = 'data/modelnet40_ply_hdf5_2048/jigsaw'
    TRAIN_DATASET = ModelNetJigsawDataLoader(DATA_PATH, split='train', n_points=args.num_point, k=3)
    TEST_DATASET = ModelNetJigsawDataLoader(DATA_PATH, split='test', n_points=args.num_point, k=3)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    MyLogger = TrainLogger(args, name=args.model.upper(), subfold='jigsaw')

    ''' === MODEL LOADING === '''
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
    shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)
    writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))

    # allow multiple GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MODEL.get_model(args=args, num_class=NUM_CLASSES, num_channel=3).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier = torch.nn.DataParallel(classifier)
    print('=' * 33)
    print('Using %d GPU,' % torch.cuda.device_count(), 'Indices: %s' % args.gpu)
    print('=' * 33)

    if args.xavier_init:
        classifier = classifier.apply(weights_init)
        MyLogger.logger.info("Using Xavier Weight Initialisation")

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)
        MyLogger.logger.info("Using Adam Optimiser")
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=1000 * args.lr,
            momentum=args.momentum)
        MyLogger.logger.info("Using SGD Optimiser")

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-3)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.7)

    for epoch in range(MyLogger.epoch, args.epoch + 1):

        ''' === Training === '''
        MyLogger.epoch_init(training=True)

        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            points, target = points.transpose(2, 1).float().cuda(), target.view(-1, 1)[:, 0].long().cuda()
            classifier.train()
            optimizer.zero_grad()

            if args.model == 'pointnet_jigsaw':
                pred, trans_feat = classifier(points)
                pred = pred.contiguous().view(-1, NUM_CLASSES)
                loss = criterion(pred, target, trans_feat)
            else:
                pred = classifier(points)
                pred = pred.contiguous().view(-1, NUM_CLASSES)
                loss = criterion(pred, target)

            loss.backward()
            optimizer.step()
            # pdb.set_trace()
            MyLogger.step_update(pred.data.max(1)[1].cpu().numpy(),
                                 target.long().cpu().numpy(),
                                 loss.cpu().detach().numpy())
        MyLogger.epoch_summary(writer=writer, training=True)

        ''' === Evaluation === '''
        with torch.no_grad():
            classifier.eval()
            MyLogger.epoch_init(training=False)

            for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                points, target = points.transpose(2, 1).float().cuda(), target.view(-1, 1)[:, 0].long().cuda()
                if args.model == 'pointnet_jigsaw':
                    pred, trans_feat = classifier(points)
                    pred = pred.contiguous().view(-1, NUM_CLASSES)
                    loss = criterion(pred, target, trans_feat)
                else:
                    pred = classifier(points)
                    pred = pred.contiguous().view(-1, NUM_CLASSES)
                    loss = criterion(pred, target)
                MyLogger.step_update(pred.data.max(1)[1].cpu().numpy(),
                                     target.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())
            MyLogger.epoch_summary(writer=writer, training=False)

            if MyLogger.save_model:
                state = {
                    'step': MyLogger.step,
                    'epoch': MyLogger.best_instance_epoch,
                    'instance_acc': MyLogger.best_instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, MyLogger.savepath)

        scheduler.step()
        if args.scheduler == 'step':
            for param_group in optimizer.param_groups:
                if optimizer.param_groups[0]['lr'] < LEARNING_RATE_CLIP:
                    param_group['lr'] = LEARNING_RATE_CLIP
        if args.bn_decay:
            momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
            if momentum < 0.01:
                momentum = 0.01
            print('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

    MyLogger.train_summary()


if __name__ == '__main__':

    args = parse_args()
    main(args)

