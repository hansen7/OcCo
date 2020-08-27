#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/main.py
#  Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_cls.py

import os, sys, torch, shutil, importlib, argparse
sys.path.append('utils')
sys.path.append('models')
from PC_Augmentation import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from ModelNetDataLoader import General_CLSDataLoader_HDF5
from Torch_Utility import copy_parameters, seed_torch
from torch.utils.tensorboard import SummaryWriter
# from Inference_Timer import Inference_Timer
from torch.utils.data import DataLoader
from Dataset_Loc import Dataset_Loc
from TrainLogger import TrainLogger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Classification')

    ''' === Training and Model === '''
    parser.add_argument('--log_dir', type=str, help='log folder [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--epoch', type=int, default=200, help='epochs [default: 200]')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model [default: pointnet_cls]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate [default: 0.5]')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum [default: 0.9]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')
    parser.add_argument('--step_size', type=int, default=20, help='lr decay step [default: 20 eps]')
    parser.add_argument('--num_point', type=int, default=1024, help='points number [default: 1024]')
    parser.add_argument('--restore', action='store_true', help='using pre-trained [default: False]')
    parser.add_argument('--restore_path', type=str, help="path to pretrained weights [default: None]")
    parser.add_argument('--emb_dims', type=int, default=1024, help='dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=20, help='number of nearest neighbors to use [default: 20]')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='use SGD optimiser [default: False]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001, 0.1 if using sgd]')
    parser.add_argument('--scheduler', type=str, default='step', help='lr decay scheduler [default: step, or cos]')

    ''' === Dataset === '''
    parser.add_argument('--partial', action='store_true', help='partial objects [default: False]')
    parser.add_argument('--bn', action='store_true', help='with background noise [default: False]')
    parser.add_argument('--data_aug', action='store_true', help='data Augmentation [default: False]')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset [default: modelnet40]')
    parser.add_argument('--fname', type=str, help='filename, used in ScanObjectNN or fewer data [default:]')

    return parser.parse_args()


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # seed_torch(args.seed)

    ''' === Set up Loggers and Load Data === '''
    MyLogger = TrainLogger(args, name=args.model.upper(), subfold='cls', filename=args.mode + '_log')
    writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))

    MyLogger.logger.info('Load dataset %s' % args.dataset)
    NUM_CLASSES, TRAIN_FILES, TEST_FILES = Dataset_Loc(dataset=args.dataset, fname=args.fname,
                                                       partial=args.partial, bn=args.bn)
    TRAIN_DATASET = General_CLSDataLoader_HDF5(file_list=TRAIN_FILES, num_point=1024)
    TEST_DATASET = General_CLSDataLoader_HDF5(file_list=TEST_FILES, num_point=1024)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ''' === Load Model and Backup Scripts === '''
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
    shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MODEL.get_model(args=args, num_channel=3, num_class=NUM_CLASSES).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier = torch.nn.DataParallel(classifier)
    # nn.DataParallel has its own issues (slow, memory expensive),
    # here are some advanced solutions: https://zhuanlan.zhihu.com/p/145427849
    print('=' * 27)
    print('Using %d GPU,' % torch.cuda.device_count(), 'Indices: %s' % args.gpu)
    print('=' * 27)

    ''' === Restore Model from Pre-Trained Checkpoints: OcCo/Jigsaw etc === '''
    if args.restore:
        checkpoint = torch.load(args.restore_path)
        classifier = copy_parameters(classifier, checkpoint, verbose=True)
        MyLogger.logger.info('Use pre-trained weights from %s' % args.restore_path)
    else:
        MyLogger.logger.info('No pre-trained weights, start training from scratch...')

    if not args.use_sgd:
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr * 100,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-3)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    LEARNING_RATE_CLIP = 0.01 * args.lr

    if args.mode == 'test':
        with torch.no_grad():
            classifier.eval()
            MyLogger.epoch_init(training=False)

            for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
                if args.model == 'pointnet_cls':
                    pred, trans_feat = classifier(points)
                    loss = criterion(pred, target, trans_feat)
                else:
                    pred = classifier(points)
                    loss = criterion(pred, target)
                MyLogger.step_update(pred.data.max(1)[1].cpu().numpy(),
                                     target.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())

            MyLogger.epoch_summary(writer=writer, training=False)
        sys.exit("Test Finished")

    for epoch in range(MyLogger.epoch, args.epoch + 1):

        ''' === Training === '''
        MyLogger.epoch_init()

        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            writer.add_scalar('Learning Rate', scheduler.get_lr()[-1], MyLogger.step)

            # Augmentation, might bring performance gains
            if args.data_aug:
                points = random_point_dropout(points.data.numpy())
                points[:, :, :3] = random_scale_point_cloud(points[:, :, :3])
                points[:, :, :3] = random_shift_point_cloud(points[:, :, :3])
                points = torch.Tensor(points)

            points, target = points.transpose(2, 1).float().cuda(), target.long().cuda()

            # FP and BP
            classifier.train()
            optimizer.zero_grad()
            if args.model == 'pointnet_cls':
                pred, trans_feat = classifier(points)
                loss = criterion(pred, target, trans_feat)
            else:
                pred = classifier(points)
                loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            MyLogger.step_update(pred.data.max(1)[1].cpu().numpy(),
                                 target.long().cpu().numpy(),
                                 loss.cpu().detach().numpy())
        MyLogger.epoch_summary(writer=writer, training=True)

        ''' === Validating === '''
        with torch.no_grad():
            classifier.eval()
            MyLogger.epoch_init(training=False)

            for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
                if args.model == 'pointnet_cls':
                    pred, trans_feat = classifier(points)
                    loss = criterion(pred, target, trans_feat)
                else:
                    pred = classifier(points)
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
                    'best_class_acc': MyLogger.best_class_acc,
                    'best_class_epoch': MyLogger.best_class_epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, MyLogger.savepath)

        scheduler.step()
        if args.scheduler == 'step':
            for param_group in optimizer.param_groups:
                if optimizer.param_groups[0]['lr'] < LEARNING_RATE_CLIP:
                    param_group['lr'] = LEARNING_RATE_CLIP

    MyLogger.train_summary()


if __name__ == '__main__':

    args = parse_args()
    main(args)
