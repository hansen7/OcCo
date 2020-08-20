#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  ref: https://github.com/charlesq34/pointnet/blob/master/sem_seg/train.py
#  ref: https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_semseg.py
#  ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_semseg.py

import os, sys, torch, shutil, argparse, importlib
sys.path.append('utils')
sys.path.append('models')
from Torch_Utility import copy_parameters, weights_init, bn_momentum_adjust
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from S3DISDataLoader import S3DISDataset_HDF5
from torch.utils.data import DataLoader
from TrainLogger import TrainLogger
from tqdm import tqdm


classes = ['ceiling', 'floor', 'wall', 'beam', 'column',
           'window', 'door', 'table', 'chair', 'sofa',
           'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')

    parser.add_argument('--log_dir', type=str, help='log path [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--test_area', type=int, default=5, help='test area, 1-6 [default: 5]')
    parser.add_argument('--epoch', default=100, type=int, help='training epochs [default: 100]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum [default: 0.9]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')
    parser.add_argument('--restore', action='store_true', help='restore the weights [default: False]')
    parser.add_argument('--restore_path', type=str, help='path to pre-saved model weights [default: ]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate in FCs [default: 0.5]')
    parser.add_argument('--bn_decay', action='store_true', help='use BN Momentum Decay [default: False]')
    parser.add_argument('--xavier_init', action='store_true', help='Xavier weight init [default: False]')
    parser.add_argument('--emb_dims', type=int, default=1024, help='embedding dimensions [default: 1024]')
    parser.add_argument('--k', type=int, default=20, help='num of nearest neighbors to use [default: 20]')
    parser.add_argument('--step_size', type=int, default=40, help='lr decay steps [default: every 40 epochs]')
    parser.add_argument('--scheduler', type=str, default='cos', help='lr decay scheduler [default: cos, step]')
    parser.add_argument('--model', type=str, default='pointnet_semseg', help='model [default: pointnet_semseg]')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimiser [default: adam, otherwise sgd]')

    return parser.parse_args()


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    root = 'data/indoor3d_sem_seg_hdf5_data'
    NUM_CLASSES = len(seg_label_to_cat)

    TRAIN_DATASET = S3DISDataset_HDF5(root=root, split='train', test_area=args.test_area)
    TEST_DATASET = S3DISDataset_HDF5(root=root, split='test', test_area=args.test_area)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    MyLogger = TrainLogger(args, name=args.model.upper(), subfold='semseg',
                           cls2name=class2label, filename=args.mode + '_log')
    MyLogger.logger.info("The number of training data is: %d" % len(TRAIN_DATASET))
    MyLogger.logger.info("The number of testing data is: %d" % len(TEST_DATASET))

    ''' === Model Loading === '''
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
    shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)
    writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MODEL.get_model(num_class=NUM_CLASSES, num_channel=9, args=args).to(device)
    criterion = MODEL.get_loss().to(device)
    classifier = torch.nn.DataParallel(classifier)
    print('=' * 27)
    print('Using %d GPU,' % torch.cuda.device_count(), 'Indices: %s' % args.gpu)
    print('=' * 27)

    if args.restore:
        checkpoint = torch.load(args.restore_path)
        classifier = copy_parameters(classifier, checkpoint, verbose=True)
        MyLogger.logger.info('Use pre-trained weights from %s' % args.restore_path)
    else:
        MyLogger.logger.info('No pre-trained weights, start training from scratch...')
        if args.xavier_init:
            classifier = classifier.apply(weights_init)
            MyLogger.logger.info("Using Xavier weight initialisation")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)
        MyLogger.logger.info("Using Adam optimiser")
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.lr * 100,
            momentum=args.momentum)
        MyLogger.logger.info("Using SGD optimiser")
    # using the similar lr decay setting from
    # https://github.com/charlesq34/pointnet/blob/master/sem_seg/train.py
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-3)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    ''' === Testing then Exit === '''
    if args.mode == 'test':
        with torch.no_grad():
            classifier.eval()
            MyLogger.epoch_init(training=False)

            for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                points, target = points.transpose(2, 1).float().cuda(), target.view(-1, 1)[:, 0].long().cuda()
                if args.model == 'pointnet_semseg':
                    seg_pred, trans_feat = classifier(points)
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                    loss = criterion(seg_pred, target, trans_feat)
                else:
                    seg_pred = classifier(points)
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                    loss = criterion(seg_pred, target)
                MyLogger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                     target.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())

            MyLogger.epoch_summary(writer=writer, training=False, mode='semseg')
        sys.exit("Test Finished")

    for epoch in range(MyLogger.epoch, args.epoch + 1):

        ''' === Training === '''
        # scheduler.step()
        MyLogger.epoch_init()

        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            writer.add_scalar('learning rate', scheduler.get_lr()[-1], MyLogger.step)
            points, target = points.float().transpose(2, 1).cuda(), target.view(-1, 1)[:, 0].long().cuda()

            classifier.train()
            optimizer.zero_grad()
            # pdb.set_trace()
            if args.model == 'pointnet_semseg':
                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                loss = criterion(seg_pred, target, trans_feat)
            else:
                seg_pred = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                loss = criterion(seg_pred, target)

            loss.backward()
            optimizer.step()

            MyLogger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                 target.long().cpu().numpy(),
                                 loss.cpu().detach().numpy())
        MyLogger.epoch_summary(writer=writer, training=True, mode='semseg')

        '''=== Evaluating ==='''
        with torch.no_grad():
            classifier.eval()
            MyLogger.epoch_init(training=False)

            for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                points, target = points.transpose(2, 1).float().cuda(), target.view(-1, 1)[:, 0].long().cuda()
                if args.model == 'pointnet_semseg':
                    seg_pred, trans_feat = classifier(points)
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                    loss = criterion(seg_pred, target, trans_feat)
                else:
                    seg_pred = classifier(points)
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                    loss = criterion(seg_pred, target)
                MyLogger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                     target.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())

            MyLogger.epoch_summary(writer=writer, training=False, mode='semseg')
            if MyLogger.save_model:
                state = {
                    'step': MyLogger.step,
                    'miou': MyLogger.best_miou,
                    'epoch': MyLogger.best_miou_epoch,
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

    MyLogger.train_summary(mode='semseg')


if __name__ == '__main__':
    args = parse_args()
    main(args)
