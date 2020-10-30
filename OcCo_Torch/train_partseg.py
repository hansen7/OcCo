#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_partseg.py

import os, sys, torch, shutil, importlib, argparse, numpy as np
sys.path.append('utils')
sys.path.append('models')
from PC_Augmentation import random_scale_point_cloud, random_shift_point_cloud
from Torch_Utility import copy_parameters, weights_init, bn_momentum_adjust
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader
from TrainLogger import TrainLogger
from tqdm import tqdm


seg_classes = {'Earphone':   [16, 17, 18],
               'Motorbike':  [30, 31, 32, 33, 34, 35],
               'Rocket':     [41, 42, 43],
               'Car':        [8, 9, 10, 11],
               'Laptop':     [28, 29],
               'Cap':        [6, 7],
               'Skateboard': [44, 45, 46],
               'Mug':        [36, 37],
               'Guitar':     [19, 20, 21],
               'Bag':        [4, 5],
               'Lamp':       [24, 25, 26, 27],
               'Table':      [47, 48, 49],
               'Airplane':   [0, 1, 2, 3],
               'Pistol':     [38, 39, 40],
               'Chair':      [12, 13, 14, 15],
               'Knife':      [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ..., 49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--log_dir', type=str, help='log folder [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--epoch', default=250, type=int, help=' epochs [default: 250]')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size [default: 16]')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum [default: 0.9]')
    parser.add_argument('--restore_path', type=str, help='path to pretrained weights [default: ]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')
    parser.add_argument('--num_point', type=int, default=2048, help='point number [default: 2048]')
    parser.add_argument('--restore', action='store_true', help='using pre-trained [default: False]')
    parser.add_argument('--use_sgd', action='store_true', help='use SGD optimiser [default: False]')
    parser.add_argument('--data_aug', action='store_true', help='data augmentation [default: False]')
    parser.add_argument('--scheduler', default='step', help='learning rate scheduler [default: step]')
    parser.add_argument('--model', default='pointnet_partseg', help='model [default: pointnet_partseg]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate in FCs [default: 0.5]')
    parser.add_argument('--bn_decay', action='store_true', help='use BN nomentum decay [default: False]')
    parser.add_argument('--xavier_init', action='store_true', help='Xavier weight init [default: False]')
    parser.add_argument('--emb_dims', type=int, default=1024, help='embedding dimensions [default: 1024]')
    parser.add_argument('--k', type=int, default=20, help='num of nearest neighbors to use [default: 20]')
    parser.add_argument('--normal', action='store_true', default=False, help='use normal [default: False]')
    parser.add_argument('--step_size', type=int, default=20, help='lr decay step [default: every 20 epochs]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate test predictions via vote [default: 3]')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    def to_categorical(y, num_class):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_class)[y.cpu().data.numpy(), ]
        if y.is_cuda:
            return new_y.cuda()
        return new_y

    ''' === Set up Loggers and Load Data === '''
    MyLogger = TrainLogger(args, name=args.model.upper(), subfold='partseg',
                           filename=args.mode + '_log', cls2name=seg_label_to_cat)
    writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TRAIN_DATASET = PartNormalDataset(root=root, num_point=args.num_point, split='trainval', use_normal=args.normal)
    TEST_DATASET = PartNormalDataset(root=root, num_point=args.num_point, split='test', use_normal=args.normal)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes, num_part = 16, 50

    ''' === Load Model and Backup Scripts === '''
    channel_num = 6 if args.normal else 3
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
    shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MODEL.get_model(part_num=num_part, num_channel=channel_num, args=args).cuda().to(device)
    criterion = MODEL.get_loss().to(device)
    classifier = torch.nn.DataParallel(classifier)

    if args.restore:
        checkpoint = torch.load(args.restore_path)
        classifier = copy_parameters(classifier, checkpoint, verbose=True)
        MyLogger.logger.info('Use pre-trained weights from %s' % args.restore_path)
    else:
        MyLogger.logger.info('No pre-trained weights, start training from scratch...')
        if args.xavier_init:
            classifier = classifier.apply(weights_init)
            MyLogger.logger.info("Using Xavier weight initialisation")

    if args.mode == 'test':
        MyLogger.logger.info('\n\n')
        MyLogger.logger.info('=' * 33)
        MyLogger.logger.info('load parrameters from %s' % args.restore_path)
        with torch.no_grad():
            test_metrics = {}
            total_correct, total_seen = 0, 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}  # {shape: []}

            for points, label, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                classifier.eval()
                cur_batch_size, num_point, _ = points.size()
                vote_pool = torch.zeros(cur_batch_size, num_point, num_part).cuda()  # (batch, num point, num part)
                points, label, target = points.transpose(2, 1).float().cuda(), label.long().cuda(), target.numpy()
                
                ''' === generate predictions from raw output (multiple via voting) === '''
                for _ in range(args.num_votes):
                    if args.model == 'pointnet_partseg':
                        seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                    else:
                        seg_pred = classifier(points, to_categorical(label, num_classes))
                    vote_pool += seg_pred  # added on probability
                
                seg_pred = vote_pool / args.num_votes
                cur_pred_val_logits = seg_pred.cpu().data.numpy()
                cur_pred_val = np.zeros((cur_batch_size, num_point)).astype(np.int32)

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]  # str, shape name
                    logits = cur_pred_val_logits[i, :, :]  # array, (num point, num part)
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0] 
                    # only consider parts from that shape

                ''' === calculate accuracy === '''
                total_correct += np.sum(cur_pred_val == target)
                total_seen += (cur_batch_size * num_point)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                ''' === calculate iou === '''
                for i in range(cur_batch_size):
                    segl = target[i, :]  # array, (num point, )
                    segp = cur_pred_val[i, :]  # array, (num point, )
                    cat = seg_label_to_cat[segl[0]]  # str, shape name
                    part_ious = [0. for _ in range(len(seg_classes[cat]))]  # parts belong to that shape
                    
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # no prediction or gt
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                            part_ious[l - seg_classes[cat][0]] = iou
                    shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
            
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            MyLogger.logger.info('test mIoU of %-14s %f' % (cat, shape_ious[cat]))

        MyLogger.logger.info('Accuracy is: %.5f' % test_metrics['accuracy'])
        MyLogger.logger.info('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
        MyLogger.logger.info('Class avg mIoU is: %.5f' % test_metrics['class_avg_iou'])
        MyLogger.logger.info('Instance avg mIoU is: %.5f' % test_metrics['instance_avg_iou'])
        sys.exit("Test Finished")

    if not args.use_sgd:
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr * 100,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    if args.scheduler is 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-3)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    for epoch in range(MyLogger.epoch, args.epoch + 1):

        MyLogger.epoch_init()

        for points, label, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):

            if args.data_aug:
                points = points.data.numpy()
                points[:, :, :3] = random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, :3] = random_shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)

            points, label, target = points.transpose(2, 1).float().cuda(), label.long().cuda(), \
                                    target.view(-1, 1)[:, 0].long().cuda()
            classifier.train()
            optimizer.zero_grad()
            if args.model == 'pointnet_partseg':
                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                loss = criterion(seg_pred, target, trans_feat)
            else:
                seg_pred = classifier(points, to_categorical(label, num_classes))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                loss = criterion(seg_pred, target)

            loss.backward()
            optimizer.step()
            MyLogger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                 target.long().cpu().numpy(),
                                 loss.cpu().detach().numpy())
        MyLogger.epoch_summary(writer=writer, training=True, mode='partseg')

        '''=== Evaluating ==='''
        with torch.no_grad():

            classifier.eval()
            MyLogger.epoch_init(training=False)

            for points, label, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.transpose(2, 1).float().cuda(), label.long().cuda(), \
                                        target.view(-1, 1)[:, 0].long().cuda()
                if args.model == 'pointnet_partseg':
                    seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                    seg_pred = seg_pred.contiguous().view(-1, num_part)
                    loss = criterion(seg_pred, target, trans_feat)
                else:
                    seg_pred = classifier(points, to_categorical(label, num_classes))
                    seg_pred = seg_pred.contiguous().view(-1, num_part)
                    loss = criterion(seg_pred, target)

                MyLogger.step_update(seg_pred.data.max(1)[1].cpu().numpy(),
                                     target.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())
            
            MyLogger.epoch_summary(writer=writer, training=False, mode='partseg')

            if MyLogger.save_model:
                state = {
                    'step': MyLogger.step,
                    'miou': MyLogger.best_miou,
                    'epoch': MyLogger.best_miou_epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, MyLogger.savepath)

            if epoch % 5 == 0:
                state = {
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, MyLogger.savepath.replace('best_model', 'model_ep%d' % epoch))

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


if __name__ == '__main__':
    args = parse_args()
    main(args)
