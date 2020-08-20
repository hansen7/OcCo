#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, logging, datetime, numpy as np, sklearn.metrics as metrics
from pathlib import Path


class TrainLogger:

    def __init__(self, args, name='model', subfold='cls', filename='train_log', cls2name=None):
        self.step = 1
        self.epoch = 1
        self.args = args
        self.name = name
        self.sf = subfold
        self.mkdir()
        self.setup(filename=filename)
        self.epoch_init()
        self.save_model = False
        self.cls2name = cls2name
        self.best_instance_acc, self.best_class_acc, self.best_miou = 0., 0., 0.
        self.best_instance_epoch, self.best_class_epoch, self.best_miou_epoch = 0, 0, 0
        self.savepath = str(self.checkpoints_dir) + '/best_model.pth'

    def setup(self, filename='train_log'):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.log_dir, filename + '.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # ref: https://stackoverflow.com/a/53496263/12525201
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # logging.getLogger('').addHandler(console) # this is root logger
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)
        self.logger.removeHandler(console)

    def mkdir(self):
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = Path('./log/')
        experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath(self.sf)
        experiment_dir.mkdir(exist_ok=True)

        if self.args.log_dir is None:
            self.experiment_dir = experiment_dir.joinpath(timestr)
        else:
            self.experiment_dir = experiment_dir.joinpath(self.args.log_dir)

        self.experiment_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.experiment_dir.joinpath('checkpoints/')
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.log_dir = self.experiment_dir.joinpath('logs/')
        self.log_dir.mkdir(exist_ok=True)

    # @property.setter
    def epoch_init(self, training=True):
        self.loss, self.count, self.pred, self.gt = 0., 0., [], []
        if training:
            self.logger.info('Epoch %d/%d:' % (self.epoch, self.args.epoch))

    def step_update(self, pred, gt, loss, training=True):
        if training:
            self.step += 1  # Use TensorFlow way to count training steps
        self.gt.append(gt)
        self.pred.append(pred)
        batch_size = len(pred)
        self.count += batch_size
        self.loss += loss * batch_size

    def epoch_update(self, training=True, mode='cls'):
        self.save_model = False
        self.gt = np.concatenate(self.gt)
        self.pred = np.concatenate(self.pred)

        instance_acc = metrics.accuracy_score(self.gt, self.pred)
        if instance_acc > self.best_instance_acc and not training:
            self.save_model = True if mode == 'cls' else False
            self.best_instance_acc = instance_acc
            self.best_instance_epoch = self.epoch

        if mode == 'cls':
            class_acc = metrics.balanced_accuracy_score(self.gt, self.pred)
            if class_acc > self.best_class_acc and not training:
                self.best_class_epoch = self.epoch
                self.best_class_acc = class_acc
            return instance_acc, class_acc
        elif mode == 'semseg':
            miou = self.calculate_IoU().mean()
            if miou > self.best_miou and not training:
                self.best_miou_epoch = self.epoch
                self.save_model = True
                self.best_miou = miou
            return instance_acc, miou
        else:
            raise ValueError('Mode is not Supported by TrainLogger')

    def epoch_summary(self, writer=None, training=True, mode='cls'):
        criteria = 'Class Accuracy' if mode == 'cls' else 'mIoU'
        instance_acc, class_acc = self.epoch_update(training=training, mode=mode)
        if training:
            if writer is not None:
                writer.add_scalar('Train Instance Accuracy', instance_acc, self.step)
                writer.add_scalar('Train %s' % criteria, class_acc, self.step)
            self.logger.info('Train Instance Accuracy: %.3f' % instance_acc)
            self.logger.info('Train %s: %.3f' % (criteria, class_acc))
        else:
            if writer is not None:
                writer.add_scalar('Test Instance Accuracy', instance_acc, self.step)
                writer.add_scalar('Test %s' % criteria, class_acc, self.step)
            self.logger.info('Test Instance Accuracy: %.3f' % instance_acc)
            self.logger.info('Test %s: %.3f' % (criteria, class_acc))
            self.logger.info('Best Instance Accuracy: %.3f at Epoch %d ' % (
                self.best_instance_acc, self.best_instance_epoch))
            if self.best_class_acc > .1:
                self.logger.info('Best Class Accuracy: %.3f at Epoch %d' % (
                    self.best_class_acc, self.best_class_epoch))
            if self.best_miou > .1:
                self.logger.info('Best mIoU: %.3f at Epoch %d' % (
                    self.best_miou, self.best_miou_epoch))

        self.epoch += 1 if not training else 0
        if self.save_model:
            self.logger.info('Saving the Model Params to %s' % self.savepath)

    def calculate_IoU(self):
        num_class = len(self.cls2name)
        Intersection = np.zeros(num_class)
        Union = Intersection.copy()
        # self.pred -> numpy.ndarray (total predictions, )

        for sem_idx in range(num_class):
            Intersection[sem_idx] = np.sum(np.logical_and(self.pred == sem_idx, self.gt == sem_idx))
            Union[sem_idx] = np.sum(np.logical_or(self.pred == sem_idx, self.gt == sem_idx))
        return Intersection / Union

    def train_summary(self, mode='cls'):
        self.logger.info('\n\nEnd of Training...')
        self.logger.info('Best Instance Accuracy: %.3f at Epoch %d ' % (
            self.best_instance_acc, self.best_instance_epoch))
        if mode == 'cls':
            self.logger.info('Best Class Accuracy: %.3f at Epoch %d' % (
                self.best_class_acc, self.best_class_epoch))
        elif mode == 'semseg':
            self.logger.info('Best mIoU: %.3f at Epoch %d' % (
                self.best_miou, self.best_miou_epoch))

    def update_from_checkpoints(self, checkpoint):
        self.logger.info('Use Pre-Trained Weights')
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_instance_epoch, self.best_instance_acc = checkpoint['epoch'], checkpoint['instance_acc']
        self.best_class_epoch, self.best_class_acc = checkpoint['best_class_epoch'], checkpoint['best_class_acc']
        self.logger.info('Best Class Acc {:.3f} at Epoch {}'.format(self.best_instance_acc, self.best_class_epoch))
        self.logger.info('Best Instance Acc {:.3f} at Epoch {}'.format(self.best_instance_acc, self.best_instance_epoch))
