#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

import os, sys, torch, argparse, importlib, numpy as np, matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../models')
from ModelNetDataLoader import General_CLSDataLoader_HDF5
from Torch_Utility import copy_parameters
from torch.utils.data import DataLoader
from Dataset_Loc import Dataset_Loc
from sklearn.manifold import TSNE
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('SVM on Point Cloud Classification')

    ''' === Network Model === '''
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--model', default='pcn_util', help='model [default: pcn_util]')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--restore_path', type=str, help="path to pretrained weights [default: None]")

    ''' === Dataset === '''
    parser.add_argument('--partial', action='store_true', help='partial objects [default: False]')
    parser.add_argument('--bn', action='store_true', help='with background noise [default: False]')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset [default: modelnet40]')
    parser.add_argument('--fname', type=str, help='filename, used in ScanObjectNN or fewer data [default:]')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    NUM_CLASSES, TRAIN_FILES, TEST_FILES = Dataset_Loc(dataset=args.dataset, fname=args.fname,
                                                       partial=args.partial, bn=args.bn)
    TRAIN_DATASET = General_CLSDataLoader_HDF5(file_list=TRAIN_FILES)
    # TEST_DATASET = General_CLSDataLoader_HDF5(file_list=TEST_FILES)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)

    MODEL = importlib.import_module(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MODEL.encoder(args=args, num_channel=3).to(device)
    encoder = torch.nn.DataParallel(encoder)

    checkpoint = torch.load(args.restore_path)
    encoder = copy_parameters(encoder, checkpoint, verbose=True)

    X_train, y_train, X_test, y_test = [], [], [], []
    with torch.no_grad():
        encoder.eval()

        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
            feats = encoder(points)
            X_train.append(feats.cpu().numpy())
            y_train.append(target.cpu().numpy())

        # for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
        #     points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
        #     feats = encoder(points)
        #     X_test.append(feats.cpu().numpy())
        #     y_test.append(target.cpu().numpy())

    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    # X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

    # In general, larger dataset/num of class require larger perplexity
    X_embedded = TSNE(n_components=2, perplexity=100).fit_transform(X_train)

    plt.figure(figsize=(16, 16))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train, cmap=plt.cm.get_cmap("jet", NUM_CLASSES))
    plt.colorbar(ticks=range(1, NUM_CLASSES + 1))
    plt.clim(0.5, NUM_CLASSES + 0.5)
    # plt.savefig('log/tsne/tsne_shapenet10_pcn.pdf')
    plt.show()

