#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://scikit-learn.org/stable/modules/svm.html
#  Ref: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

import os, sys, torch, argparse, datetime, importlib, numpy as np
sys.path.append('utils')
sys.path.append('models')
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ModelNetDataLoader import General_CLSDataLoader_HDF5
from Torch_Utility import copy_parameters
# from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from Dataset_Loc import Dataset_Loc
from sklearn import svm, metrics
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('SVM on Point Cloud Classification')

    ''' === Network Model === '''
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--model', default='pcn_util', help='model [default: pcn_util]')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--restore_path', type=str, help="path to pre-trained weights [default: None]")
    parser.add_argument('--grid_search', action='store_true', help='opt parameters via Grid Search [default: False]')

    ''' === Dataset === '''
    parser.add_argument('--partial', action='store_true', help='partial objects [default: False]')
    parser.add_argument('--bn', action='store_true', help='with background noise [default: False]')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset [default: modelnet40]')
    parser.add_argument('--fname', type=str, default="", help='filename, used in ScanObjectNN [default: ]')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    _, TRAIN_FILES, TEST_FILES = Dataset_Loc(dataset=args.dataset, fname=args.fname,
                                             partial=args.partial, bn=args.bn)
    TRAIN_DATASET = General_CLSDataLoader_HDF5(file_list=TRAIN_FILES)
    TEST_DATASET = General_CLSDataLoader_HDF5(file_list=TEST_FILES)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=Falses, num_workers=4)

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

        for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
            feats = encoder(points)
            X_test.append(feats.cpu().numpy())
            y_test.append(target.cpu().numpy())

    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

    # Optional: Standardize the Feature Space
    # X_train, X_test = scale(X_train), scale(X_test)

    ''' === Simple Trial === '''
    linear_svm = svm.SVC(kernel='linear')
    linear_svm.fit(X_train, y_train)
    y_pred = linear_svm.predict(X_test)
    print("\n", "Simple Linear SVC accuracy:", metrics.accuracy_score(y_test, y_pred), "\n")

    rbf_svm = svm.SVC(kernel='rbf')
    rbf_svm.fit(X_train, y_train)
    y_pred = rbf_svm.predict(X_test)
    print("Simple RBF SVC accuracy:", metrics.accuracy_score(y_test, y_pred), "\n")

    ''' === Grid Search for SVM with RBF Kernel === '''
    if not args.grid_search:
        sys.exit()
    print("Now we use Grid Search to opt the parameters for SVM RBF kernel")
    # [1e-3, 5e-3, 1e-2, ..., 5e1]
    gamma_range = np.outer(np.logspace(-3, 1, 5), np.array([1, 5])).flatten()
    # [1e-1, 5e-1, 1e0, ..., 5e1]
    C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5])).flatten()
    parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}

    svm_clsf = svm.SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=8, verbose=1)

    start_time = datetime.datetime.now()
    print('Start Param Searching at {}'.format(str(start_time)))
    grid_clsf.fit(X_train, y_train)
    print('Elapsed time, param searching {}'.format(str(datetime.datetime.now() - start_time)))
    sorted(grid_clsf.cv_results_.keys())

    # scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
    y_pred = grid_clsf.best_estimator_.predict(X_test)
    print("\n\n")
    print("="*37)
    print("Best Params via Grid Search Cross Validation on Train Split is: ", grid_clsf.best_params_)
    print("Best Model's Accuracy on Test Dataset: {}".format(metrics.accuracy_score(y_test, y_pred)))
