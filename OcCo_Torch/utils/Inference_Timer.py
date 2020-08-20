#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, torch, time, numpy as np

class Inference_Timer:
    def __init__(self, args):
        self.args = args
        self.est_total = []
        self.use_cpu = True if (self.args.gpu == 'None') else False
        self.device = 'CPU' if self.use_cpu else 'GPU'
        if self.use_cpu:
            os.environ['OMP_NUM_THREADS'] = "1"
            os.environ['MKL_NUM_THREADS'] = "1"
            print('Now we calculate the inference time on a single CPU')
        else:
            print('Now we calculate the inference time on a single GPU')
        self.args.batch_size, self.args.epoch = 2, 1
    #  1D BatchNorm requires more than 1 sample to compute std
    #  ref: https://github.com/pytorch/pytorch/issues/7716
    #  otherwise remove the 1D BatchNorm,
    #  since its contribution to the inference is negligible
    #  ref:

    def update_args(self):
        return self.args

    def single_step(self, model, data):
        if not self.use_cpu:
            torch.cuda.synchronize()
        start = time.time()
        output = model(data)
        if not self.use_cpu:
            torch.cuda.synchronize()
        end = time.time()
        self.est_total.append(end - start)
        return output

    def update_single_epoch(self, logger):
        logger.info("Model: {}".format(self.args.model))
        logger.info("Average Inference Time Per Example on Single {}: {:.3f} milliseconds".format(
            self.device, np.mean(self.est_total)*1000))
