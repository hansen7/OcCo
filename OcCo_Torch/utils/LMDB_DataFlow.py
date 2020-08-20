#  Copyright (c) 2020. Author: Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/data_util.py


import numpy as np
from tensorpack import dataflow


def resample_pcd(pcd, n):
    """drop or duplicate points so that input of each object has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


class PreprocessData(dataflow.ProxyDataFlow):

    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        """get the number of batches"""
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)  # int(False) == 0

    def __iter__(self):
        """generating data in batches"""
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]  # reset holder as empty list => holder = []
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        """
        Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        """
        ids = np.stack([x[0] for x in data_holder])
        inputs = [resample_pcd(x[1], self.input_size) if x[1].shape[0] > self.input_size else x[1]
                  for x in data_holder]
        inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(np.float32)
        npts = np.stack([x[1].shape[0] if x[1].shape[0] < self.input_size else self.input_size
                         for x in data_holder]).astype(np.int32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        return ids, inputs, npts, gts


def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
    """ Load LMDB Files, then Generate Training Batches"""
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)  # buffer_size
        df = dataflow.PrefetchData(df, num_prefetch=500, num_proc=1)  # multiprocess the data
    df = BatchData(df, batch_size, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, num_proc=8)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size
