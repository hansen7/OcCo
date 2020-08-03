## Data Setup

#### OcCo

We construct the training data based on ModelNet in the same format of the data provided in PCN which is based on ShapeNet. To generate your own data, we provided a similar way as PCN:

```
utils/
```



#### Classification

In the classification tasks, we use the following benchmark dataset:

- `ModelNet10`([link](http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip))

- `ModelNet40`([link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip))

- `ShapeNet10` and `ScanNet10` are from [PointDAN](https://github.com/canqin001/PointDAN) 

- `ShapeNet Occuluded` and `ModelNet Occuluded` are generated via `utils/lmdb2hdf5.py` on the OcCo data

- `ScanObjectNN` are obtained via equiries to the author of [paper](https://arxiv.org/abs/1908.04616)

For ModelNet40, we noticed that this newer provided [source](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) in PointNet++ will result in performance gain in the model, we stick to the original data used in the PointNet and DGCNN to make a fair comparison.



#### Semantic Segmentation

- We use the provided S3DIS data from PointNet, which is also used in DGCNN.

	see [here](https://github.com/charlesq34/pointnet/blob/master/sem_seg/download_data.sh) for the downloading details, it is worth noting that if you download from the original S3DIS and preprocess via `utils/collect_indoor3d_data.py` and `utils/gen_indoor3d_h5.py`, you need to delete an extra file in the raw data ([ref](https://github.com/charlesq34/pointnet/issues/45)).

	

#### Jigsaw Puzzles

```

```



