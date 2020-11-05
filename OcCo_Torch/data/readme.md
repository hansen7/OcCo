## Data Setup

#### OcCo

We construct the training data based on ModelNet in the same format of the [data](https://drive.google.com/drive/folders/1M_lJN14Ac1RtPtEQxNlCV9e8pom3U6Pa) provided in PCN which is based on ShapeNet. **You can find our generated dataset based on ModelNet40 [here](https://drive.google.com/drive/folders/1gXNcARYxAh8I4UskbDprJ5fkbDSKPAsH?usp=sharing)**, this is similar with the resources used in the PCN and its follow-ups (summarised [here](https://github.com/hansen7/OcCo/issues/2)).

If you want to generate your own data, please check our provided instructions from <a href="../../render/readme.md">render/readme.md</a>.



#### Classification

In the classification tasks, we use the following benchmark datasets:

- `ModelNet10`[[link](http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)]

- `ModelNet40`[[link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)]

- `ShapeNet10` and `ScanNet10` are from [PointDAN](https://github.com/canqin001/PointDAN)] 

- `ScanObjectNN` are obtained via enquiry to the author of [[paper](https://arxiv.org/abs/1908.04616)]

- `ShapeNet/ModelNet Occluded`  are generated via `utils/lmdb2hdf5.py` on the OcCo pre-trained data:

	```bash
	python lmdb2hdf5.py \
		--partial \
		--num_scan 10 \
		--fname train \
		--lmdb_path ../data/modelnet40_pcn \
		--hdf5_path ../data/modelnet40/hdf5_partial_1024 ;
	```

For `ModelNet40`, we noticed that this newer [source](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) provided in PointNet++ will result in performance gains, yet we stick to the original data used in the PointNet and DGCNN to make a fair comparison.



#### Semantic Segmentation

We use the provided S3DIS [data](https://github.com/charlesq34/pointnet/blob/master/sem_seg/download_data.sh) from PointNet, which is also used in DGCNN.

Please see [here](https://github.com/charlesq34/pointnet/blob/master/sem_seg/download_data.sh) for the download details, it is worth mentioning that if you download from the original S3DIS and preprocess via <a href="../utils/collect_indoor3d_data.py">utils/collect_indoor3d_data.py </a>and <a href="../utils/gen_indoor3d_h5.py">utils/gen_indoor3d_h5.py</a>, you need to delete an extra symbol in the raw file ([reference](https://github.com/charlesq34/pointnet/issues/45)).



#### Part Segmentation

we use the data provided in the PointNet, which is also used in DGCNN.



#### Jigsaw Puzzles

Please check `utils/3DPC_Data_Gen.py` for details, as well as the original paper.
