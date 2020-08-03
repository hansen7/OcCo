#!/usr/bin/env bash

cd ../

# train pointnet-occo model on ModelNet, from scratch
python train_completion.py \
	--gpu 0,1 \
	--dataset modelnet \
	--model pointnet_occo \
	--log_dir modelnet_pointnet_vanilla ;

# train dgcnn-occo model on ShapeNet, from scratch
python train_completion.py \
	--gpu 0,1 \
	--batch_size 16 \
	--dataset shapenet \
	--model dgcnn_occo \
	--log_dir shapenet_dgcnn_vanilla ;
