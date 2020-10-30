#!/usr/bin/env bash

cd ../

# train pointnet_jigsaw on ModelNet40, from scratch
python train_jigsaw.py \
	--gpu 0 \
	--model pointnet_jigsaw \
	--bn_decay \
	--xavier_init \
	--optimiser Adam \
	--scheduler step \
	--log_dir modelnet40_pointnet_scratch ;


# train dgcnn_jigsaw on ModelNet40, from scratch
python train_jigsaw.py \
	--gpu 0 \
	--model dgcnn_jigsaw \
	--optimiser SGD \
	--scheduler cos \
	--log_dir modelnet40_dgcnn_scartch ; 
