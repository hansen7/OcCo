#!/usr/bin/env bash

cd ../

# training pointnet on ModelNet40, from scratch
python train_cls.py \
	--gpu 0 \
	--model pointnet_cls \
	--dataset modelnet40 \
	--log_dir modelnet40_pointnet_scratch ;


# fine tuning pcn on ScanNet10, using jigsaw pre-trained checkpoints
python train_cls.py \
	--gpu 0 \
	--model pcn_cls \
	--dataset scannet10 \
	--log_dir scannet10_pcn_jigsaw \
	--restore \
	--restore_path log/completion/modelnet_pcn_vanilla/checkpoints/best_model.pth ;


# fine tuning dgcnn on ScanObjectNN(OBJ_BG), using jigsaw pre-trained checkpoints
python train_cls.py \
	--gpu 0,1 \
	--epoch 250 \
	--use_sgd \
	--scheduler cos \
	--model dgcnn_cls \
	--dataset scanobjectnn \
	--bn \
	--log_dir scanobjectnn_dgcnn_occo \
	--restore \
	--restore_path log/completion/modelnet_dgcnn_vanilla/checkpoints/best_model.pth ;


# test pointnet on ModelNet40 from pre-trained checkpoints
python train_cls.py \
	--gpu 1 \
	--epoch 1 \
	--mode test \
	--model pointnet_cls \
	--dataset modelnet40 \
	--log_dir modelnet40_pointnet_scratch \
	--restore \
	--restore_path log/cls/modelnet40_pointnet_scratch/checkpoints/best_model.pth ;
