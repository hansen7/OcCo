#!/usr/bin/env bash

cd ../

# training pointnet on ShapeNetPart, from scratch
python train_partseg.py \
	--gpu 0 \
	--normal \
	--bn_decay \
	--xavier_init \
	--model pointnet_partseg \
    --log_dir pointnet_scratch ;


# fine tuning pcn on ShapeNetPart, using jigsaw pre-trained checkpoints
python train_partseg.py \
	--gpu 0 \
	--normal \
	--bn_decay \
	--xavier_init \
	--model pcn_partseg \
	--log_dir pcn_jigsaw \
	--restore \
	--restore_path log/jigsaw/modelnet_pcn_vanilla/checkpoints/best_model.pth ;


# fine tuning dgcnn on ShapeNetPart, using occo pre-trained checkpoints
python train_partseg.py \
	--gpu 0, 1 \
	--normal \
	--use_sgd \
	--xavier_init \
	--scheduler cos \
	--model dgcnn_partseg \
	--log_dir dgcnn_occo \
	--restore \
	--restore_path log/completion/modelnet_dgcnn_vanilla/checkpoints/best_model.pth ;


# test fine tuned pointnet on ShapeNetPart, using multiple votes
python train_partseg.py \
	--gpu 0 \
	--epoch 1 \
	--mode test \
	--num_votes 3 \
	--model pointnet_partseg \
	--log_dir pointnet_scratch \
	--restore \
	--restore_path log/partseg/pointnet_occo/checkpoints/best_model.pth ;
