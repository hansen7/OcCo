#!/usr/bin/env bash

cd ../

# train pointnet_semseg on 6-fold cv of S3DIS, from scratch
for area in $(seq 1 1 6)
do
python train_semseg.py \
	--gpu 0,1 \
	--model pointnet_semseg \
	--bn_decay \
	--xavier_init \
	--test_area ${area} \
	--scheduler step \
	--log_dir pointnet_area${area}_scratch ;
done

# fine tune pcn_semseg on 6-fold cv of S3DIS, using jigsaw pre-trained weights
for area in $(seq 1 1 6)
do
python train_semseg.py \
	--gpu 0,1 \
	--model pcn_semseg \
	--bn_decay \
	--test_area ${area} \
	--log_dir pcn_area${area}_jigsaw \
	--restore \
	--restore_path log/jigsaw/modelnet_pcn_vanilla/checkpoints/best_model.pth ;
done

# fine tune dgcnn_semseg on 6-fold cv of S3DIS, using occo pre-trained weights
for area in $(seq 1 1 6)
do
python train_semseg.py \
	--gpu 0,1 \
	--test_area ${area} \
	--optimizer sgd \
	--scheduler cos \
	--model dgcnn_semseg \
	--log_dir dgcnn_area${area}_occo \
	--restore \
	--restore_path log/completion/modelnet_dgcnn_vanilla/checkpoints/best_model.pth ;
done

