#!/usr/bin/env bash

cd ../

# train pointnet_semseg on Urban, from scratch
python train_semseg_city.py \
	--gpu 6,7 \
	--model pointnet_semseg \
	--bn_decay \
	--xavier_init \
	--scheduler step \
	--log_dir pointnet_city_scratch ;


# train pointnet_semseg on Urban, from scratch
python train_semseg_city.py \
	--gpu 6,7 \
	--model pointnet_semseg \
	--bn_decay \
	--xavier_init \
	--scheduler step \
    --log_dir pointnet_city_pretrained \
	--restore \
	--restore_path log/jigsaw/modelnet_pointnet_vanilla/checkpoints/best_model.pth ;
