#!/usr/bin/env bash

cd ..
# load PointNet params from OcCo Pre-Trained model

python train_cls.py --gpu 0 \
	--log_dir './log/pretrained/pcn_cls_modelnet_new_cd_15cls_init_ep20_b64' \
	--model pcn_cls \
	--dataset scanobjectnn \
	--dataset_file _1024 \
	--just_save ;
		  
python utils/transfer_pretrained_w.py \
	--source_path='./log/pretrained/pcn_cls_modelnet_new_cd_10cls_init_ep20_b64/model.ckpt' \
	--target_path='./log/pretrained/pcn_cls_modelnet_new_cd_15cls_init_ep20_b64/model.ckpt';

python train_cls.py --gpu 0 \
	--log_dir './log/pretrained/pcn_cls_modelnet_new_cd_15cls_init_ep20_b64' \
	--model pcn_cls \
	--dataset scanobjectnn \
	--dataset_file _1024 \
	--restore \
    --restore_path './log/pretrained/pcn_cls_modelnet_new_cd_15cls_init_ep20_b64';
