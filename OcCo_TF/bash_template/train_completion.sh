#!/bin/bash

cd ..

python train_completion.py \
    --gpu 7 \
    --lr_decay \
    --batch_size 16 \
    --log_dir ./log/completion/pcn_cd_vanilla ;

#python train_completion.py \
#	--gpu 4 \
#	--lmdb_train ./data/modelnet40_pcn/ModelNet40_train_1024_middle.lmdb \
#	--lmdb_valid ./data/modelnet40_pcn/ModelNet40_test_1024_middle.lmdb \
#	--log_dir ./log/log_completion/pointnet_cd_new_modelnet_lr1e-4_b8 \
#	--base_lr=0.0001 \
#	--epoch 100 \
#	--lr_decay \
#	--model_type pointnet_cd \
#	--batch_size 8;
