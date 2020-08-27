#!/usr/bin/env bash

cd ../

# grid search the best parameters of a svm with rbf kernel on ModelNet40 encoded by OcCo PCN
python train_svm.py \
	--gpu 0 \
	--grid_search \
	--model pcn_util \
	--dataset modelnet40 \
	--restore_path log/completion/modelnet_pcn_vanilla/checkpoints/best_model.pth ;


# ... on ScanObjectNN(OBJ_BG) encoded by OcCo DGCNN
python train_svm.py \
	--gpu 0 \
	--grid_search \
	--batch_size 8 \
	--model dgcnn_util \
	--dataset scanobjectnn \
	--bn \
	--restore_path log/completion/modelnet_dgcnn_vanilla/checkpoints/best_model.pth ;
