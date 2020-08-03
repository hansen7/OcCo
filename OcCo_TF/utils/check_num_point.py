#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

# Author: Hanchen Wang, hw501@cam.ac.uk

import numpy as np
import os, json, argparse
from data_util import lmdb_dataflow
from io_util import read_pcd
from tqdm import tqdm

MODELNET40_PATH = r"../render/dump_modelnet_normalised_"
SCANNET10_PATH = r"../data/ScanNet10"
SHAPENET8_PATH = r"../data/shapenet"


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='modelnet40', help="modelnet40, shapenet8 or scannet10")
	
	args = parser.parse_args()
	os.system("mkdir -p ./dump_sum_points")
	
	if args.dataset == 'modelnet40':
		shape_names = open(r'../render/shape_names.txt').read().splitlines()
		file_ = open(r'../render/ModelNet_flist_normalised.txt').read().splitlines()
		
		print("=== ModelNet40 ===\n")
		for t in ['train', 'test']:
			# for res in ['fine', 'middle', 'coarse', 'supercoarse']:
			for res in ['supercoarse']:
				sum_dict = {}
				for shape in shape_names:
					sum_dict[shape] = np.zeros(3,dtype=np.int32)  # num of objects, num of points, average
				
				model_list = [_file for _file in file_ if t in _file]
				for model_id in tqdm(model_list):
					model_name = model_id.split('/')[0]
					for i in range(10):
						partial_pc = read_pcd(os.path.join(MODELNET40_PATH + res, 'pcd', model_id + '_%d.pcd' % i))
						sum_dict[model_name][1] += len(partial_pc)
						sum_dict[model_name][0] += 1
					
					sum_dict[model_name][2] = sum_dict[model_name][1]/sum_dict[model_name][0]
				
				f = open("./dump_sum_points/modelnet40_%s_%s.txt" % (t, res), "w+")
				for key in sum_dict.keys():
					f.writelines([key, str(sum_dict[key]), '\n'])
				f.close()
				print("=== ModelNet40 %s %s Done ===\n" % (t, res))
	
	elif args.dataset == 'shapenet8':
		print("\n\n=== ShapeNet8 ===\n")
		for t in ['train', 'valid']:
			sum_dict = json.loads(open(os.path.join(SHAPENET8_PATH, 'keys.json')).read())
			for key in sum_dict.keys():
				sum_dict[key] = np.zeros(3)  # num of objects, num of points, average
			
			# the data stored in the lmdb files is with varying number of points
			df, num = lmdb_dataflow(lmdb_path=os.path.join(SHAPENET8_PATH, '%s.lmdb' % t),
			                        batch_size=1, input_size=1000000, output_size=1, is_training=False)
			
			data_gen = df.get_data()
			for _ in tqdm(range(num)):
				ids, _, npts, _ = next(data_gen)
				model_name = ids[0][:8]
				sum_dict[model_name][1] += npts[0]
				sum_dict[model_name][0] += 1
				
				sum_dict[model_name][2] = sum_dict[model_name][1] / sum_dict[model_name][0]
				
			f = open("./dump_sum_points/shapenet8_%s.json" % t, "w+")
			for key in sum_dict.keys():
				f.writelines([key, str(sum_dict[key]), '\n'])
			# f.write(json.dumps(sum_dict))
			f.close()
			print("=== ShapeNet8 %s Done ===\n" % t)
	
	elif args.dataset == 'scannet10':
		print("\n\n=== ScanNet10 is not ready yet ===\n")
		
	else:
		raise ValueError('Assigned dataset do not exist.')
