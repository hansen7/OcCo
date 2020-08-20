This directory contains code that generates partial point clouds objects. 

To start with:

1. Download an Install [Blender](https://blender.org/download/)

2. Create a list of normalized 3D objects to be rendered, which should be in `.obj` format, we provide `ModelNet_Flist.txt`. as a template. We also provide `PC_Normalisation.py` for normalization.

3. To generate the rendered depth image from 3d objects (you might need to install a few more supportive packages, i.e. `Imath, OpenEXR`, due to the differences in the development environments)

	```bash
	# blender -b -P Depth_Renderer.py [data directory] [file list] [output directory] [num scans per model]
	
	blender -b -P render_depth.py ../data/modelnet40 ModelNet_Flist.txt ./dump 10
	```

	The generated intermediate files are in OpenEXR format (`*.exr`). You can also modify the intrinsics of the camera model in `Depth_Renderer.py`, which will be automatically saved in the `intrinsics.txt`.

4. To re-project the partial occluded point cloud from the depth image:

	``` bash
	python EXR_Process.py \
		--list_file ModelNet_Flist.txt \
	    --intrinsics intrinsics.txt \
	    --output_dir ./dump \
	    --num_scans 10 ;
	```

	This will convert the `*.exr` files into depth images (`*.png`) then point clouds (`*.pcd`)

5. Now use `OcCo_Torch/utils/LMDB_Writer.py` to convert all the  `pcd` files into `lmdb` dataloader:

	```bash
	python LMDB_Writer.py \
		--list_path ../render/ModelNet_Flist.txt \
	    --complete_dir ../data/modelnet40 \
	    --partial_dir ../render/dump/pcd \
	    --num_scans 10 \
	    --output_file ../data/MyTrain.lmdb ;
	```

6. Now you can pre-train the models via OcCo on your own constructed data, enjoy :)
