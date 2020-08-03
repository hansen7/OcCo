#!/bin/bash

docker run -it \
	--rm \
	--shm-size=1g \
	--runtime=nvidia \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v "$(dirname $PWD):/workspace/OcCo_Torch" \
	-v "/scratch/hw501/data_source/:/scratch/hw501/data_source/" \
	-v "/scratches/mario/hw501/data_source:/scratches/mario/hw501/data_source/" \
	-v "/scratches/weatherwax_2/hwang/OcCo/data/:/scratches/weatherwax_2/hwang/OcCo/data/" \
	occo_torch bash

# -v + any external directories if you are using them
