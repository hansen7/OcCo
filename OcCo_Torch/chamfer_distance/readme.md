# Chamfer Distance for PyTorch

This is an implementation of the Chamfer Distance as a module for PyTorch. It is written as a custom C++/CUDA extension. It is developed by [Chris](https://github.com/chrdiller/pyTorchChamferDistance) at TUM.

As it is using PyTorch's [JIT compilation](https://pytorch.org/tutorials/advanced/cpp_extension.html), there are no additional prerequisite steps (e.g., `build` or `setup`) that have to be taken. Simply import the module as shown below, CUDA and C++ code will be compiled on the first run, which additionally takes a few seconds.

### Usage
```python
import torch
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

# both points clouds have shapes of (batch_size, n_points, 3), wherer n_points can be different

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2)))/2  
```

### Integration
This code has been integrated into the [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) library for 3D Deep Learning by NVIDIAGameWorks. You probably want to take a look at it if you are working on some 3D ([pytorch3d](https://github.com/facebookresearch/pytorch3d) is also recommended)

### Earth Mover Distance
For the implementation of earth mover distance, we recommend [Kaichun's](https://github.com/daerduoCarey/PyTorchEMD) :)
