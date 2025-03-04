# tiny2
Implementation of core ideas from the paper "Landolsi, Kahl 2024, Tiny models 
from tiny data: Textual and null-text inversion for few-shot distillation".

In the current version, only the core image generation method (TINT) is 
included. Other parts may be added later.

Note that significant source code clean-up was made after the paper was
completed. Results generated from this source code should be similar but may
not be identical to the results presented in the paper.

## Prepare for use
Make sure that all third-party dependencies are installed. Preferrably, create
a new virtual environment and install required packages. This can be done e.g.
using
  ```
  python3 -m venv tiny2venv
  source tiny2venv/bin/activate
  pip install -r requirements.txt
  ```
It might of course also be possible to use an existing virtual environment. All
direct dependencies are rather common packages. However, if using the super-resolution
features, the HAT module needs to be installed, which references some less common
dependencies that may clutter your virtual environment.

If there are dependency verison issues, the file ```requirements_frozen.txt``` is
also available, showing the exact versions used in a test run. This file should 
not be used directly, since it also includes the HAT package which was installed
from a local source, and since you may want to use newer packages when available.
However, it could be used as a reference for troubleshooting.

The code was tested using Python 3.10.12, but should work fine across a range
of Python versions.

In order to use the super-resolution features, the HAT method from 
https://github.com/XPixelGroup/HAT must be setup according to the following steps:

 - Clone their repo from https://github.com/XPixelGroup/HAT

 - In the virtual environment associated with this project, install the HAT
   dependencies and the HAT module itself according to instructions in their
   README.md (using pip install and setup.py).
   
 - Download their model ```Real_HAT_GAN_sharper.pth``` and put it in the
   ```pretrained_models``` folder.

 - Manually fix a compatibility issue in BasicSR (HAT requirement) documented
   at https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985.
   Instructions reproduced here for convenience: 
   Open ```tiny2venv/lib/python3.10/site-packages/basicsr/data/degradations.py```
   On line 8, change
   ```from torchvision.transforms.functional_tensor import rgb_to_grayscale```   
   to ```from torchvision.transforms.functional import rgb_to_grayscale```

The code was tried with commit ```1b22ba0aff82d9d041f5bfa763f82649e6c23d99``` of the HAT repo.

## How to use
When all dependencies are prepared, it should be possible to make a test run
using 
  ```
  python3 scripts/generate.py --config demo_example/config.yaml
  ```
This should create a set of output images in ```output/demo_example/gendata``` that
should look reasonably similar to the support images in ```demo_examples``` (should
be birds in 84x84 resolution). The support images in ```demo_examples``` are from the
CUB dataset [Wah et al. 2011, The caltech-ucsd birds-200-2011 dataset. Technical
Report CNS-TR-2011-001]

To experiment on your own images, make a new config file and edit the input image
paths. The input image resolution also likely requires some consideration.

## Computational requirements
The code was tested to run fine on an NVidia GeForce 4090 with 24 Gb VRAM. On this
GPU, the textual inversion / null-text inversion steps in the provided example
take around 16 min. After these initial steps, it takes around 1.5s for each new
image.

## Adjusting parameters for input image size
The default stable diffusion model used runs best at 512x512 resolution.
Therefore, input images must be resized to this resolution first. If your input
image is already close to or larger than 512x512, set ```enable_superres``` to
```False```. This will resample your images to 512x512 with bilinear interpolation
instead of running super resolution.

If your inputs are significantly smaller than 512x512, some combination of
bilinear upsampling and super-resolution is needed. Check out the ```superres.py```
code and the HAT documentation for details and adjust the ```superres/pre_resolution```
parameter in the config file. The ```post_resolution``` parameter should always match the
intrinsic resolution of the diffusion model (512x512 if using defaults).

## License and citation
The code is released under the MIT license, see the LICENSE file for details.

If you find the code useful, please consider citing our paper:
```
@article{landolsi2024tiny,
  title={Tiny models from tiny data: Textual and null-text inversion for few-shot distillation},
  author={Landolsi, Erik and Kahl, Fredrik},
  journal={arXiv preprint arXiv:2406.03146},
  year={2024}
}
```
