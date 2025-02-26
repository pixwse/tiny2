# tiny2
Implementation of core ideas from the paper "Landolsi, Kahl 2024, Tiny models 
from tiny data: Textual and null-text inversion for few-shot distillation".

In the current version, only the core image generation method (TINT) is 
included. Other parts may be added later.

Note that significant source code clean-up was made after the paper was
completed. Results generated from this source code should be similar but may
not be identical to the results presented in the paper.

## How to use
Make sure that all third-party dependencies are installed. Preferrably, create
a new virtual environment and install required packages. This can be done e.g.
using
  ```
  python3 -m venv tiny2venv
  source tiny2venv/bin/activate
  pip install -r requirements.txt
  ```
The code was tested using Python 3.10.12, but should work fine across a large
range of Python versions.

In order to use the super-resolution features, the HAT repo available at
https://github.com/XPixelGroup/HAT must be accessible from the python path
(not included in requirements.txt). Furthermore, their model 
Real_HAT_GAN_sharper.pth must be downloaded and put in the pretrained_models
folder. See the HAT repo for instructions.

When all dependencies are prepared, it should be possible to make a test run
using 
  ```
  python3 scripts/generate.py --config demo_example/config.yaml
  ```
This should create a set of output images in output/demo_example/gendata that
should look reasonably similar to the support images in demo_examples (should
be brown-ish cups in 128x128 resolution). The support images in demo_examples
are from the CUB dataset [Wah et al. 2011, The caltech-ucsd birds-200-2011 dataset.
Technical Report CNS-TR-2011-001]

To experiment on your own images, make a new config file and edit the input image
paths. The input image resolution also likely requires some consideration.

## Computational requirements
The code was tested to run fine on an NVidia GeForce 4090 with 24 Gb VRAM. On this
GPU, the textual inversion / null-text inversion steps in the provided example
takes around 16 min. After these initial steps, it take around 1.5s for each new
image.

## Adjusting parameters for input image size
The default stable diffusion model used runs best at 512x512 resolution.
Therefore, input images must be resized to this resolution first. If your input
image is already close to or larger than 512x512, set enable_superres to False.
This will resample your images to 512x512 with bilinear interpolation.

If your inputs are significantly smaller than 512x512, some combination of
bilinear upsampling and super-resolution is needed. Check out the superres.py
code and the HAT documentation for details and adjust the superres/pre_resolution
parameter in the config file. The post_resolution parameter should always match the
intrinsic resolution of the diffusion model (512x512 if using defaults).

## License and citation
The code is released under the MIT license, see the LICNSE file for details.

If you find the code useful, please consider citing our paper:
```
@article{landolsi2024tiny,
  title={Tiny models from tiny data: Textual and null-text inversion for few-shot distillation},
  author={Landolsi, Erik and Kahl, Fredrik},
  journal={arXiv preprint arXiv:2406.03146},
  year={2024}
}
```
