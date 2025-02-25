import os
import torch
import torch.nn.functional as F

import hat.archs
import hat.data
import hat.models
import hat.archs.hat_arch

from tiny2 import utils

"""Wrapper around the HAT super-resolution model. Source:
https://github.com/XPixelGroup/HAT
"Chen et al. 2023, Activating More Pixels in Image Super-Resolution Transformer"
The HAT code must be accessible in the Python path.
"""


class SuperresParams(utils.DictParamsMixin):
    """Parameters controlling the super-resolution used in some generators
    """

    def __init__(self):
        self.type = 'HAT'

        # Checkpoint file. Used both to define which file to load and to setup
        # the model to match the file.
        self.checkpoint = 'pretrained_models/Real_HAT_GAN_sharper.pth'

        # Rescale the input to this resolution before running the superres step
        self.pre_resolution = torch.Size((0, 0))

        # Rescale the input to this resolution after running the superres step.
        # This will be the final resolution of the method.
        self.post_resolution = torch.Size((0, 0))


class HatModel:
    """Convenience wrapper for a HAT super-resolution model.

    This wrapper makes it fit our framework better, includes some pre- and
    post rescaling, etc. The core model is from
    https://github.com/XPixelGroup/HAT
    and described in the paper [Chen et al., Activating More Pixels in Image
    Super-Resolution Transformer, CVPR 2023]
    """

    def __init__(
            self, 
            params: SuperresParams,
            device = utils.default_device()):

        assert params.type == 'HAT'
        self.params = params

        # Note: This is consistent with the default checkpoint
        # (Real_HAT_GAN_sharper.pth). May need to be modified if using a
        # different checkpoint
        self.model = hat.archs.hat_arch.HAT(
            img_size=64,
            in_chans=3,
            embed_dim=180,
            depths=(6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6),
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=2.,
            upscale=4,
            img_range=1.,
            upsampler='pixelshuffle',
            resi_connection='1conv')

        full_path = os.path.join(params.path, params.checkpoint)
        state_dict = torch.load(full_path)
        self.device = device
        self.model.load_state_dict(state_dict['params_ema'])
        self.model.to(device)

    def run(self, images: torch.Tensor) -> torch.Tensor:
        """Runs the image upscaling.

        Args:
          images: NCHW tensor containing one or more images, pixel values 0..1
        
        Returns:
          upscaled image(s) (NCHW tensor)
        """
        assert len(images.shape) == 4
        assert torch.min(images) >= 0.0
        assert torch.max(images) <= 1.0

        images = F.interpolate(
            images, self.params.pre_resolution, mode='bilinear')
        
        upscaled_images = utils.run_chunkwise(self.model, images, 1)
        upscaled_images = torch.clamp(upscaled_images, 0, 1)

        upscaled_images = F.interpolate(
            upscaled_images, self.params.post_resolution, mode='bilinear')

        return upscaled_images


def get_model(params):
    """Get a super-resolution model as defined by the parameters.
    """
    if params.type == 'HAT':
        return HatModel(params)
    else:
        raise Exception('Type unknown')


# ----------------------------------------------------------------------------
# Demo code

def demo_superres():

    image_path = 'YOUR 96x96 IMAGE HERE'
    image = utils.load_tensor_image(image_path)

    params = SuperresParams()
    params.pre_resolution = torch.Size((96, 96))
    params.post_resolution = torch.Size((512, 512))
    model = get_model(params)

    upscaled_image = model.run(image)
    utils.save_tensor_image('output/superres_demo_output.png')
