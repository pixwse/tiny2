# Original source file from https://github.com/pixwse/tiny2
# Copyright(c) 2025 Erik Landolsi, MIT license, see the LICENSE file.
import gc, logging, random, typing
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from tiny2 import diffusion as t2diff
from tiny2 import utils
from tiny2 import fsc


"""Generators that can be used in the generative distillation setup. Only two
generators included now, but more could be added following the same interface.
"""

# ----------------------------------------------------------------------------
# Parameters

class SdTextinvParams(utils.DictParamsMixin):
    """Parameters for SdTextinvGenerator
    """

    def __init__(self):

        # Overall generative model type
        self.type = 'sd_textinv'

        # Choice of SD version
        self.sd_model_name = 'runwayml/stable-diffusion-v1-5'

        # Number of iterations to run in the SD image generation procedure
        self.gen_iterations: int = 50

        # Set to true to bypass the textinv specialization and just generate
        # images of cats instead. For troubleshooting.
        self.bypass_textinv = False

        # Params for the textinv training
        self.textinv = t2diff.TextinvParams()

        # Classifier-free guidance
        self.cfg = 7.5

        # Set to true to run the generation using fp16 precision
        self.gen_fp16 = False

        # Set to true to enable super resolution as pre-processing. If false,
        # directly upsample to self.superres.post_resolution instead.
        self.enable_superres = False

        # Parameters for the super-resolution preprocessing of inputs, before
        # running the textual inversion.
        self.superres = fsc.superres.SuperresParams()


class TintParams(utils.DictParamsMixin):
    """Parameters for our main approach - TINT (Textual Inversion + Null-text)
    """
    def __init__(self):

        # Overall generative model type
        self.type = 'tint'

        # Parameters of the basic SdTextinv method (diffusion and textual
        # inversion model)
        self.base_sd_textinv = SdTextinvParams()

        # Blend alpha for combining inverted support exampls and new noise
        # contributions. Setting to 1.0 will ignore inverted supports and
        # generate completely new images, while 0.0 will just reproduce the
        # support examples.
        self.alpha = 1.0

        # If true, alpha is drawn at random from [0..1]. This means that more
        # examples will be generated that are closer to the support examples.
        # If enabled, the fixed alpha parameter above will be ignored.
        self.random_alpha = False

        # Set to true to run the image generation in FP16. Note: This has the
        # additional effect that the generator can only be specialized once.
        self.gen_fp16 = False


class IGenerator(ABC):
    """Interface for a generator in the generative distillation framework.
    """

    def __init__(self):
        self.logger = utils.get_null_logger()

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def get_nof_accepted_examples(self) -> int:
        """Return the number of accepted examples produced

        This is to be compatible to DinoGenerator that rejects some examples.
        Here, all examples are returned from __call__ every time, so this is
        just the number of examples returned.
        """

    @abstractmethod
    def specialize(
            self, 
            inputs: torch.Tensor, 
            description: str = '') -> typing.Any:
        """Specialize the generator to output images that look like inputs.
        Also resets the 'number of accepted examples' counter.

        Args: 
          inputs: NCHW tensor of images that the new images should look like
          description: Optional text description of the class type. If
            specified, it could be used with text-to-image methods.
        
        Returns:
          Specialization data that can be saved to a file, for later reference.
        """

    @abstractmethod
    def load_specialization(self, file_name: str):
        """Load an existing specialization from disk.
        """

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        """Generate a new example according to the current specialization.
        """


class SdTextinvGenerator(IGenerator):
    """Generator using stable diffusion (SD) and textual inversion.
    
    A textual inversion model is run, adapting a new token to the input
    images. This token is then used to generate new examples.
    """

    def __init__(self, params: SdTextinvParams):
        super().__init__()
        self.params = params
        self.specialized = False
        self.diffmodel = None
        self.image_counter = 0
        self.template = 'A photo of an #object'
        self.token_config = [t2diff.TokenConfig('#object', 'shared', 'object')]

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def specialize(
            self, 
            inputs: torch.Tensor, 
            description: str = '') -> typing.Any:
        """Specialize the model to generate examples looking like the inputs
        """

        # Make sure that the remaints of any previous model are deallocated
        self.diffmodel = None

        if description != '':
            words = description.split(' ')
            if len(words) > 1:
                raise ValueError('Only a single-word description expected')
            
            self.token_config[0].init = description

        # Resize the inputs to the SD models trained resolution.
        # Could be more options here, like running the SD model on lower
        # resolution, formulating the loss differently etc.
        if self.params.enable_superres:
            self.superres = fsc.superres.get_model(self.params.superres)
            
            with torch.no_grad():
                self.resized_supports = self.superres.run(inputs)

            # We can't fit the SR model and the diff model in memory at the
            # same time.
            del self.superres
            self.superres = None
            gc.collect()
        
        else:
            self.resized_supports = F.interpolate(
                inputs, self.params.superres.post_resolution, mode='bilinear')

        # Create a specialized textinv model using the support images
        # Start with a fresh diff model
        self.diffmodel = t2diff.DiffModel(self.params.sd_model_name)
        self.diffmodel.nof_iterations = self.params.gen_iterations

        # Pick target resolution from one training example
        self.image_size = inputs.shape[2:4]

        # For debugging, provide the option to skip specialization
        if self.params.bypass_textinv:
            self.logger.warning('SdTextinvGenerator: specialization bypassed for debug purposes')
            self.specialized = True
            self.template = 'A photo of something'
            return None

        # Run the actual textinv training
        self.logger.info('SdTextinvGenerator: starting textinv training')
        self.textinv_model = t2diff.train_textinv(
            self.diffmodel, 
            self.resized_supports, 
            self.template, 
            self.token_config,
            self.logger,
            self.params.textinv)

        # Note: We don't train the textinv in half precision, just run it
        if self.params.gen_fp16:
            self.diffmodel.half()

        self.logger.info('SdTextinvGenerator: textinv training done')
        self.specialized = True
        self.image_counter = 0

        # Gather all specialization data for use with load_specialization
        patch = t2diff.DiffModelPatch()
        patch.set_base_model_data(self.diffmodel)
        patch.set_textinv_data(self.token_config, self.textinv_model)

        save_data = {
            'image_size': self.image_size,
            'diff_model_patch': patch
        }
        return save_data

    def load_specialization(self, file_name: str):
        self.logger.info('SdTextinvGenerator: loading specialization')

        # Make sure that the remaints of any previous model are deallocated
        self.diffmodel = None

        # Start with a fresh diff model
        self.diffmodel = t2diff.DiffModel(self.params.sd_model_name)
        self.diffmodel.nof_iterations = self.params.gen_iterations

        # Load the saved data and setup the model accirdingly
        save_data = torch.load(file_name)
        self.image_size = save_data['image_size']
        patch : t2diff.DiffModelPatch = save_data['patch']
        t2diff.setup_saved_textinv(
            self.diffmodel, patch.get_textinv_embeddings())

        self.specialized = True
        self.image_counter = 0
        self.logger.info('SdTextinvGenerator: done loading specialization')

    def __call__(self) -> torch.Tensor:
        """Generate one example looking like the inputs.
        Requires that the model has first been specialized.
        """
        if not self.specialized or not self.diffmodel:
            raise Exception('The model must be specialized first')

        self.diffmodel.cf_guidance = self.params.cfg
        image = self.diffmodel.sample_image(self.template)
        image = F.interpolate(
            image, self.image_size, mode='bilinear', antialias=True)

        self.image_counter += 1
        return image
    
    def get_nof_accepted_examples(self) -> int:
        """Return the number of accepted examples produced.
        See IGenerator for details.
        """
        return self.image_counter


class TintGenerator(IGenerator):
    """Main approach - TINT, combining Textual Inversion and Null-text
    """

    def __init__(self, params: TintParams):
        super().__init__()
        self.params = params
        self.image_counter = 0
        self.specialized = False
        self.debug_fixed_support_no = None # For debugging

        # Modules
        self.base_generator = SdTextinvGenerator(params.base_sd_textinv)
        self.nulltext_module = None

    def set_logger(self, logger: logging.Logger):
        self.base_generator.set_logger(logger)
        self.logger = logger

    def specialize(self, inputs: torch.Tensor, description: str = '') -> typing.Any:
        """Specialize the generator to generate images similar to the inputs.
        """

        if self.params.gen_fp16 and self.specialized:
            raise Exception('Using gen_fp16, the generator can only be specialized once.')

        nof_supports = inputs.shape[0]

        # First, specialize the base generator (run the textual inversion)
        self.logger.info('TintGenerator: starting textinv training')
        textinv_data = self.base_generator.specialize(inputs, description)
        assert self.base_generator.diffmodel # Should be created by specialize
        self.logger.info('TintGenerator: textinv training done')

        # Save (enable for debugging, to avoid having to run the entire thing every time)
        patch = t2diff.DiffModelPatch()
        patch.set_base_model_data(self.base_generator.diffmodel)
        patch.set_textinv_data([], textinv_data)

        # Also run a null-inversion on each input
        self.logger.info('TintGenerator: starting nulltext optimization')
        self.nulltext_module = t2diff.NullInversion(
            self.base_generator.diffmodel)

        self.nullinv_data : list[t2diff.NullInversionData] = []
        for ix in range(nof_supports):
            self.logger.info(f'TintGenerator: nulltext inversion of support ex {ix}')
            nullinv_data = self.nulltext_module.invert(
                self.base_generator.resized_supports[ix:ix+1], 
                self.base_generator.template)
            self.nullinv_data.append(nullinv_data)
            nullinv_data.prompt = self.base_generator.template
            patch.append_nulltext_data(nullinv_data)
        self.logger.info('TintGenerator: nulltext optimization done')

        if self.params.gen_fp16:
            self.base_generator.diffmodel.half()
            for nid in self.nullinv_data:
                nid.half()

        self.image_counter = 0
        self.specialized = True
        save_data = {
            'image_size': self.base_generator.image_size,
            'diff_model_patch': patch
        }
        return save_data

    def load_specialization(self, file_name: str):

        # This will load the file twice, but no big deal
        self.base_generator.load_specialization(file_name)
        patch : t2diff.DiffModelPatch = torch.load(file_name)['diff_model_patch']

        assert self.base_generator.diffmodel != None 
        self.nulltext_module = t2diff.NullInversion(
            self.base_generator.diffmodel)

        self.nullinv_data = patch.get_nulltext_data()

    def __call__(self) -> torch.Tensor:
        """Generate one example looking like the inputs.
        Requires that the model has first been specialized.
        """
        assert self.base_generator.diffmodel
        assert self.nulltext_module

        # Get a random latent out of the nof_support ones
        nof_supports = len(self.nullinv_data)

        if self.debug_fixed_support_no is None:
            # Default behavior
            support_no = random.randint(0, nof_supports-1)
        else:
            # For debug / visualization etc
            support_no = self.debug_fixed_support_no

        latents0 = self.nullinv_data[support_no].latents

        # Generate a random alpha, if enabled (experimental)
        if self.params.random_alpha:
            alpha = random.random()
        else:
            alpha = self.params.alpha

        # Blend latent0 with noise and rescale the norm (see the paper for 
        # details). If alpha = 1.0, we get a completely random example and if
        # alpha = 0.0, we will just reproduce the support examples.
        noise = torch.randn_like(latents0)
        latents1 = (1 - alpha) * latents0 + alpha * noise
        norm_l1 = torch.sqrt(torch.mean(latents1**2))
        norm_l0 = torch.sqrt(torch.mean(latents0**2))
        norm_noise = torch.sqrt(torch.mean(noise**2))
        norm_out = (1 - alpha) * norm_l0 + alpha * norm_noise 
        latents1 = latents1 / norm_l1 * norm_out

        # Generate an image from latent1
        with torch.no_grad():
            temp_nullinv_data = t2diff.NullInversionData()
            temp_nullinv_data.latents = latents1
            temp_nullinv_data.prompt = self.nullinv_data[support_no].prompt
            temp_nullinv_data.unc_embeddings = self.nullinv_data[support_no].unc_embeddings

            image = self.nulltext_module.reconstruct(temp_nullinv_data)

            image_lowres = F.interpolate(
                image, self.base_generator.image_size, 
                mode='bilinear', antialias=True)
          
        self.image_counter += 1
        return image_lowres

    def get_nof_accepted_examples(self) -> int:
        """Return the number of accepted examples produced
        See IGenerator for details.
        """
        return self.image_counter


def get_generator(params) -> IGenerator:
    """Create a generator based on the parameters. 
    
    The generator is not yet specialized; the caller needs to run 
    generator.specialize(inputs, labels) before generating examples.
    """
    if params.type == 'sd_textinv':
        return SdTextinvGenerator(params)
    elif params.type == 'tint':
        return TintGenerator(params)
    else:
        raise ValueError(f'Type unknown: {params.type}')
