import torch
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    DDPMScheduler, 
    DDIMScheduler)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer)

from tiny2 import utils

# ----------------------------------------------------------------------------
# DiffModel parameters

class DiffModelParams(utils.DictParamsMixin):

    def __init__(self):
        # Stable diffusion model (huggingface identifier)
        self.model = 'runwayml/stable-diffusion-v1-5'

        # Scheduler type (DDIM, DDPM or UniPCMultistep)
        self.scheduler = 'DDIM'

        # Initial random seed
        self.seed = 30

# ----------------------------------------------------------------------------
# DiffModel implementation

class DiffModel:
    """Full stable diffusion model with helper functions.
    
    This class contains all parts of an SD model, including its most common
    parameters. Also contains basic helper functions for doing the most common
    operations on the diffusion model, like creating initial noise, producing
    conditional embeddings from a text prompt, etc.
    
    The class is designed such that common operations in running a diffusion
    model are encapsulated in small pieces, such that it's easy to experiment
    with different varieties without too much code duplication. 

    It is recommended to create an instance of this class using get_diff_model
    below.
    """

    # ------------------------------------------------------------------------
    # Setup

    def __init__(
            self, 
            model_name: str = "runwayml/stable-diffusion-v1-5",
            scheduler_name: str = "DDIM",
            seed: int = 30,
            fp16: bool = False,
            device = utils.default_device(),
            timestep_spacing = 'trailing') -> None:
        """Create a diffusion model.

        Args:
          model_name: Model name on the hugging face hub
          scheduler_name: Which scheduler to use (can often be switched without
            retraining)
          seed: Random seed
          fp16: Set to true to create the model with half precision
          device: pytorch device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.scheduler_name = scheduler_name

        self.dtype = torch.float32 if not fp16 else torch.float16

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_name, subfolder='vae', torch_dtype=self.dtype)
        
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder='tokenizer')
        
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            model_name, subfolder='text_encoder', torch_dtype=self.dtype)
        
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_name, subfolder='unet', torch_dtype=self.dtype)
        
        if scheduler_name == 'DDIM':
            self.scheduler = DDIMScheduler.from_pretrained(
                model_name, subfolder='scheduler', timestep_spacing=timestep_spacing)

        elif scheduler_name == 'UniPCMultistep':
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                model_name, subfolder='scheduler', timestep_spacing=timestep_spacing)

        elif scheduler_name == 'DDPM':
            self.scheduler = DDPMScheduler.from_pretrained(
                model_name, subfolder='scheduler', timestep_spacing=timestep_spacing)

        else:
            raise ValueError(f'scheduler name invalid: {scheduler_name}')

        self.device = device
        self.vae.to(device)
        self.unet.to(device)
        self.text_encoder.to(device)
        self.seed = seed
        self.reset_seed()

        # Default settings, can be changed from the outside.
        self.nof_iterations = 25
        self.cf_guidance = 7.5 # classifier-free guidance
        self.width = 512
        self.height = 512

        # VAE properties (could potentially vary depending on settings)
        self.vae_block_size = 8
    
    def half(self):
        """Convert the model to half precision. 
        
        When adjusting a model with textinv or nulltextinv, it can be useful to
        first store it in full precision, then change precision when the
        training is done.
        """
        self.vae.half()
        self.text_encoder.half()
        self.unet.half()
        self.dtype = torch.float16

    def reset_seed(self):
        """Reset the random seed.
        
        This is useful for e.g. running repeated experiments with varying
        parameters, to show only the impact of those parameters.
        """
        self.generator = torch.manual_seed(self.seed)

    def set_timesteps(self):
        """Set the scheduler timesteps. Must be run after changing nof_iterations.
        """
        self.scheduler.set_timesteps(self.nof_iterations)

    def get_timestep(self, ix: int):
        """Get timestep from index
        
        Args:
          ix: Index into the scheduler timesteps list. Index 0 is the highest 
            timestep. If ix is too large, returns 0.
        """
        if ix < len(self.scheduler.timesteps):
            return self.scheduler.timesteps[ix]
        else:
            return 0

    # ------------------------------------------------------------------------
    # Encode/decode (input image to VAE latents)

    def encode_normalized(self, image: torch.Tensor) -> torch.Tensor:
        """Encodes the image(s) into latents (at t=0)

        Same functionality as encode below, but with alerady normalized 
        (zero-mean) image input.

        Args:
          image: NCHW tensor with normalized (zero-mean) values 
        
        Returns:
          NC'H'W' tensor with latents
        """
        assert len(image.shape)==4
        latents = self.vae.encode(image)['latent_dist'].mean # type:ignore
        return self.vae.config['scaling_factor'] * latents

    def decode_normalized(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latent (at t=0) into a normalized image
        
        Args:
          latents: NC'H'W' tensor with latents
        
        Returns:
          NCHW image with normalized (zero-mean) values
        """
        assert len(latents.shape)==4
        latents = 1 / self.vae.config['scaling_factor'] * latents
        return self.vae.decode(latents).sample # type:ignore

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an image to latents (at t=0)
        
        Scales the image to roughly zero-mean, unit variance, runs the vae 
        encoder and scales the latents to the range expected by the diffusion
        model. This is the exact inverse of decode above.

        Args:
          image: Input image, NCHW tensor with values in [0, 1]
        
        Returns:
          NC'H'W' tensor with latents
        """
        assert len(image.shape)==4
        image = 2*(image - 0.5)
        return self.encode_normalized(image)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Get an image from final latents.
        
        Includes scaling the latents from the scale used within SD to what 
        the VAE wants and scaling the image to be floats in [0..1]
        """
        assert len(latents.shape)==4
        image = self.decode_normalized(latents)    

        # Convert from normalized to [0..1]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # ------------------------------------------------------------------------
    # Text embedding / conditioning

    def get_token_ids(self, prompts: str | list[str]) -> torch.Tensor:
        """Return token IDs for the prompt 
        
        Using the default settings (pad/truncate to max length)

        Args:
          prompts: One or several text prompts

        Returns:
          NxL integer tensor with token IDs, where N is the number of
            input prompts and L is the max sequence length. The tensor
            is put on self.device.
        """
        tokens = self.tokenizer(
            prompts, 
            padding="max_length",
            max_length=self.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt")
        return tokens.input_ids.to(self.device)
    
    def get_id_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from token IDs

        Args:
          token_ids: NxL integer tensor with token IDs, where N is the
            number of input prompts and L is the max sequence length.

        Returns:
          NxLxF tensor, where F is the embedding dimensionality 
        """
        # Note: The embeddings are the sum of individual input token
        # embeddings and position embeddings
        return self.text_encoder(token_ids).last_hidden_state

    @torch.no_grad()
    def get_text_embeddings(
            self,
            prompt: str,
            include_null: bool = False, 
            repeat: int = 1) -> torch.Tensor:
        """Helper function for computing the text embedding of a prompt.

        Uses no_grad, since we can't get gradients with respect to the input 
        text anyway.
        
        Args:
          prompt: Text prompt
          include_null: If True, also includes text embeddings of the null 
            prompt ("") for use with classifier-free guidance. In this case,
            the null embedding is returned as the first embedding, and the 
            prompt embedding as the second.
          repeat: Number of times the embeddings should be repeated (for use 
            with batch size > 1)

        Returns:
          NxLxF tensor with text embeddings of the prompt (and optionally
            the null embedding), where N is the number of prompts (1 or 2
            times 'repeat'), L is the max sequence length and F is the
            embedding dimensionality.
        """
        token_ids = self.get_token_ids(prompt)
        text_embeddings = self.get_id_embeddings(token_ids)
        text_embeddings = text_embeddings.expand(repeat, -1, -1)

        if include_null:
            null_token_ids = self.get_token_ids("")
            uncond_embeddings = self.get_id_embeddings(null_token_ids)
            uncond_embeddings = uncond_embeddings.expand(repeat, -1, -1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # ------------------------------------------------------------------------
    # Noise/latent prediction substeps

    def change_timestep(
            self, 
            noise_pred: torch.Tensor, 
            timestep_from: int,
            timestep_to: int,
            z_from: torch.Tensor) -> torch.Tensor:
        """Given the latent from a certain timestep, predict the latent at a
        different timestep (previous or next) by adding/subtracting noise.

        Args:
          noise_pred: Estimated noise at timestep 'timestep_from' (unet output)
          timestep_from: The timestep to move from
          timestep_to: The timestep for which we want to estimate the new data
          z_from: The latent at timstep 'timestep_from'
        
        Returns:
          The estimated latent at timestep 'timestep_to'
        """

        # Get alpha_cumprod (alpha-bar in most diffusion papers) for the 
        # 'from' and 'to' timesteps
        abar_from = self.scheduler.alphas_cumprod[timestep_from]
        abar_to = self.scheduler.alphas_cumprod[timestep_to]

        # Get 1 - alpha-bar (called 'beta-bar' in most papers) 
        bbar_from = 1 - abar_from
        bbar_to = 1 - abar_to

        # Change timestep. This is just applying the relation
        #   z_t = sqrt(abar)*z0 + sqrt(bbar)*epsilon_t
        # twice. First from z_from to z0, then from z0 to z_to
        z0 = (z_from - bbar_from**0.5 * noise_pred) / abar_from**0.5
        z_to = abar_to**0.5 * z0 + bbar_to**0.5 * noise_pred

        # Equivalent formulations, for reference:
        # --
        # kz = (abar_to / abar_from)**0.5
        # ke = (1-abar_to)**0.5 - abar_to**0.5 * (1-abar_from)**0.5 / abar_from**0.5
        # --
        # kz = (abar_to / abar_from)**0.5
        # ke = abar_to**0.5 * ( 
        #     (1-abar_to)**0.5 / abar_to**0.5 - 
        #     (1-abar_from)**0.5 / abar_from**0.5
        #     )
        # print(f'kz={kz}, ke={ke}')
        # z_to = kz * z_from + ke * noise_pred


        return z_to

    def prepare_latents(
            self, 
            latents: torch.Tensor,
            expand_count: int, 
            timestep: int):
        """Expand and scale the latents, as is typically done within a denoising
           loop.
           
           The latents are expanded such that we can process run the model for
           several conditionings in parallel (typically one text prompt and
           one unconditional embedding).
           
           Args:
             latents: Original, unscaled input latents
             expand_count: Number of times to expand the latents. Should
               correspond to the number of prompts to run in parallel later.
        """
        prep_latents = torch.cat([latents] * expand_count)
        prep_latents = self.scheduler.scale_model_input(
            prep_latents, timestep=timestep)
        return prep_latents

    def predict_noise(
            self, 
            latents: torch.Tensor,
            timesteps: int | list[int],
            conditioning: torch.Tensor) -> torch.Tensor:
        """Run the u-net to predict noise
        Slightly more convenient than calling the u-net directly.
        """
        return self.unet(
            latents,
            timesteps, 
            encoder_hidden_states=conditioning).sample

    def apply_cfg(
            self,
            np_uncond: torch.Tensor,
            np_cond: torch.Tensor) -> torch.Tensor:
        """Apply classifier-free guidance.
        
        Args:
          np_uncond: Unconditional embedding (null-text)
          np_cond: Conditional embedding (prompt)
        """
        return np_uncond + self.cf_guidance * (np_cond - np_uncond)

    def denoise(
            self, 
            latents: torch.Tensor,
            conditioning: torch.Tensor,
            verbose = False) -> torch.Tensor:
        """Run the actual denoising over all timesteps
        
        Args:
          latents: Latents at time t=T
          conditioning: 2 stacked conditioning tensors (null-conditioning and
            actual conditioning, typically as returned from get_text_embeddings)
          verbose: Set to true to print debug info to the console
        
        Returns:
          Latents at time t=0
        """

        self.set_timesteps()

        # Run the denoising process
        for ix, t in enumerate(self.scheduler.timesteps):
            if verbose:
                print(f'Diffusion step {ix}')

            # Scale and expand the latents
            prep_latents = self.prepare_latents(latents, 2, t)

            # Run the network
            noise_pred = self.predict_noise(prep_latents, t, conditioning)

            # Perform classifier-free guidance
            np_uncond, np_text = noise_pred.chunk(2)
            noise_pred = self.apply_cfg(np_uncond, np_text)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def partial_denoise(
            self, 
            latents: torch.Tensor,
            conditioning: torch.Tensor,
            timesteps) -> torch.Tensor:
        """Run the actual denoising over all timesteps
        
        Args:
          latents: Latents at the highest time step in 'timesteps'
          conditioning: 2 stacked conditioning tensors (null-conditioning and
            actual conditioning, typically as returned from get_text_embeddings)
          verbose: Set to true to print debug info to the console
        
        Returns:
          Latents at the last time in 'timesteps'
        """

        self.set_timesteps()

        # Run the denoising process
        for ix, t in enumerate(timesteps):

            # Scale and expand the latents
            prep_latents = self.prepare_latents(latents, 2, t)

            # Run the network
            noise_pred = self.predict_noise(prep_latents, t, conditioning)

            # Perform classifier-free guidance
            np_uncond, np_text = noise_pred.chunk(2)
            noise_pred = self.apply_cfg(np_uncond, np_text)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def denoise_unguided(
            self, 
            latents: torch.Tensor,
            conditioning: torch.Tensor,
            verbose = False) -> torch.Tensor:
        """Run the actual denoising without classifier-free guidance
        
        Args:
          latents: Latents at time t=T
          conditioning: Conditioning tensor (without null-conditioning)
          verbose: Set to true to print debug info to the console
        
        Returns:
          Latents at time t=0
        """
        self.set_timesteps()

        # Run the denoising process
        for ix, t in enumerate(self.scheduler.timesteps):
            if verbose:
                print(f'Diffusion step {ix}')

            # Scale and expand the latents
            prep_latents = self.prepare_latents(latents, 1, t)

            # Run the network
            noise_pred = self.predict_noise(prep_latents, t, conditioning)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents

    # -------------------------------------------------------------------------
    # Convenient complete top-level functions

    def noise_to_image(
            self,
            latents: torch.Tensor,
            prompt: str, 
            verbose = False) -> torch.Tensor:
        """Run a default generative loop (denoising process) with guidance

        If a prompt is defined, applies classifier-free guidance with the
        configured scale, otherwise defaults to the unguided version.

        Args:
          latents: NC'H'W' tensor with latents
          prompt: Text prompt for conditioning or '' for unconditional.
        
        Returns:
          Generated image(s) (NCHW tensor)
        """
        if prompt == '':
            return self.noise_to_image_unconditional(latents, verbose)

        assert len(latents.shape) == 4
        batch_size = latents.shape[0]

        # Create conditional embeddings, run the denoising and return
        conditioning = self.get_text_embeddings(prompt, True, batch_size)
        latents = self.denoise(latents, conditioning, verbose)
        image = self.decode(latents)
        return image

    def noise_to_image_unconditional(
            self,
            latents: torch.Tensor,
            verbose = False) -> torch.Tensor:
        """Run a default denoising without conditioning input.
        """

        assert len(latents.shape) == 4
        batch_size = latents.shape[0]

        # Create conditional embeddings, run denoising and decode
        conditioning = self.get_text_embeddings('', False, batch_size)
        latents = self.denoise_unguided(latents, conditioning)
        image = self.decode(latents)
        return image

    # ------------------------------------------------------------------------
    # Sampling

    def sample_noise(self, batch_size:int = 1) -> torch.Tensor:
        """Sample an instance of the initial latent noise
        """
        dims = (
            batch_size, 
            self.unet.config.in_channels, 
            self.height // self.vae_block_size, 
            self.width // self.vae_block_size)
        
        latents = torch.randn(
            dims, generator=self.generator, dtype=self.dtype).to(self.device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def sample_image(self, prompt: str, verbose = False) -> torch.Tensor:
        """Complete sampling procedure, generating an image from the prompt.

        Using no_grad for convenience. If you want gradients, compose the
        individual parts manually instead.
        """
        latents = self.sample_noise(1)
        image = self.noise_to_image(latents, prompt, verbose)
        return image


# ----------------------------------------------------------------------------
# DDIM inversion

def ddim_inversion(
        diffmodel: DiffModel,
        latents: torch.Tensor, 
        conditioning: torch.Tensor, 
        return_all_latents = False,
        last_is_np = False) -> list[torch.Tensor]:
    """Runs a basic DDIM inversion loop.
    
    Args:
      diffmodel: Diffusion model to use
      latents: Initial latents at time t=0 (e.g. from diffmodel.encode)
      conditioning: Conditioning input (typically a text embedding)
      return_all_latents: If true, the latents used for each timestep will be
        returned. Otherwise, only the last timestep latents will be returned
        (still as one element of a list)
      last_is_np: Experimental, replaces the last latent with the noise
        prediction, to remove the remaining traces of the original input.

    Returns:
      if return_all_latents, include latents from all timesteps (including the
      input latents). Otherwise, only return the last latents (as a one-element
      list)
        
    Note that we can't use CFG during the DDIM inversion approach, at least not
    in an obvious way that will work. The null-text inversion paper just did
    conditioning without extra guidance in the DDIM inversion step.
    """

    t_from = 0
    diffmodel.set_timesteps()
    all_latents = []

    if return_all_latents:
        all_latents.append(latents)

    # Run the forward process
    for i in range(diffmodel.nof_iterations):

        backward_ix = len(diffmodel.scheduler.timesteps) - i - 1
        t_to = diffmodel.scheduler.timesteps[backward_ix]

        # It would seem more natural to use t_from in the call to predict_noise,
        # but that doesn't work as well, not sure why. Using t_to is consistent
        # with the null-text inversion code [Mokady et al. 2022]
        noise_pred = diffmodel.predict_noise(latents, t_to, conditioning)

        # print(f'Changing from {t_from} to {t_to}')
        latents = diffmodel.change_timestep(noise_pred, t_from, t_to, latents)

        if return_all_latents:
            all_latents.append(latents)

        t_from = t_to

    if return_all_latents:

        if last_is_np:
            all_latents[-1] = noise_pred
        return all_latents
    else:
        if last_is_np:
            return [noise_pred]
        else:
            return [latents] # Return only the last one


# ----------------------------------------------------------------------------
# Factory function

def get_diff_model(params: DiffModelParams):
    """Get a diffusion model using the provided parameters (or defaults)
    """
    return DiffModel(params.model, params.scheduler, params.seed)


# ----------------------------------------------------------------------------
# Demo code

def demo_ddim_inversion():
    """Demo the DDIM inversion + reconstruction.
    
    This is done by first running the noising step, then denoising. The
    null-text inversion procedure is not used, so the output image should be
    similar but not identical to the input.
    """

    image_path = 'YOUR IMAGE HERE'
    image = utils.load_tensor_image(image_path).unsqueeze(0)

    diffmodel = DiffModel(model_name='stabilityai/stable-diffusion-2-1')
    diffmodel.nof_iterations = 5

    prompt = 'A photo of a brown cup' # Or = ''
    conditioning = diffmodel.get_text_embeddings(prompt)

    with torch.no_grad():
        z0 = diffmodel.encode(image)
        zT = ddim_inversion(diffmodel, z0, conditioning)[-1]

        diffmodel.cf_guidance = 1.0
        rec_image = diffmodel.noise_to_image(zT, prompt, True)
        utils.save_tensor_image(rec_image, 'output/rec_bike_step5.png')


def demo_generate():
    """Demo of basic image generation
    """

    seed = 15
    it = 25
    cfg = 5

    utils.set_random_seed(15)
    diffmodel = DiffModel(seed=15)
    diffmodel.nof_iterations = it
    diffmodel.cf_guidance = cfg

    print('Starting generation')
    with torch.no_grad():
        latents = diffmodel.sample_noise(1)
        rec_image = diffmodel.noise_to_image(latents, 'A brown cup with a handle', True)

    utils.save_tensor_image(rec_image, f'output/gen_it{it}_cfg{cfg}.png')
