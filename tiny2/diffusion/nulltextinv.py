# Original source file from https://github.com/pixwse/tiny2
# Copyright(c) 2025 Erik Landolsi, MIT license, see the LICENSE file.
import torch
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from tiny2.diffusion.diffmodel import *
from tiny2 import utils

"""Main functionality for null-text inversion according to "Mokady et al. 2022,
Null-text inversion for editing real images using guided diffusion models".
Inspired by the paper and the author's original repo
https://github.com/google/prompt-to-prompt (Apache 2.0 license), but rewritten
from scratch to fit our framework.
"""

class NullInversionData:
    """Class representing the output from a null-text inversion procedure.
    This is all data required to later reconstruct the input image.
    """

    def __init__(self):

        # Latent noise at the final timestep (NC'H'W' tensor with N=1)
        self.latents = torch.Tensor()

        # Uncondintional embeddings at each timestep (NL tensors with N=1)
        self.unc_embeddings : list[torch.Tensor] = []

        # Original image file name. For convenience when loading/storing this
        # structure to disk.
        self.input_file_name = ''

        # Prompt used when running the inversion. For convenience when
        # loading/storing this structure to disk.
        self.prompt = ''

    def half(self):
        """Convenience function, changes all contents to half precision
        """
        self.latents = self.latents.type(torch.float16)
        for ix in range(len(self.unc_embeddings)):
            self.unc_embeddings[ix] = self.unc_embeddings[ix].type(
                torch.float16)


class NullInvParams(utils.DictParamsMixin):
    """Parameters for the null-text inversion procedure
    """

    def __init__(self):

        # Number of optimization steps to run for each timestep
        self.nof_inner_steps = 10

        # Stop optimizing when the loss drops below this level
        self.epsilon = 1e-5

        # Learning rate
        self.lr = 1e-2

        # Set to True to skip the optimization, essentially just producing a 
        # DDIM inversion, returning the unadjusted null-text, embeddings and 
        # final latents such that we can still reconstruct the input using 
        # NullInversion.reconstruct. Useful for comparison.
        self.bypass_inversion = False


class NullInversion:
    
    def __init__(
            self, 
            diffmodel: DiffModel,
            params = NullInvParams()):
        """Construct a null-text inversion engine.
        
        Args:
          diffmodel: Underlying diffusion model
          params: Algorithm parameters (see NullInvParams)
        """
        self.diffmodel = diffmodel
        self.params = params

    def _get_alpha_cumprod(self, timestep: int) -> float:
        return self.diffmodel.scheduler.alphas_cumprod[timestep]

    def _optimize_null_embeddings_one_timestep(
            self,
            t: int,
            t_prev: int,
            uncond_embedding: torch.Tensor,
            cond_embedding: torch.Tensor,
            latents: torch.Tensor, 
            latents_prev: torch.Tensor) -> torch.Tensor:
        """Optimize the null-text embeddings of a single time step

        Args:
          t: Timestep (in 0...1000)
          t_prev: Timestep of latents_prev, that we're reconstructing
          uncond_embedding: Initial unconditional embedding, optimization starting point
          cond_embedding: Conditional embedding, will be kept fixed
          latents: Latents of the current timestep
          latents_prev: Latents of the previous timestep, used as optimization target 
        """

        # Predict the noise using the conditional embedding for the current
        # timestep. This will be fixed throughout the optimization.
        with torch.no_grad():
            np_cond = self.diffmodel.predict_noise(latents, t, cond_embedding)

        # The original implementation did something like
        # lr = 1e-2 * (1. - ix / 100.)
        # which introduces a dependency on the number of diffusion timesteps.
        # The paper only mentions LR = 0.01, which we found to work just as well.
        uncond_embedding = uncond_embedding.clone().detach()
        uncond_embedding.requires_grad = True
        optimizer = Adam([uncond_embedding], lr = self.params.lr)

        if self.params.bypass_inversion:
            return uncond_embedding

        # Optimization loop
        for _ in range(self.params.nof_inner_steps):

            self.diffmodel.unet.zero_grad(set_to_none=True)

            # Predict noise using the unconditional embedding (under optimization)
            np_uncond = self.diffmodel.predict_noise(latents, t, uncond_embedding)

            # Combine the conditional and unconditional estimate using CFG
            noise_pred = self.diffmodel.apply_cfg(np_uncond, np_cond)

            # Estimate the previous latents
            latents_prev_rec = self.diffmodel.change_timestep(
                noise_pred, t, t_prev, latents)

            # Optimize (compare estimate with actual)
            loss = nnf.mse_loss(latents_prev_rec, latents_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()

            # The original implementation compared to epsilon + ix * 2e-5,
            # which introduces a dependency on the number of diffusion 
            # timesteps. The paper only epsilon = 1e-5, which we found to work
            # just as well.
            if loss_item < self.params.epsilon:
                break
        
        return uncond_embedding

    def _optimize_null_embeddings(
            self,
            prompt: str,
            all_latents: list[torch.Tensor]
            ) -> list[torch.Tensor]:
        """Runs the null embeddings loop.

        Args:
          prompt: Text prompt to use as conditioning input
          all_latents: List of the latents (NC'H'W') for each time step, as
            produced by a previous ddim inversion pass.

        This is the core functionality. Runs a diffusion loop and optimizes
        the null embedding at each time step.
        
        Returns:
          List of optimized null embeddings, one for each time step.
        """
        conditioning = self.diffmodel.get_text_embeddings(prompt, True)
        uncond_embedding, cond_embedding = conditioning.chunk(2)

        uncond_embeddings_list = []
        current_latents = all_latents[-1]

        for ix, t in enumerate(self.diffmodel.scheduler.timesteps):

            # print(f'Null-optimization: i={ix}')

            # Latents of previous timestep (used in the target function)
            latents_prev = all_latents[len(all_latents) - ix - 2]
            t_prev = self.diffmodel.get_timestep(ix+1)

            # Run the optimization for this timestep
            uncond_embedding = self._optimize_null_embeddings_one_timestep(
                t, t_prev,
                uncond_embedding, cond_embedding,
                current_latents, latents_prev)

            uncond_embeddings_list.append(uncond_embedding.detach())

            # Perform the final update of the latents based on the optimized
            # unconditional embeddings, similar to the inner loop of 
            # DiffModel.noise_to_image
            with torch.no_grad():
                conditioning = torch.cat([uncond_embedding, cond_embedding])

                prep_latents = self.diffmodel.prepare_latents(current_latents, 2, t)
                noise_pred = self.diffmodel.predict_noise(prep_latents, t, conditioning)

                [np_uncond, np_cond] = noise_pred.chunk(2)
                noise_pred = self.diffmodel.apply_cfg(np_uncond, np_cond)
                
                current_latents = self.diffmodel.change_timestep(
                    noise_pred, t, t_prev, current_latents)

        # Release some CUDA memory
        self.diffmodel.unet.zero_grad(set_to_none=True)

        return uncond_embeddings_list
    
    def invert(
            self, 
            image: torch.Tensor,
            prompt: str
            ) -> NullInversionData:
        """Runs the entire null-text inversion procedure.

        Args:
          image: NCHW tensor
          prompt: Text prompt to use, or '' for an unconditional process.

        Returns:
          Structure with all data required to reconstruct the input image.
        """
        assert len(image.shape)==4

        with torch.no_grad():
            latents_t0 = self.diffmodel.encode(image)
            conditioning = self.diffmodel.get_text_embeddings(prompt)
            all_latents = ddim_inversion(self.diffmodel, latents_t0, conditioning, True)
        
        unc_embeddings = self._optimize_null_embeddings(prompt, all_latents)

        result = NullInversionData()
        result.unc_embeddings = unc_embeddings
        result.latents = all_latents[-1]
        result.prompt = prompt
        return result

    def reconstruct(
            self, 
            nullinv_data: NullInversionData
            ) -> torch.Tensor:
        """Reconstructs an image from the null-text inversion data.

        This loop is similar to the basic diffusion reconstruction loop found
        in e.g. DiffModel, with the exception that the optimized null-text
        embeddings are used for each timestep.

        Args:
          nullinv_data: Output from a previous run of the invert procedure.
        
        Returns:
          The reconstructed input image.
        """
        # Create text embeddings. The unconditional embedding will then be
        # changed for every iteration
        cond_emb = self.diffmodel.get_text_embeddings(nullinv_data.prompt, False)
        conditioning = torch.cat([nullinv_data.unc_embeddings[0], cond_emb])

        # Create initial noise and prepare the denoising schedule
        latents = nullinv_data.latents  
        self.diffmodel.set_timesteps()

        # Run the denoising process
        with torch.no_grad():
            for ix, t in enumerate(self.diffmodel.scheduler.timesteps):
                conditioning[0] = nullinv_data.unc_embeddings[ix]
                t_prev = self.diffmodel.get_timestep(ix+1)

                # expand and scale the latents
                prep_latents = self.diffmodel.prepare_latents(latents, 2, t)

                # Run the u-net
                noise_pred = self.diffmodel.predict_noise(prep_latents, t, conditioning)

                # Perform classifier-free guidance
                np_uncond, np_cond = noise_pred.chunk(2)
                noise_pred = self.diffmodel.apply_cfg(np_uncond, np_cond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.diffmodel.change_timestep(noise_pred, t, t_prev, latents)

        # Convert to image and return
        image = self.diffmodel.decode(latents)
        return image
