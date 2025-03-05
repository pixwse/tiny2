# Original source file from https://github.com/pixwse/tiny2
# Copyright(c) 2025 Erik Landolsi, MIT license, see the LICENSE file.
import random
import torch
import torch.nn.functional as F
import logging
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset, DataLoader

from tiny2 import utils
from tiny2.diffusion import *

"""Textual inversion 

Following "Gal et al. 2022, An image is worth one word: Personalizing text-to-
image generation using textual inversion". Inspired by the paper and the
author's original repo (https://github.com/rinongal/textual_inversion (MIT 
license) but rewritten from scratch to fit our framework.

The textual inversion is configured using a prompt template where each token
has a #-prefix. Then, there is a token config struct that defines the
properties of each spefic #-token.

Example:

  template = "An image of a #brown #cup"
  token_config = [
    {'placeholder': '#brown', 'type': 'shared', 'init': 'brown'}
    {'placeholder': '#cup',   'type': 'shared', 'init': 'cup'}
  ]

The config struct is designed to be extensible, treating tokens in different
ways. Currently, only the 'shared' type is implemented (basic token, shared
between all input images). 

The training procedure directly modifies the tokenizer and the embeddings in
the diffusion model. It also outputs the new embeddings, for saving etc.

Example output:
  {'#brown': [...], '#cup': [...]}

To generate images using the optimized model, just call the diffusion model
as usual with one or more new tokens included. For example:

  img = model.sample_image("A children's drawing of a #brown #cup")
"""

# ----------------------------------------------------------------------------
# Config 

class TextinvParams(utils.DictParamsMixin):
    """Parameters for the textual inversion training.
    """

    def __init__(self):

        # Total number of training steps. 5000 was used in the
        # original paper.
        self.nof_steps = 5000

        # Only include time steps above this number in the textinv training.
        # Might help reduce artifacts due to upsampling of the input image.
        self.min_timestep = 250

        # Initial learning rate 
        self.init_lr = 1e-3

        # Experimental feature: Train only using the exact same timesteps
        # that are defined in the scheduler. The idea was that it might lead
        # to a more focused training (but seems to not make a big difference)
        self.use_exact_timesteps = False

        # Set to true to save samples generated at periodic timesteps during
        # the optimization, to see how the results progress
        self.save_intermediate_samples = False

        # Filename base to use for saving intermediate samples (if enabled)
        self.intermediate_samples_filename_base = ''

        # Experimental feature. Speficies a roi, [xmin, xmax, ymin, ymax]
        # where the loss is applied. If None, the whole image is used.
        self.roi: list[int]|None = None

        # RandAug augmentation parameters. Set to zero to disable augmentation.
        self.randaug_num: int = 0
        self.randaug_magn: int = 0

        # Set to True to print stuff to the console
        self.verbose = False


class TokenConfig:
    """Struct-like class defining the configuration for one token.
    See the file-level comment for details.
    """

    def __init__(
            self,
            placeholder = '',
            type = 'shared',
            init = ''
            ):
        """Initialie a token config
        
        Args:
          placeholder: Placeholder for token to optimize (e.g. '#cup')
          type: Must be 'shared' (other options may be added later)
          init: Word used to initialize the embedding.
        """
        
        assert type == 'shared'
        self.placeholder = placeholder
        self.type = type
        self.init = init

# -----------------------------------------------------------------------------
# Default prompt templates, copied from the original textinv repo

def get_default_templates(name: str):
    if name == 'imagenet_templates_small':
        return [
            'a photo of a #object',
            'a rendering of a #object',
            'a cropped photo of the #object',
            'the photo of a #object',
            'a photo of a clean #object',
            'a photo of a dirty #object',
            'a dark photo of the #object',
            'a photo of my #object',
            'a photo of the cool #object',
            'a close-up photo of a #object',
            'a bright photo of the #object',
            'a cropped photo of a #object',
            'a photo of the #object',
            'a good photo of the #object',
            'a photo of one #object',
            'a close-up photo of the #object',
            'a rendition of the #object',
            'a photo of the clean #object',
            'a rendition of a #object',
            'a photo of a nice #object',
            'a good photo of a #object',
            'a photo of the nice #object',
            'a photo of the small #object',
            'a photo of the weird #object',
            'a photo of the large #object',
            'a photo of a cool #object',
            'a photo of a small #object',
            'an illustration of a #object',
            'a rendering of a #object',
            'a cropped photo of the #object',
            'the photo of a #object',
            'an illustration of a clean #object',
            'an illustration of a dirty #object',
            'a dark photo of the #object',
            'an illustration of my #object',
            'an illustration of the cool #object',
            'a close-up photo of a #object',
            'a bright photo of the #object',
            'a cropped photo of a #object',
            'an illustration of the #object',
            'a good photo of the #object',
            'an illustration of one #object',
            'a close-up photo of the #object',
            'a rendition of the #object',
            'an illustration of the clean #object',
            'a rendition of a #object',
            'an illustration of a nice #object',
            'a good photo of a #object',
            'an illustration of the nice #object',
            'an illustration of the small #object',
            'an illustration of the weird #object',
            'an illustration of the large #object',
            'an illustration of a cool #object',
            'an illustration of a small #object',
            'a depiction of a #object',
            'a rendering of a #object',
            'a cropped photo of the #object',
            'the photo of a #object',
            'a depiction of a clean #object',
            'a depiction of a dirty #object',
            'a dark photo of the #object',
            'a depiction of my #object',
            'a depiction of the cool #object',
            'a close-up photo of a #object',
            'a bright photo of the #object',
            'a cropped photo of a #object',
            'a depiction of the #object',
            'a good photo of the #object',
            'a depiction of one #object',
            'a close-up photo of the #object',
            'a rendition of the #object',
            'a depiction of the clean #object',
            'a rendition of a #object',
            'a depiction of a nice #object',
            'a good photo of a #object',
            'a depiction of the nice #object',
            'a depiction of the small #object',
            'a depiction of the weird #object',
            'a depiction of the large #object',
            'a depiction of a cool #object',
            'a depiction of a small #object'
        ]
    else:
        raise Exception('Template name unknown')


class TextInvDataset(Dataset):
    """Dataset wrapper for textinv training.

    Related to torch.data.TensorDataset, but includes a text prompt and 
    image number in the output (tailored for textinv training).
    """

    def __init__(
            self, 
            inputs: torch.Tensor,
            prompts: list[str],
            random_prompts: bool = False,
            randaug_num: int = 0,
            randaug_magn: int = 0,
            dataset_size = 500) -> None:
        """Initialize a dataset wrapper for training a textinv model

        Args:
          inputs: NCHW tensor of input examples to adjust to (pixel range 0..1)
          prompts: One text prompt per input image or a set of prompts to choose
            from randonly, depending on the 'random_prompts' flag
          random_prompts: If true, one prompt is assigned to each image at
            random. Otherwise, they are paired (must be the same number)
          randaug_num: 'number' parameter in the RandAug augmentation
          randaug_num: 'magnitude' parameter in the RandAug augmentation
          dataset_size: Specifies the virtual dataset size, use to control how
            many examples to consider being part of one epoch.
        """

        if not random_prompts:
            assert inputs.shape[0] == len(prompts)

        assert torch.min(inputs) >= 0.0
        assert torch.max(inputs) <= 1.0

        self.inputs = inputs
        self.prompts = prompts
        self.random_prompts = random_prompts
        self.dataset_size = dataset_size

        if randaug_num > 0:
            self.augmenter = utils.get_float_randaug(randaug_num, randaug_magn)
        else:
            self.augmenter = None

    def __getitem__(self, index):
        """Get one training example for the textinv training.
            
        Args:
          index: Training example number

        Returns:
          (image, prompt)
        """
        image_no = index % self.inputs.shape[0]

        if self.augmenter:
            image = self.augmenter(self.inputs[image_no])
        else:
            image = self.inputs[image_no]

        if self.random_prompts:
            prompt_no = random.randint(0, len(self.prompts)-1)
            return image, self.prompts[prompt_no]
        else:
            return image, self.prompts[image_no]

    def __len__(self):
        return self.dataset_size


# ----------------------------------------------------------------------------
# Load/save

def make_textinv_dict(
        model: DiffModel,
        placeholder_tokens: list[str], 
        placeholder_ids: list[int]) -> dict:
    """Make a dict containing all information about a textinv model that needs 
    to be saved. See save_textinv below for more info.
    """

    new_weights = model.text_encoder.get_input_embeddings(
        ).weight[placeholder_ids]

    save_data = {}
    for ix, token in enumerate(placeholder_tokens):
        save_data[token] = new_weights[ix].clone()
    
    return save_data


def save_textinv(
        model: DiffModel,
        placeholder_tokens: list[str],
        placeholder_ids: list[int],
        file_name: str):
    """Save a textual inversion model to a file.
     
    The file will contain a dictionary on the form
      {"#token1": [...], "#token2": [...]}
    """

    save_data = make_textinv_dict(model, placeholder_tokens, placeholder_ids)
    torch.save(save_data, file_name)


def setup_saved_textinv(model: DiffModel, saved_data: dict):
    """Setup a textual inversion model with saved new text embeddings
    """

    # Add placeholder tokens and get their IDs
    placeholder_tokens = list(saved_data.keys())
    nof_added_tokens = model.tokenizer.add_tokens(placeholder_tokens)
    assert nof_added_tokens == len(placeholder_tokens)
    placeholder_ids = model.tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Initialize embeddings of the new tokens from the provided weights
    model.text_encoder.resize_token_embeddings(len(model.tokenizer))
    token_embeddings = model.text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token, id in zip(placeholder_tokens, placeholder_ids):
            token_embeddings[id] = saved_data[token].clone()


def load_textinv(model: DiffModel, file_name: str):
    """Setup a textual inversion model saved to file by save_textinv
    """
    saved_data = torch.load(file_name)
    setup_saved_textinv(model, saved_data)

# ----------------------------------------------------------------------------
# Error checking

def _check_hashtag_token(
        model: DiffModel,
        template: str):
    """Check that the template looks correct.
    
    Return true if everything looks correct and false if there is a '#' token
    in the template (indicating that some #-word has not been added properly)
    """
    hashtag_id = model.tokenizer.encode('#', add_special_tokens=False)[0]
    template_ids = model.tokenizer.encode(template, add_special_tokens=False)
    return not hashtag_id in template_ids


def _check_token_consistency(
        token_config: list[TokenConfig],
        templates: list[str]):
    """Raise an exception if any of the templates uses tokens that are not in
    the config (since that would cause unexpected results).
    """

    placeholders = [tc.placeholder for tc in token_config]

    for template in templates:
        words = template.split()
        for word in words:
            if word[0] == '#':
                if not word in placeholders:
                    raise ValueError(f'token {word} not included in token config') 


# ----------------------------------------------------------------------------
# Misc tempate/token setup helpers

def add_token_index(token_base: str, index: int):
    """Add an index suffix to a token.
    
    Intended for UNIQUE and SUBSPACE tokens. For example, 
    add_token_index('#pose', 1) returns '#pose/1'
    """
    return token_base + '/' + str(index)


def add_token_index_range(token_base: str, nof_indices: int) -> list[str]:
    """Create a set of tokens with suffixes 0..nof_indices-1.
    
    For example, add_token_index_range('#pose', 3) returns
    ['#pose/0', '#pose/1', '#pose/2']
    """
    tokens = [add_token_index(token_base, ix) for ix in range(nof_indices)]
    return tokens


def expand_tokens(
        token_config: list[TokenConfig], 
        nof_images: int) -> tuple[list[str], list[str]]:
    """Get all individual tokens, placeholders and init tokens.

    At the moment, we only extract the placeholders and init tokens
    from the config, but we could add more functionality in the future.
    
    Returns:
      Tuple containing expanded tokens and corresponding init tokens.
    """
    placeholder_tokens = []
    init_tokens = []

    for tc in token_config:
        if tc.type == 'shared':
            placeholder_tokens.append(tc.placeholder)
            init_tokens.append(tc.init)
        else:
            raise ValueError(f'token type not supported: {tc.type}')

    return placeholder_tokens, init_tokens


# ----------------------------------------------------------------------------
# Complete train/test procedures

class IntermediateTextinvResultsSaver:
    """Helper class for saving intermediate results during the optimization.
    """

    def __init__(
            self, 
            model: DiffModel, 
            prompt: str,
            filename_base: str,
            interval = 500,
            nof_examples = 10):

        self.model = model
        self.prompt = prompt
        self.filename_base = filename_base
        self.interval = interval
        self.nof_examples = nof_examples
        self.current_iteration = 0

        self.latents = []
        for _ in range(self.nof_examples):
            self.latents.append(model.sample_noise())

    def poll(self):
        """Call once every operation step, will save sample images periodically
        """
        self.current_iteration += 1
        if self.current_iteration % self.interval == 0:
            self._do_save(self.current_iteration)            

    def _do_save(self, iteration_no: int):
        for ix in range(self.nof_examples):
            img = self.model.noise_to_image(self.latents[ix], self.prompt)
            file_name = f'{self.filename_base}_it{iteration_no}_s{ix}.jpg'
            utils.save_tensor_image(img, file_name)


def _get_embeddings_data(model: DiffModel) -> torch.Tensor:
    """Get the tensor containing all token embeddings.
    
    Returns:
      TxF tensor, where T is the number of tokens and F the
      feature space dimensionality.
    """
    embeddings_module = model.text_encoder.get_input_embeddings()
    assert type(embeddings_module) == torch.nn.Embedding
    return embeddings_module.weight.data


def _init_token_embeddings(
        model: DiffModel,
        nof_support_images: int,
        template: str,
        token_config: list[TokenConfig]) -> tuple[list[str], list[int]]:
    """Initialize token embeddings for new tokens.
    
    Args:
      model: The model to modify, original SD model without new tokens added.
      nof_support_images: Currently unused, reserved for future use.
      template: Text prompt, including new tokens
      token_config: Config for each new token (see TokenConfig)

    Returns:
      (tokens, ids): Tuple with list of tokens and IDs for the added tokens
    """

    # Get placeholder and init tokens
    placeholder_tokens, init_tokens = expand_tokens(
        token_config, nof_support_images)

    # Get the IDs of the initializer tokens
    # Note: The treatment of </w> etc is a bit confusing. For some reason,
    #   model.tokenizer.encode('brown cup') and 
    #   model.tokenizer.encode(['brown', 'cup'])
    # produce different IDs. The convert_tokens_to_ids only supports vector 
    # input. When encoding an entire prompt, the IDs match the output of
    #   model.tokenizer.encode('brown cup') 
    # which are identical to the IDs returned by
    #   model.tokenizer.convert_tokens_to_ids(['brown</w>', 'cup</w>'])
    init_tokens = [t + '</w>' for t in init_tokens]
    init_ids = model.tokenizer.convert_tokens_to_ids(init_tokens)

    # Add placeholder tokens and get their IDs
    nof_added_tokens = model.tokenizer.add_tokens(placeholder_tokens)
    assert nof_added_tokens == len(placeholder_tokens)
    placeholder_ids = model.tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Initialize embeddings of new tokens from their corresponding init token
    model.text_encoder.resize_token_embeddings(len(model.tokenizer))
    token_embeddings = _get_embeddings_data(model)

    with torch.no_grad():
        for p_id, i_id in zip(placeholder_ids, init_ids):
            token_embeddings[p_id] = token_embeddings[i_id].clone()
    
    # Check for mistakes. It turns out that if the template contains #-words
    # that have not been added, a hashtag token is added. This can be used to
    # check for consistency
    if not _check_hashtag_token(model, template):
        raise ValueError(
            'Some # word in the template does not have a matching token config')

    return placeholder_tokens, placeholder_ids


def _ensure_size(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Ensure that an image batch has size (width, height), resize if needed.
    """
    assert len(images.shape) == 4
    if not images.shape[2:4] == torch.Size([width, height]):
        return F.interpolate(images, (width, height), mode='bilinear')
    else:
        return images


def _sample_timesteps(
    model: DiffModel, batch_size: int, params: TextinvParams) -> torch.Tensor:
    """Get random timstep(s) to use in the optimization.

    If params.use_exact_timesteps is set, we will only choose from the exact
    timesteps available in the schedule. Otherwise, timsteps will be selected
    at random. 

    Returns:
      Tensor containing 'batch_size' randomly selected timesteps.
    """
    if params.use_exact_timesteps:
        filtered_timesteps = model.scheduler.timesteps[
            model.scheduler.timesteps > params.min_timestep]
        indices = torch.randint(0, len(filtered_timesteps), (batch_size,))
        timesteps = filtered_timesteps[indices]
        return timesteps.to(model.device)

    else:
        return torch.randint(
            params.min_timestep,
            model.scheduler.config.num_train_timesteps,
            (batch_size,), 
            device=model.device)


def _get_latent_roi(
            roi_rect: list[float], 
            block_size: float,
            latent: torch.Tensor) -> torch.Tensor:
        """Get a ROI in latent space from coordinates defined in image space

        Args:
          roi_rect: [xmin, xmax, ymin, ymax] of the ROI in image space
          block_size: Number of image pixels (in each dimension) corresponding 
            to a latent pixel
          latent: Input latent
        
        Returns:
          Tensor containing the latent part corresponding to the image ROI
        """
        xmin, xmax = int(xmin / block_size), int(xmax / block_size)
        ymin, ymax = int(ymin / block_size), int(ymax / block_size)
        roi = latent[:, :, xmin:xmax+1, ymin:ymax+1]
        return roi


def _do_train_textinv(
        model: DiffModel,
        train_dataset: TextInvDataset,
        template: str,
        token_config: list[TokenConfig],
        logger: logging.Logger,
        params: TextinvParams) -> dict:
    """Internal function - does all actual work

    This function encapsulates the common parts between train_textinv
    and train_textinv_multitemplate below - either using the same prompt
    for all images, potentially with more tokens, or using the multi-prompt
    method from the original textinv paper.

    Args:
      model: The model to update
      train_dataset: Dataset returning augmented support images + prompts. The
        prompts could be unique for each image, shared for all, or selected
        randomly.
      template: One of the prompts, only used for generating progress images
        during the optimization (if params.save_intermediate_samples is set).
      logger: Logger for logging progress.
      params: Misc method parameters.

    Returns:
      The optimized embeddings as a dictionary, mapping token name to embedding
      vector. Note that the input model will also be updated with these new
      embeddings and is ready to use directly.
    """
    logger.info('textinv: preparing training')
    placeholder_tokens, placeholder_ids = _init_token_embeddings(
        model, 0, template, token_config)

    # Freeze everything except the token embeddings
    model.vae.requires_grad_(False)
    model.unet.requires_grad_(False)
    text_model = model.text_encoder.text_model
    text_model.encoder.requires_grad_(False)
    text_model.final_layer_norm.requires_grad_(False)
    text_model.embeddings.position_embedding.requires_grad_(False)

    # Gather all parameters and initialize the optimizer
    # Note: The textinv paper [Gal et al] used a base LR of 0.005 (scaled by
    # batch size + nof GPUs to 0.04)
    all_params = model.text_encoder.get_input_embeddings().parameters()
    optimizer = torch.optim.AdamW(all_params, lr=1e-3)

    # keep original embeddings such that we can restore the once we don't want
    # to update + create mask to use to selectively restore the frozen ones 
    orig_embeddings = _get_embeddings_data(model).clone().to(model.device)

    frozen_ixs = torch.ones(
        (len(model.tokenizer),), dtype=torch.bool).to(model.device)

    for id in placeholder_ids:
        frozen_ixs[id] = False

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Setup LR scheduler
    max_epochs = int(params.nof_steps / len(train_loader))
    lr_scheduler = get_scheduler(
        'constant', optimizer=optimizer, num_warmup_steps = 100,
        num_training_steps=params.nof_steps, num_cycles = 1)

    model.text_encoder.train()
    model.set_timesteps()

    if params.save_intermediate_samples:
        results_saver = IntermediateTextinvResultsSaver(
            model, template, params.intermediate_samples_filename_base, 500, 5)

    # Optimization loop
    logger.info('textinv: starting training')
    for epoch_no in range(max_epochs):
        logger.info(f'textinv: epoch_no: {epoch_no}')

        loss_sum = 0
        count = 0
        for images, template in train_loader:
            assert type(images) == torch.Tensor
            batch_size = images.shape[0]
            image_hires = _ensure_size(images, model.width, model.height)

            # Tokenize the template
            token_ids = model.get_token_ids(template)

            if params.verbose and count % 10 == 0:
                print(f'count: {count}')

            # Convert images to latent space
            latents = model.encode(image_hires)

            # Run foward diffusion
            noise = model.sample_noise(batch_size)
            timesteps = _sample_timesteps(model, batch_size, params)
            noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            # Must be done within the loop, since the embeddings are updated
            conditioning = model.get_id_embeddings(token_ids)
            conditioning = conditioning.expand(batch_size, *conditioning.shape[1:])

            # Predict the noise residual
            model_pred = model.predict_noise(noisy_latents, timesteps, conditioning)

            # Compute loss and run optimizer
            if not params.roi:
                loss = F.mse_loss(model_pred, noise, reduction='mean')
            else:
                pred_roi = _get_latent_roi(params.roi, model.vae_block_size, model_pred)
                noise_roi = _get_latent_roi(params.roi, model.vae_block_size, noise)
                loss = F.mse_loss(pred_roi, noise_roi, reduction='mean')
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_sum += loss.item()
            count += 1

            # Revert updates done to frozen embeddings
            with torch.no_grad():
                _get_embeddings_data(model)[frozen_ixs] = orig_embeddings[frozen_ixs]
            
            if params.save_intermediate_samples:
                assert results_saver != None
                results_saver.poll()

        average_loss = loss_sum / count
        logger.info(f'textinv: iteration_count: {count}')
        logger.info(f'textinv: average_loss: {average_loss}')

    dict = make_textinv_dict(model, placeholder_tokens, placeholder_ids)
    return dict


def train_textinv(
        model: DiffModel, 
        support_images: torch.Tensor,
        template: str,
        token_config: list[TokenConfig],
        logger: logging.Logger, 
        params: TextinvParams) -> dict:
    """Top-level textinv training procedure.

    Updates the diff model with new tokens and their optimized weights.
    To use the new model, just generate an image from the diffusion model
    as usual, e.g. using model.sample_image()

    Args:
      model: DiffModel instance with all SD components 
      support_images: Images showing our object (NCHW, pixel values in [0..1])
      template: A text template used for the prompt. Examples:
        "An image of a #cup"
        "An image of a #brown #cup"
        "An image of a #cup viewed from #pose in the style of #style"
      token_config: Configuration for the placeholder tokens used in the
        template. See file-level comment.
      logger: Logger, used for monitoring progress
      params: Training parameters, see TextinvParams
        
    Returns:
      The optimized embeddings as a dictionary, mapping token name to embedding
      vector. Note that the input model will also be updated with these new
      embeddings and is ready to use directly.
    """

    assert torch.min(support_images) >= 0.0
    assert torch.max(support_images) <= 1.0
    _check_token_consistency(token_config, [template])

    # Setup the dataset using the same prompt for each support image
    prompts = [template] * len(support_images)
    train_dataset = TextInvDataset(
        support_images, prompts, False, 
        params.randaug_num, params.randaug_magn)

    return _do_train_textinv(
        model, train_dataset, template, token_config, logger, params)


def train_textinv_multitemplate(
        model: DiffModel, 
        support_images: torch.Tensor,
        init_token: str,
        logger: logging.Logger, 
        params: TextinvParams
        ) -> dict:
    """Alternative top-level textinv training procedure

    This version uses multiple text templates that are selected at random. This
    is more similar to the original textinv paper. Only a single adjusted token
    is supported in this version, since that is how the templates are expressed.

    To use the new model, just generate an image from the diffusion model as
    usual, e.g. using model.sample_image()

    Args:
      model: DiffModel instance with all SD components 
      support_images: NCHW tensor with images showing our object
      init_token: Word to use for initializing the optimized token.
      logger: Logger, used for monitoring progress
      params: Training parameters, see TextinvParams
        
    Returns:
      The optimized embeddings as a dictionary, mapping token name to embedding
      vector. Note that the input model will also be updated with these new
      embeddings, and is ready to use directly.
    """

    assert torch.min(support_images) >= 0.0
    assert torch.max(support_images) <= 1.0

    # Setup the dataset to use randomly selected templates
    templates = get_default_templates('imagenet_templates_small')
    train_dataset = TextInvDataset(
        support_images, templates, True, 
        params.randaug_num, params.randaug_magn)

    # Create token config using the single token '#object'
    token_config = [TokenConfig('#object', 'shared', init_token)]
    _check_token_consistency(token_config, templates)

    return _do_train_textinv(
        model, train_dataset, templates[0], token_config, logger, params)
