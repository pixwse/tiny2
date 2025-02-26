# Original source file from https://github.com/pixwse/tiny2
# Copyright(c) 2025 Erik Landolsi, MIT license, see the LICENSE file.
import filelock, logging, os
import torch

from tiny2 import utils
from tiny2 import fsc

"""Functions for creating archives of pre-generated distillation training data.

Each archive may be split into several files, for computational efficiency when
creating the archives. They should all be interpreted as a single archive, that
could be unzipped into the same folder.
"""

class ClassInputDef:
    def __init__(self):
        self.class_name = ''
        self.supports: list[str] = []


class BatchGenParams(utils.DictParamsMixin):
    def __init__(self):

        # Input images
        self.inputs: list[ClassInputDef] = []

        # Generative model
        self.generator = fsc.generators.TintParams()

        # Set to True to load saved textinv/nulltext data from a previous run. This
        # is useful for quickly creating more images similar the prior run.
        self.load_specialization = False

        # Root directory of the output. One directory per class will be created
        # below this directory.
        self.output_dir: str = ''

        # Number of examples to generate per class
        self.nof_examples: int = 1000 

        # Set to true to output to a set of zip files directly
        self.output_zip = False

        # Optional text prompt
        self.prompt = ''


def batch_generate(
        params: BatchGenParams,
        logger: logging.Logger):
    """Top-level generation of a batch of images similar to support examples

    Args:
      params: Parameters, see BatchGenParams
      logger: Logger for progress reporting
    """

    utils.set_random_seed(129)
    logger.info('batch_generate: parameters: ' + str(params.to_dict()))

    # Ensure that all required dirs exist. Note: The intermediate 'episide0'
    # dir is to keep the same output format as for multi-episode experiments
    # (that are not yet cleaned-up and made public)
    logger.info('batch_generate: creating directories')
    episode_dir = f'{params.output_dir}/episode0'
    gendata_dir = f'{episode_dir}/gendata'
    examples_dir = f'{episode_dir}/examples'
    support_dir = f'{episode_dir}/support'
    specialization_dir = f'{episode_dir}/specialization'
    with filelock.SoftFileLock(f'{params.output_dir}/makedir.lock'):
        utils.ensure_dir_exists(examples_dir)
        utils.ensure_dir_exists(gendata_dir)
        utils.ensure_dir_exists(support_dir)
        utils.ensure_dir_exists(specialization_dir)

    # Load support examples
    logger.info('batch_generate: loading support examples')
    support_images = utils.load_image_batch('', params.inputs)
    utils.save_tensor_image(support_images, f'{support_dir}/support.png')

    # Setup the generator
    generator = fsc.get_generator(params.generator)
    generator.set_logger(logger)

    # Setup multi-process-safe zip file for storing the outputs
    zip_file_name = f'{gendata_dir}/gendata'
    zip_file = utils.MultiProcessZipFile(zip_file_name, 1000)

    # Specialize the generator
    output_path = f'{specialization_dir}/specialization.pth'
    if params.load_specialization:
        logger.info('batch_generate: loading existing specialization')
        generator.load_specialization(output_path)
    else:
        logger.info('batch_generate: running new specialization')
        spec_data = generator.specialize(support_images, params.prompt)
        torch.save(spec_data, output_path)

    # Generate examples for this class
    log_interval = 50
    nof_example_images = 10
    image_no = 0
    while generator.get_nof_accepted_examples() < params.nof_examples:

        if image_no % log_interval == 0:
            logger.info(f'batch_generate: generating image {image_no}')

        # Generate the image
        image = generator()

        # Save the image to disk and to zip file
        file_name = f'image{image_no:05}.png'
        if params.output_zip:
            # Save to 'example', add to zip, keep only the first few examples as images
            full_output_file = f'{examples_dir}/{file_name}'
            utils.save_tensor_image(image, full_output_file)
            zip_file.write(full_output_file, file_name)
            if image_no >= nof_example_images:
                os.remove(full_output_file)
        else:
            # Save as image in gendata, don't need separate examples in this case
            full_output_file = f'{gendata_dir}/{file_name}'
            utils.save_tensor_image(image, full_output_file)

        if image_no % log_interval == 0:
            logger.info(f'batch_generate: done generating image {image_no}')
    
        image_no += 1

    logger.info(f'batch_generate: {image_no} images generated, all done!')
