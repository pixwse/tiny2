import argparse, yaml
from tiny2 import fsc
from tiny2 import utils

"""Script for batch generation of images that look similar to a set of support
examples. Configurable using a yaml file, see demo_example/config.yaml.
"""

if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate new images that look similar to a set of support examples")
    parser.add_argument("--config", default=None, type=str, help="Config file, see demo_example/config.yaml for an example")
    args = parser.parse_args()

    # Load config
    with open(args.config, "rt") as file:
        yamldata = yaml.load(file)

    params = fsc.BatchGenParams()
    params.from_dict(yamldata)

    # Create logger
    logger = utils.get_default_logger('generate', params.output_dir)

    # Run actual experiment
    fsc.batch_generate(params, logger)
