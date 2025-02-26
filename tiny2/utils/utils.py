# Original source file from https://github.com/pixwse/tiny2
# Copyright(c) 2025 Erik Landolsi, MIT license, see the LICENSE file.
import copy, logging, os, random, typing
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F

from .io import *

"""Common tools for DL with pytorch.
"""

# ------------------------------------------------------------------------------
# Parameter structs

def _get_slashpath_reference(dict: dict, key_path: str) -> tuple[dict|None, str]:
    """Helper function for '/' path treatment.
    Finds a reference to the second-to last element, since that's one that
    we can return by-reference. Returns a dict reference and a key such that
    ref[key] is the object we're looking for.
    
    If the dict does not contain objects matching the key_path, the function
    returns (None, error_key), where error_key was the first key that was not
    found in the dict.
    """
    keys = key_path.split('/')
    ref = dict
    for k in keys[:-1]:
        if not k in ref:
            return None, k
        else:
            ref = ref[k]

    if not keys[-1] in ref:
        return None, keys[-1]

    return ref, keys[-1]


def update_dict(
        dict: dict, 
        key_path: str, 
        value: object):
    """Safer and extended version of dict.update.
    
    Updates a nested dictionary value in-place. Supports '/'-paths, throws an
    error if the element to update is not already in the dictionary.
    Example:
      dict = {'name': 'test', 'point': {'x':10, 'y':15}}
      update_dict(dict, 'point/x', 30)
    """
    ref, last_key = _get_slashpath_reference(dict, key_path)
    if not ref:
        raise ValueError(f"dict path invalid: key '{last_key}' not found")
    else:
        ref[last_key] = value


def override_baseline_dict(baseline: dict, overrides: dict) -> dict:
    """Clone dictionary, updating certain values on the fly.
    
    Produce a new dictionary where some values in a nested baseline dictionary
    have been updated, without changing the original. The changing if values is 
    done using key paths with '/'-separators.
    Example:
      baseline = {'name': 'test', 'point': {'x':10, 'y':15}}
      override = {'point/x': 15}
    override_baseline_dict(baseline, override) now returns:
      {'name': 'test', 'point': {'x':15, 'y':15}}
    The original baseline dict is not affected
    """
    updated = copy.deepcopy(baseline)
    for (key_path, value) in overrides.items():
        update_dict(updated, key_path, value)
    
    return updated


def has_param(dict: dict, param: str) -> bool:
    """Check if a parameter or parameter group exists in a dictionary.
    
    Supports deep indexing with '/' paths, such that we can call e.g.
    has_param(dict, 'optimizer/scheduler/init_lr')

    Returns:
      True if the parameter exists and has a non-None value, otherwise False.
    """
    ref, last_key = _get_slashpath_reference(dict, param)
    if not ref:
        return False
    else:
        return ref[last_key] != None


def get_param(dict: dict, param: str):
    """Get a parameter from a dictionary with deep indexing.
    Example:
      get_param(dict, 'optimizer/scheduler/init_lr')
    """
    ref, last_key = _get_slashpath_reference(dict, param)
    if not ref:
        raise ValueError(f"dict path invalid: key '{last_key}' not found")
    else:
        return ref[last_key]


def get_with_default(
        dict: dict, 
        param: str,
        default):
    """Deep-indexing version of get with default.
    
    Get a parameter from a dictionary if defined, otherwise return a specified 
    default value. Supports deep indexing, such that we can call e.g.
    get_with_default(dict, 'optimizer/scheduler/init_lr', 0.1)
    """
    if has_param(dict, param):
        return get_param(dict, param)
    else:
        return default


class DictParamsMixin:
    """Helper base class for parameter structs
    
    Parameter classes deriving from DictParamMixin can be fully typed, which
    aids editing, spotting errors, etc. At the same time, they can be easily
    converted to/from dictionaries, supporting easy logging, serialization,
    overriding when running multiple experiments, etc.
    """

    def _get_fields(self) -> list[str]:
        """Get all fields (attributes that are not dunders, not functions) 
        """
        fields = [
            x 
            for x in dir(self) 
            if not x.startswith('__') and not callable(getattr(self, x))]
        return fields

    def to_dict(self) -> dict:
        """Convert the class to a dictionary.

        All native fields (str, bool, int, float) are converted directly, 
        while nested DictParamMixin objects are converted recursively.
        """
        keys = self._get_fields()
        result = {}
        for k in keys:
            val = self.__getattribute__(k)
            result[k] = (
                val.to_dict() if isinstance(val, DictParamsMixin) else val)

        return result

    def from_dict(self, dict: dict):
        """Initializes the parameter structure from a dictionary.

        Raises:
          Raises exception if there is a field in the dictionary that is
          not defined in the parameter structure.
        """
        our_keys = self._get_fields()
        for in_key, in_val in dict.items():
            if not in_key in our_keys:
                raise Exception(f'unknown field in input: {in_key}')

            our_val = self.__getattribute__(in_key)
            if isinstance(our_val, DictParamsMixin): 
                if in_val is not None:
                    our_val.from_dict(in_val)
            else:
                self.__setattr__(in_key, in_val)

    def update(self, key_path: str, value: typing.Any) -> None:
        """Update the params struct

        Supports deep indexing.
        Example:
          params.update('classifier/nof_iterations', 10)
        """
        dict = self.to_dict()
        update_dict(dict, key_path, value)
        self.from_dict(dict)
    
    def get(self, key_path: str, default=None) -> typing.Any:
        """Get a value with deep indexing

        If the value is not defined, the provided default value is returned.
        """
        dict = self.to_dict()
        return get_with_default(dict, key_path, default)

    def override(self, overrides: dict) -> 'DictParamsMixin':
        """Clones the dictparams object and overrides selecte parameters
        
        See override_baseline_dict above for more info (for now).
        """
        dict = self.to_dict()
        for (key_path, value) in overrides.items():
            update_dict(dict, key_path, value)
        
        # updated = self.__class__() # Note: copy.deepcopy(self) doesn't work!
        updated = copy.deepcopy(self)
        updated.from_dict(dict)
        return updated


# ------------------------------------------------------------------------------
# Logging

def get_default_logger(
        name: str, 
        path: str) -> logging.Logger:
    """Create a logger with our default style

    Args:
      name: Logger (and log file) name
      path: Directory where log files are stored (created if needed)
    """
    ensure_dir_exists(path)

    logging.addLevelName(logging.INFO, 'INF')
    logging.addLevelName(logging.WARNING, 'WRN')
    logging.addLevelName(logging.ERROR, 'ERR')

    logging.basicConfig(
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s', 
        datefmt="%Y-%m-%dT%H:%M:%S")

    logger = logging.getLogger('main')
    return logger


def get_null_logger() -> logging.Logger:
    """Return a logger that has no effect
    """
    logger = logging.getLogger("null_logger")
    logger.addHandler(logging.NullHandler())
    return logger


# ----------------------------------------------------------------------------
# Torch stuff

def default_device() -> torch.device:
    """Return the default device (CUDA if available, otherwise CPU)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_random_seed(seed: int) -> None:
    """Set random seed for python, numpy and torch.
    Call this to get deterministic results (good for debugging).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_chunkwise(
        model: torch.nn.Module, 
        inputs: torch.Tensor, 
        chunk_size: int) -> torch.Tensor:
    """Run a model on the input, chunk by chunk.
    
    This produces the exact same results as running output = model(inputs)
    directly, but saves memory by splitting the input into chunks and then 
    concatenating the results afterwards.

    Args:
      model: Model to run
      inputs: NCHW tensor of all inputs
      chunk_size: Number of images to use in each run

    Returns:
      Tensor with results
    """
    input_chunks = inputs.split(chunk_size)
    output_chunks = []

    for input_chunk in input_chunks:
        output_chunk = model(input_chunk)
        output_chunks.append(output_chunk)

    output = torch.cat(output_chunks)
    return output


# -----------------------------------------------------------------------------
# Image conversion

def resize_image(
        source: torch.Tensor, 
        size: tuple[float, float], 
        bilinear=False) -> torch.Tensor:
    """Convenience image resizing function
    
    Calls nn.functional.interpolate with options making it match PIL.resize for
    images. In contrast to F.interpolate, it also supports both NCHW and CHW 
    inputs.

    Args:
      source: NCHW or CHW float tensor
      size: Target size (height, width)
      linear: If true, uses bilinear interpolation. Default is False (matching
        PIL's resize)
    
    Returns:
      NCHW or CHW tensor (matching the source) with the new size.
    """

    if bilinear:
        mode = 'bilinear'
        antialias = True
        align_corners = False
    else:
        mode = 'nearest-exact'
        antialias = False
        align_corners = None

    if len(source.shape) == 4:
        # Already NCHW, ok
        return F.interpolate(
            source,
            size, 
            mode=mode, antialias=antialias, align_corners=align_corners)

    elif len(source.shape) == 3:
        # CHW, need to unsqueeze/squeeze
        output = F.interpolate(
            source.unsqueeze(0),
            size, 
            mode=mode, antialias=antialias, align_corners=align_corners)
        return output.squeeze(0)
    
    else:
        raise ValueError(f'unexpected shape: {source.shape}')


def save_tensor_image(image: torch.Tensor, path: str) -> None:
    """Save a tensor as an image file.
    If the tensor is 4-dimensional, save each slice as a separate file.
    """

    ensure_dir_exists(os.path.dirname(path))

    if len(image.shape) == 4 and image.shape[0] > 1:
        (base_path, ext) = os.path.splitext(path)
        for ix in range(image.shape[0]):
            save_tensor_image(image[ix], f'{base_path}_{ix}{ext}')

    else:
        save_image(image, path)


def load_tensor_image(path: str) -> torch.Tensor:
    """Load an image to a CHW tensor with values in the range [0..1] into
    the default device.
    """
    image = Image.open(path)
    tensor = transforms.ToTensor()(image)
    tensor = tensor.to(default_device())
    return tensor


def load_image_batch(path: str, names: list[str]) -> torch.Tensor:
    image_list = [load_tensor_image(os.path.join(path, n)) for n in names]
    images = torch.stack(image_list)
    return images


# ----------------------------------------------------------------------------
# Misc

_T = typing.TypeVar('_T')
def inverse_normalization(
        mean: _T,
        stdev: _T
        ) -> tuple[_T, _T]:
    """Compute inverse normalization values
    
    Given means/stdevs for normalizing a tensor (e.g. an image), produce the
    corresponding means/stdevs that inverts this normalization, i.e. such
    that Compose(Normalize(inv_mean, inv_stdev), Normalize(mean, stdev))
    is the identity transform.

    Args:
      mean: channel-wise means (e.g. RGB values)
      stdev: channel-wise standard deviations (e.g. RGB values)
    
    Returns:
      (inv_mean, inv_stdev) of the inverse transform.
    """

    # Inverse normalization:
    # b = (a - m) / s ==>
    # sb = a - m, a = sb + m = s * (b + m/s) = 
    # (b - (m/b)) / (1/s)

    if type(mean) == torch.Tensor:
        assert type(stdev) == torch.Tensor
        assert mean.shape == stdev.shape

        inv_mean = -mean/stdev
        inv_stdev = 1.0/stdev
        return (inv_mean, inv_stdev) # type:ignore

    elif type(mean) == list:
        # Expecting list of floats
        assert type(stdev) == list
        assert len(mean) == len(stdev)

        inv_mean = [-m/s for (m,s) in zip(mean,stdev)]
        inv_stdev = [1.0/s for s in stdev]

        return (inv_mean, inv_stdev) # type:ignore

    else:
        raise ValueError(f'input type unsupprted: {type(mean)}')


# ----------------------------------------------------------------------------
# Augmentation

def get_float_randaug(
        randaug_num: int,
        randaug_magn: int,
        mean: list[float]|torch.Tensor|None = None,
        stdev: list[float]|torch.Tensor|None = None):
    """Get a transform that applies RandAugment on a float image.

    The original implementation was for uint8 images. Sometimes, in practise,
    we may want to run it on images already converted to float, and perhaps
    also already normalized. This function returns a composite transform that
    first (optionally) undoes any normalization, then converts back to uint8,
    applies the RandAugment augmentation, and converts back to the original
    domain.

    Args:
      randaug_num: 'num' parameter of the RandAugment method
      randaug_magn: 'magn' parameter of the RandAugment method
      mean: Channel-wise mean values used when normalizing the input in the
        first place (or None if no normalization was made)
      stdev: Channel-wise standard deviations used when normalizing the input
        in the first place (or None if no normalization was made)
    
    Returns:
      Transform that applies the RandAugment on float inputs.
    """

    if mean != None:
        assert stdev != None

        inv_m, inv_s = inverse_normalization(mean, stdev)

        transf = transforms.Compose([
            transforms.Normalize(inv_m, inv_s),
            transforms.ConvertImageDtype(torch.uint8),
            transforms.RandAugment(randaug_num, randaug_magn),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, stdev),
        ])

        return transf
    
    else:
        assert stdev is None

        transf = transforms.Compose([
            transforms.ConvertImageDtype(torch.uint8),
            transforms.RandAugment(randaug_num, randaug_magn),
            transforms.ConvertImageDtype(torch.float),
        ])

        return transf
