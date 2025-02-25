import torch

from .diffmodel import DiffModel
from . import textinv, nulltextinv


"""Common data class for storing various adaptations of a diffusion model.

The idea is to have a single file format for such adaptations, such that we
can easily create an adaped diffusion model by loading such an adaptation file.
Right now, the class supports textual inversion and null-text inversion, but
this could be extended to low-rank adaptations and similar.

The data to save is organized in a dictionary that is saved with torch.save.

File contents example:

  file_data = {
    format: 'pixw.se/format:diffusion-model-patch:dev.1',
    base_model: {
      name: 'runwayml/stable-diffusion-v1-5',
      cf_guidance: 7.5,
      nof_iterations: 25,
      width: 512,
      height: 512
    }
    textual_inversion: 
      token_config: [
        {placeholder: '#brown', type: 'shared', init: 'brown'},
        {placeholder: '#cup', type: 'shared', init: 'cup'}
      ], 
      embeddings: {
        #brown: <tensor>,
        #cup: <tensor>,
      }
    }
    nulltext_inversion: [
      {
        input_file_name: 'mycup1.jpg',
        latents: <tensor>,
        prompt: 'A photo of a #brown #cup',
        unc_embeddings: [<tensor0>, ..., <tensor24>]
      },
      {
        input_file_name: 'mycup2.jpg', 
        ...
      },
      ...
    ]
  }
"""

class DiffModelPatch:
    """Class representing various adjustments made to a diffusion model.

    Intended for saving and loading such adjustments.    

    Supports textual inversion information (adjusted embeddings of selected
    tokens) and null-text inversion (adjusted embeddings of the null-text
    tokens). In the future, we could add various low-rank adaptations etc.

    Multiple null text inversions are supported. This is since the textinv
    can be computed using several images. Then we can store one nulltext-inv
    per input image for the textinv.
    """

    def __init__(self):
        self.data = {}
        self.data['format'] = 'pixw.se/format:diffusion-model-patch:dev.1'
        self.data['textual_inversion'] = None
        self.data['nulltext_inversion'] = []

    def set_base_model_data(self, diffmodel: DiffModel):
        self.data['base_model'] = {
            'name': diffmodel.model_name,
            'cf_guidance': diffmodel.cf_guidance,
            'nof_iterations': diffmodel.nof_iterations,
            'width': diffmodel.width, 
            'height': diffmodel.height
        }

    def set_textinv_data(
            self,
            token_config: list[textinv.TokenConfig],
            embeddings: dict):

        self.data['textual_inversion'] = {}
        self.data['textual_inversion']['token_config'] = [
            vars(tc) for tc in token_config]

        self.data['textual_inversion']['embeddings'] = embeddings

    def get_textinv_embeddings(self) -> dict:
        return self.data['textual_inversion']['embeddings']

    def append_nulltext_data(self, nti_data: nulltextinv.NullInversionData):
        nti_dict = {
            'input_file_name': nti_data.input_file_name,
            'latents': nti_data.latents,
            'prompt': nti_data.prompt,
            'unc_embeddings': nti_data.unc_embeddings
        }
        self.data['nulltext_inversion'].append(nti_dict)

    def get_nulltext_data(self) -> list[nulltextinv.NullInversionData]:
        output = []
        for d in self.data['nulltext_inversion']:
            nti_data = nulltextinv.NullInversionData()
            nti_data.latents = d['latents']
            nti_data.unc_embeddings = d['unc_embeddings']
            nti_data.input_file_name = d['input_file_name']
            nti_data.prompt = d['prompt']
            output.append(nti_data)

        return output

    def save(self, file_name: str):
        torch.save(self.data, file_name)
        
    def load(self, file_name: str):
        self.data = torch.load(file_name)
