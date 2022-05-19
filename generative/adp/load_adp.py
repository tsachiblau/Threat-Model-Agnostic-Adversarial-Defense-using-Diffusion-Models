import yaml
import os
import torch
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

from generative.adp.main import parse_args_and_config
from generative.adp.ncsnv2.runners.ncsn_runner import get_model
from generative.adp.purification.adp import adp


class RecursiveNamespace(SimpleNamespace):

  @staticmethod
  def map_entry(entry):
    if isinstance(entry, dict):
      return RecursiveNamespace(**entry)

    return entry

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    for key, val in kwargs.items():
      if type(val) == dict:
        setattr(self, key, RecursiveNamespace(**val))
      elif type(val) == list:
        setattr(self, key, list(map(self.map_entry, val)))


def load_adp(args_main):
    args, config = parse_args_and_config()

    with open(os.path.join('generative/adp/ncsnv2/configs', '{}.yml'.format(args_main.dataset)), 'r') as f:
        config_ebm = yaml.load(f, Loader=yaml.Loader)
        config_ebm = RecursiveNamespace(**config_ebm)
        config_ebm.device = config.device.ebm_device

    diff_model = get_model(config_ebm)
    states_ebm = torch.load(os.path.join('generative/adp', 'best_checkpoint_without_denoising.pth'))
    diff_model = torch.nn.DataParallel(diff_model)
    diff_model.load_state_dict(states_ebm[0], strict=True)
    diff_model = diff_model.module
    wraper = adp_wraper(diff_model, config, args, args_main)

    return wraper



class adp_wraper(torch.nn.Module):
    def __init__(self, diff_model, config, args, args_main):
        super().__init__()
        self.args_main = args_main
        self.args = args
        self.network_ebm = diff_model
        self.config = config
        self.config.purification.rand_type = "gaussian"


    def forward(self, x_adv, rep=1):
        rep_x_adv = x_adv.repeat([rep, 1, 1, 1])
        tensor_res = eval(self.config.purification.purify_method)(rep_x_adv, self.network_ebm,
                                                                                  self.config.purification.max_iter,
                                                                                  mode="purification",
                                                                                  config=self.config)[0][-1]
        tensor_res = tensor_res.reshape(-1, x_adv.shape[0], x_adv.shape[1], x_adv.shape[2], x_adv.shape[3]).mean(0)

        return tensor_res


        '''
        import matplotlib
        matplotlib.use('TKAgg')
        img = tensor_res[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()
        plt.figure()
        plt.imshow(img)
        plt.show()
        '''

