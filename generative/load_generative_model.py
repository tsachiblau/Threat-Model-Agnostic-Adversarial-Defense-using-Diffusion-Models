import torch
from generative.ddim.load_ddim import load_ddim
from generative.adp.load_adp import load_adp



def load_generative_model(args):
    if args.arch in ['ddim']:
        generative_model = load_ddim(args)
    if args.arch in ['adp']:
        generative_model = load_adp(args)
    if args.arch in ['at', 'uncovering', 'trades', 'per']:
        generative_model = empty_wraper(args)

    return generative_model


class empty_wraper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = None

    def forward(self, x_adv, rep=1):
        return x_adv

