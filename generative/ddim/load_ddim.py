import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from functools import partial

from generative.ddim.runners.diffusion import Diffusion
from generative.ddim.runners.diffusion import Model
from generative.ddim.functions.ckpt_util import get_ckpt_path
from generative.ddim.datasets import data_transform, inverse_data_transform




torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default='generative/ddim/configs/cifar10.yml', help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default='exp',
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        default=True,
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=20, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--adv_threat_model", type=str, default='None', help="")
    parser.add_argument("--batch_size", type=str, default='None', help="")
    parser.add_argument("--debug", type=str, default='None', help="")
    parser.add_argument("--first_step", type=str, default='None', help="")
    parser.add_argument("--show_plots", type=str, default='None', help="")
    parser.add_argument("--adv_EOT", type=int, default=20, help="")
    parser.add_argument("--cls_model", type=str, default='clean-wrn-28-10.pth', help="")
    parser.add_argument("--arch", type=str, default='ddim', help="")
    parser.add_argument("--adv_attack_type", type=str, default='ddim', help="")
    parser.add_argument("--adv_epsilon", type=str, default='ddim', help="")
    parser.add_argument("--adv_step_size", type=str, default='ddim', help="")
    parser.add_argument("--dataset", type=str, default='cifar10', help="")


    args = parser.parse_args()
    args.log_path = os.path.join('generative', 'ddim', args.exp, "logs", args.doc)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True
                overwrite = True
                # if overwrite:
                #     shutil.rmtree(args.log_path)
                #     shutil.rmtree(tb_path)
                #     os.makedirs(args.log_path)
                #     if os.path.exists(tb_path):
                #         shutil.rmtree(tb_path)
                # else:
                #     print("Folder exists. Program halted.")
                #     sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def add_diffusion_noise(x, num_of_steps, betas, config):
    n = x.size(0)
    t = (torch.ones(n) * num_of_steps).cuda()
    a = (1 - betas).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1)

    x_trans = data_transform(config, x)
    # get the same noise as the forward
    z_noise = torch.randn_like(x)
    x2 = x_trans * a.sqrt() + z_noise * (1.0 - a).sqrt()

    return x2


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas



def sample_image(x, model, betas, first_step, args, config, last=True):
    try:
        skip = args.skip
    except Exception:
        skip = 1

    num_timesteps = betas.shape[0]

    if args.sample_type == "generalized":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = list(range(0, first_step, skip))
            if first_step not in seq:
                seq += [first_step]

        elif args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from generative.ddim.functions.denoising import generalized_steps

        xs = generalized_steps(x, seq, model, betas, eta=args.eta)
        x = xs
    else:
        raise NotImplementedError

    '''
    x_save = [inverse_data_transform(config, x_i) for x_i in x[0]]
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TKAgg')

    for i in range(len(x_save)-2):
        plt.figure(figsize=(10, 10))
        plt.imshow(x_save[i].detach().cpu().squeeze().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()
        plt.savefig(f'images/2_{i+1}')
        plt.close()
    '''

    if last:
        x = x[0][-1]


    return x


def load_ddim(args_main):
    args, config = parse_args_and_config()
    diff_model = Model(config)
    diff_model.load_state_dict(torch.load(f"generative/ddim/model-790000.ckpt"))

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float()
    wraper = ddim_wraper(diff_model, betas, config, args, args_main)
    return wraper



class ddim_wraper(torch.nn.Module):
    def __init__(self, diff_model, betas, config, args, args_main):
        super().__init__()
        self.diff_model = diff_model
        self.betas = betas
        self.config = config
        self.args = args
        self.args_main = args_main


    def forward(self, x_attacked, rep=1):
        if x_attacked.device != self.betas.device:
            self.betas = self.betas.cuda()

        x2 = add_diffusion_noise(x_attacked, self.args_main.first_step, self.betas, self.config)
        x3 = sample_image(x2, self.diff_model, self.betas, self.args_main.first_step, self.args, self.config, last=True)
        x4 = inverse_data_transform(self.config, x3)

        return x4



