import argparse
import sys
import os
import torch
import numpy as np
from datetime import datetime
import random

from cls.load_cls_model import load_cls_model
from generative.load_generative_model import load_generative_model
from eval_defense import eval_defense
from cls.get_cls_transform import get_cls_transform




class Logger(object):
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='eval_generative_defense')
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--debug", type=bool, default=False, help="")
    parser.add_argument("--log_path", type=str, default='logs', help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--sanity_check", type=bool, default=False, help="")
    parser.add_argument("--show_plots", type=bool, default=False, help="")
    parser.add_argument("--dataset", type=str, default='cifar10', help="")
    parser.add_argument("--cls_model", type=str, default='clean-wrn-28-10.pth', help="")

    ## generative params ##
    parser.add_argument("--first_step", type=int, default=140, help="")
    parser.add_argument("--arch", type=str, default='ddim', help="")
    parser.add_argument("--timesteps", type=int, default=100, help="")

    ## adv params ##
    parser.add_argument("--adv_num_steps", type=int, default=20, help="")
    parser.add_argument("--adv_epsilon", type=float, default=0.03137254, help="")
    parser.add_argument("--adv_step_size", type=float, default=0.00392156, help="")
    parser.add_argument("--adv_attack_type", type=str, default='white', help="")
    parser.add_argument("--adv_threat_model", type=str, default='linf', help="")
    parser.add_argument("--adv_EOT", type=int, default=20, help="")


    args = parser.parse_args()
    args.adv_step_size = 2.5 * (args.adv_epsilon / args.adv_num_steps)
    return args


def main():
    args = parse_args()
    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #set logger
    now = datetime.now()
    str_date_time = now.strftime("%d-%m-%Y_%H:%M")
    log_file_name = 'arch_{}_threat_{}_type_{}_time_{}'.format(args.arch, args.adv_threat_model, args.adv_attack_type, str_date_time)
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    sys.stdout = Logger(os.path.join('logs', log_file_name + '.txt'))

    print('#' * 50)
    print(args)
    print('#' * 50)

    generative_model = load_generative_model(args)
    generative_model = torch.nn.DataParallel(generative_model).cuda().eval()

    cls_tansform = get_cls_transform(args)
    cls_model = load_cls_model(args, cls_tansform)
    cls_model = torch.nn.DataParallel(cls_model).cuda().eval()

    eval_defense(generative_model, cls_model, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
