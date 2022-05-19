import os
import torch
import torchvision.models as models

from cls.cls_nets.wrn_28_10 import WideResNet
from cls.cls_nets.net_models.at_cifar import ResNet50
from cls.cls_nets.net_models.model_zoo import WideResNet as WideResNet_gowal
from cls.cls_nets.net_models.trades import WideResNetTRADES
from cls.cls_nets.net_models.per import CifarResNetFeatureModel, AttackerModel
from cls.cls_nets.resnet_clean import resnet18

def load_cls_model(args, cls_trans):
    if args.arch in ['ddim', 'adp']:
        if args.cls_model in ['clean-wrn-28-10.pth']:
            cls_model = WideResNet(num_classes=10)
            model_path = os.path.join('cls', 'cls_models', args.cls_model)
            states = torch.load(model_path)
            if isinstance(states, list):
                cls_model.load_state_dict(states[0])
            else:
                cls_model.load_state_dict(states)
        elif args.cls_model in ['resnet18_clean.pt']:
            cls_model = resnet18()
            model_path = os.path.join('cls', 'cls_models', args.cls_model)
            states = torch.load(model_path)
            cls_model.load_state_dict(states)



    elif args.arch in ['at']:
        if args.cls_model not in ['l2']:
            cls_model = ResNet50(num_classes=10)
            model_path = os.path.join('cls', 'cls_models', 'models', 'at', 'l_inf', 'cifar10_rn50.pt')
            states = torch.load(model_path)
            cls_model.load_state_dict(states)
        else:
            cls_model = ResNet50(num_classes=10)
            model_path = os.path.join('cls', 'cls_models', 'models', 'at', 'l_2', 'cifar10_rn50.pt')
            states = torch.load(model_path)
            cls_model.load_state_dict(states)
    elif args.arch in ['per']:
        cls_model = ResNet50(num_classes=10)
        model_path = os.path.join('cls', 'cls_models', 'models', 'per', 'pat_alexnet_1.pt')
        states = torch.load(model_path)
        cls_model.load_state_dict(states['model'])
        cls_model = AttackerModel(cls_model)
        cls_model = CifarResNetFeatureModel(cls_model)

    elif args.arch in ['uncovering']:
        if args.cls_model not in ['l2']:
            cls_model = WideResNet_gowal(num_classes=10)
            model_path = os.path.join('cls', 'cls_models', 'models', 'uncovering', 'l_inf', 'cifar10_wrn28_10.pt')
            states = torch.load(model_path)
            cls_model.load_state_dict(states)
        else:
            cls_model = WideResNet_gowal(num_classes=10, depth=70, width=16)
            model_path = os.path.join('cls', 'cls_models', 'models', 'uncovering', 'l_2', 'cifar10_wrn70_16.pt')
            states = torch.load(model_path)
            cls_model.load_state_dict(states)
    elif args.arch in ['trades']:
        cls_model = WideResNetTRADES(num_classes=10)
        model_path = os.path.join('cls', 'cls_models', 'models', 'trades', 'l_inf', 'cifar10_wrn34_10.pt')
        states = torch.load(model_path)
        cls_model.load_state_dict(states)


    return cls_wrapper(cls_model, cls_trans, args)



class cls_wrapper(torch.nn.Module):
    def __init__(self, cls_model, cls_trans, args):
        super().__init__()
        self.cls_model = cls_model
        self.args = args
        self.cls_trans = cls_trans

    def forward(self, x):
        logits = self.cls_model(self.cls_trans(x))
        return logits

