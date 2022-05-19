from torchvision import transforms


def get_cls_transform(args):
    if args.arch in ['ddim', 'adp']:
        if args.cls_model in ['clean-wrn-28-10.pth']:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
        elif args.cls_model in ['resnet18_clean.pt']:
            norm_mean = [0.4914, 0.4822, 0.4465]
            norm_std = [0.2471, 0.2435, 0.2616]
        trans = transforms.Compose([transforms.Normalize(norm_mean, norm_std)])
    elif args.arch in ['at', 'per']:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        trans = transforms.Compose([transforms.Normalize(norm_mean, norm_std)])
    elif args.arch in ['uncovering']:
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2471, 0.2435, 0.2616]
        trans = transforms.Compose([transforms.Normalize(norm_mean, norm_std)])
    elif args.arch in ['trades']:
        norm_mean = [0., 0., 0.]
        norm_std = [1., 1., 1.]
        trans = transforms.Compose([transforms.Normalize(norm_mean, norm_std)])

    return trans

