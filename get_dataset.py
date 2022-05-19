import torch
import torchvision
from torchvision import transforms
import numpy as np



def get_dataset(args):
    if args.dataset in ['cifar10']:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()]))
        testloader = [torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)]
    elif args.dataset in ['cifar-c']:
        dataset_files = ['brightness.npy', 'contrast.npy', 'defocus_blur.npy', 'elastic_transform.npy',
                         'fog.npy', 'frost.npy', 'gaussian_blur.npy', 'gaussian_noise.npy', 'glass_blur.npy',
                         'impulse_noise.npy', 'jpeg_compression.npy', 'motion_blur.npy', 'pixelate.npy',
                         'saturate.npy', 'shot_noise.npy', 'snow.npy', 'spatter.npy', 'speckle_noise.npy',
                         'zoom_blur.npy']

        labels = torch.from_numpy(np.load('./data/CIFAR-10-C/labels.npy'))
        testloader = []
        for dataset_idx in range(len(dataset_files)):
            # construct ds
            imgs = torch.from_numpy(np.load('./data/CIFAR-10-C/' + dataset_files[dataset_idx]))
            imgs = imgs / 255.
            imgs = imgs.permute(0, 3, 1, 2)
            dataset = torch.utils.data.TensorDataset(imgs, labels)
            testloader.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True))

    return testloader


def get_classes(args):
    classes = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
    return classes

