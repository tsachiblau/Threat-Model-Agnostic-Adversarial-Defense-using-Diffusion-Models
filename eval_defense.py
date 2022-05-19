import torch
from torch import nn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from get_dataset import get_dataset
from analysis import plot_attack


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

def white_l2(generative_model, model, args, x, y):
    num_iter = args.adv_num_steps
    delta = torch.zeros_like(x, requires_grad=True)
    for t in range(num_iter):
        diff_output = generative_model(x + delta, 10)
        loss = nn.CrossEntropyLoss()(model(diff_output), y)
        loss.backward()
        delta.data += args.adv_step_size * delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
        delta.data *= args.adv_epsilon / norms(delta.detach()).clamp(min=args.adv_epsilon)
        delta.grad.zero_()

    return delta.detach()


def white_EOT_l2(generative_model, cls_model, args, x, y):
    x_repeat = x.repeat([args.adv_EOT, 1, 1, 1]).cuda()
    y_repeat = y.repeat([args.adv_EOT]).cuda()
    delta = torch.zeros_like(x).cuda()

    for iter in range(args.adv_num_steps):
        delta_repeat = delta.repeat([args.adv_EOT, 1, 1, 1]).cuda()
        delta_repeat.requires_grad = True
        diff_output = generative_model(x_repeat + delta_repeat, 10)
        cls_output = cls_model(diff_output)
        loss = nn.CrossEntropyLoss()(cls_output, y_repeat)
        loss.backward()
        delta_mean = delta_repeat.grad.reshape(args.adv_EOT, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).mean(0).detach()
        delta.data += args.adv_step_size * delta_mean / norms(delta_mean)
        delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
        delta.data *= args.adv_epsilon / norms(delta.detach()).clamp(min=args.adv_epsilon)


    return delta




def no_attack(x, y):
    delta = torch.zeros_like(x, device='cuda')
    return delta


def grey(cls_model, args, x, y):
    delta = torch.zeros_like(x, requires_grad=True, device='cuda')

    for iter in range(args.adv_num_steps):
        cls_output = cls_model(x + delta)
        loss = nn.CrossEntropyLoss()(cls_output, y)
        loss.backward()
        delta.data = (delta.data + args.adv_step_size * delta.grad.detach().sign()).clamp(-args.adv_epsilon, args.adv_epsilon)
        delta.data += torch.clip(x + delta, 0, 1) - x - delta
        delta.grad.zero_()

    return delta.detach()



def BPDA_EOT(generative_model, cls_model, args, x, y):
    x_repeat = x.repeat([args.adv_EOT, 1, 1, 1])
    y_repeat = y.repeat([args.adv_EOT])
    delta = torch.zeros_like(x).cuda()

    for iter in range(args.adv_num_steps):
        delta_repeat = delta.repeat([args.adv_EOT, 1, 1, 1]).cuda()
        delta_repeat.requires_grad = True
        with torch.no_grad():
            diff_output = generative_model(x_repeat + delta_repeat, 10)
        diff_output.requires_grad = True
        cls_output = cls_model(diff_output)
        loss = nn.CrossEntropyLoss()(cls_output, y_repeat)
        classifier_grad = torch.autograd.grad(loss, diff_output)[0]
        delta_mean = classifier_grad.reshape(-1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).mean(0)
        delta.data = (delta.data + args.adv_step_size * delta_mean.sign()).clamp(-args.adv_epsilon, args.adv_epsilon)
        delta.data += torch.clip(x + delta, 0, 1) - x - delta

    return delta



def white(generative_model, cls_model, args, x, y):
    delta = torch.zeros_like(x, requires_grad=True, device='cuda')

    for iter in range(args.adv_num_steps):
        diff_output = generative_model(x + delta, 10)
        cls_output = cls_model(diff_output)
        loss = nn.CrossEntropyLoss()(cls_output, y)
        loss.backward()
        delta.data = (delta + args.adv_step_size * delta.grad.detach().sign()).clamp(-args.adv_epsilon, args.adv_epsilon)
        delta.data += torch.clip(x + delta, 0, 1) - x - delta
        delta.grad.zero_()

    return delta.detach()
    '''
    import matplotlib
    matplotlib.use('TKAgg')
    img = delta[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img -= np.min(img)
    img /= np.max(img)
    plt.figure()
    plt.imshow(img)
    plt.show()
    '''


def white_EOT(generative_model, cls_model, args, x, y):
    x_repeat = x.repeat([args.adv_EOT, 1, 1, 1]).cuda()
    y_repeat = y.repeat([args.adv_EOT]).cuda()
    delta = torch.zeros_like(x).cuda()

    for iter in range(args.adv_num_steps):
        delta_repeat = delta.repeat([args.adv_EOT, 1, 1, 1]).cuda()
        delta_repeat.requires_grad = True
        diff_output = generative_model(x_repeat + delta_repeat, 10)
        cls_output = cls_model(diff_output)
        loss = nn.CrossEntropyLoss()(cls_output, y_repeat)
        loss.backward()
        delta_mean = delta_repeat.grad.reshape(args.adv_EOT, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).mean(0)
        delta.data = (delta.data + args.adv_step_size * delta_mean.sign()).clamp(-args.adv_epsilon, args.adv_epsilon)
        delta.data += torch.clip(x + delta, 0, 1) - x - delta

    return delta



def get_threat_model(generative_model, cls_model, args):
    if args.adv_threat_model in ['l2']:
        if args.adv_attack_type in ['no_attack']:
            threat_model = partial(no_attack)
        elif args.adv_attack_type in ['white']:
            threat_model = partial(white_l2, generative_model, cls_model, args)
        elif args.adv_attack_type in ['white_EOT']:
            threat_model = partial(white_EOT_l2, generative_model, cls_model, args)
        else:
            raise Exception('adv_threat_model doesnt exists')

    elif args.adv_threat_model in ['linf']:
        if args.adv_attack_type in ['no_attack']:
            threat_model = partial(no_attack)
        elif args.adv_attack_type in ['grey']:
            threat_model = partial(grey, cls_model, args)
        elif args.adv_attack_type in ['BPDA_EOT']:
            threat_model = partial(BPDA_EOT, generative_model, cls_model, args)
        elif args.adv_attack_type in ['white']:
            threat_model = partial(white, generative_model, cls_model, args)
        elif args.adv_attack_type in ['white_EOT']:
            threat_model = partial(white_EOT, generative_model, cls_model, args)
        else:
            raise Exception('adv_threat_model doesnt exists')

    return threat_model



def eval_defense(generative_model, cls_model, args):
    cls_model.eval()
    threat_model = get_threat_model(generative_model, cls_model, args)
    test_dataset = get_dataset(args)


    list_acc = []
    for test_dataset_i in test_dataset:
        list_clean_correct = []
        list_attack_correct = []
        list_restored_correct = []
        list_MSE = []

        sample_num = 1
        for batch_num, (x, y) in enumerate(test_dataset_i):
            x, y = x.cuda(), y.cuda()
            delta = threat_model(x, y)

            with torch.no_grad():
                bool_clean = cls_model(x).max(dim=1)[1].eq(y)
                bool_attack = cls_model(x + delta).max(dim=1)[1].eq(y)

            #restore x with generative
            with torch.no_grad():
                x_restored = generative_model(x + delta, 10)
                cls_output = cls_model(x_restored)
                bool_restored = cls_output.max(dim=1)[1].eq(y)
                y_hat_restored = cls_output.max(dim=1)[1]


            # plt.figure(figsize=(10, 10))
            # plt.imshow((x).detach().cpu().squeeze().permute(1, 2, 0).numpy())
            # plt.axis('off')
            # plt.show()
            # plt.savefig(f'images_diff/{batch_num}_clean')
            # plt.close()
            #
            #
            # plt.figure(figsize=(10, 10))
            # img = (delta).detach().cpu().squeeze().permute(1, 2, 0).numpy()
            # img = (img - img.min())/img.max()
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
            # plt.savefig(f'images_diff/{batch_num}_delta')
            # plt.close()


            '''
            plt.figure(figsize=(10, 10))
            img = (delta).detach().cpu().squeeze().permute(1, 2, 0).numpy()
            img = (img - img.min())/img.max()
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            plt.savefig(f'images/2_delta')
            plt.close()


            plt.figure(figsize=(10, 10))
            plt.imshow((x).detach().cpu().squeeze().permute(1, 2, 0).numpy())
            plt.axis('off')
            plt.show()
            plt.savefig(f'images/2_clean')
            plt.close()


            plt.figure(figsize=(10, 10))
            plt.imshow((x + delta).detach().cpu().squeeze().permute(1, 2, 0).numpy())
            plt.axis('off')
            plt.show()
            plt.savefig(f'images/2_noisy')
            plt.close()
            '''

            if args.show_plots == True:
                plot_attack(x, y, x_restored, y_hat_restored, delta, sample_num, args)

            for i in range(len(bool_clean)):
                list_clean_correct.append(bool_clean[i].item())
                list_attack_correct.append(bool_attack[i].item())
                list_restored_correct.append(bool_restored[i].item())
                list_MSE.append(torch.norm(x[i].detach() - x_restored[i].detach()).item())
                # calc all acc
                orig_acc = np.sum(list_clean_correct) / len(list_clean_correct)
                attacked_acc = np.sum(list_attack_correct) / len(list_clean_correct)
                restored_acc = np.sum(list_restored_correct) / len(list_clean_correct)
                avg_mse = np.sum(list_MSE) / len(list_clean_correct)

                #print to log
                print('sample num: {}    original acc: {}  attacked acc: {}    restored acc: {}    avg mse: {}'.format(sample_num, orig_acc, attacked_acc, restored_acc, avg_mse))
                sample_num += 1


            if sample_num > 2 and args.sanity_check:
                break

        list_acc.append(restored_acc)
        print(list_acc)

