import numpy as np
import matplotlib.pyplot as plt
import os
from get_dataset import get_classes


def plot_attack(x, y, x_restored, y_hat, delta, img_num, args):
    if len(x) > 1:
        x, y, x_restored, y_hat, delta = x[0].unsqueeze(0), y[0], x_restored[0].unsqueeze(0), y_hat[0], delta[0].unsqueeze(0)

    classes = get_classes(args)
    fig = plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(x.detach().squeeze().cpu().permute(1, 2, 0).numpy())
    plt.title('original img \nclass: {}'.format(classes[y]))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    img = delta.detach().squeeze().cpu().permute(1, 2, 0).numpy()
    img -= np.min(img)
    img /= np.max(img)
    plt.imshow(img)
    plt.title('delta')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow((x+delta).detach().squeeze().cpu().permute(1, 2, 0).numpy())
    plt.title('x+delta')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow((x_restored).detach().squeeze().cpu().permute(1, 2, 0).numpy())
    plt.title('x restored \nclass: {}'.format(classes[y_hat]))
    plt.axis('off')

    fig.tight_layout()
    plt.show()

    if os.path.isdir('images') == False:
        os.mkdir('images')
    img_path = os.path.join('images', str(img_num))

    fig.savefig(img_path)
