import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torchvision
import numpy as np


def show_image(image):
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show(block=False)



def show_images_grid(images):
    show_image(torchvision.utils.make_grid(images, padding=2))
    #plt.show()

