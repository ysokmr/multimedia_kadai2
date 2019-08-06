from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def make_image(size=(28, 28), cls=0):
    img = np.array(Image.new('RGB', size, (0, 0, 0)))
    if cls == 0:
        w = random.randrange(1, size[1]//2)
        img[:, :w] = np.random.randint(0, 256, (size[0], w, 3))
    elif cls == 1:
        w = random.randrange(1, size[0]//2)
        img[:w] = np.random.randint(0, 256, (w, size[1], 3))
    elif cls == 2:
        w = random.randrange(1, size[1]//2)
        img[:, -w:] = np.random.randint(0, 256, (size[0], w, 3))
    elif cls == 3:
        w = random.randrange(1, size[0]//2)
        img[-w:] = np.random.randint(0, 256, (w, size[1], 3))
    return img

def make_image_set(num, size=(28, 28)):
    imset = []
    for i in range(num):
        c = random.randint(0, 3)
        im = make_image(size, c)
        imset.append((im, c))
    return imset



if __name__ == '__main__':
    plt.subplot(2, 2, 1)
    plt.imshow(make_image(cls=0))
    plt.subplot(2, 2, 2)
    plt.imshow(make_image(cls=1))
    plt.subplot(2, 2, 3)
    plt.imshow(make_image(cls=2))
    plt.subplot(2, 2, 4)
    plt.imshow(make_image(cls=3))
    
    plt.show()
