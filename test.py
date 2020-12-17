from utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X, y_vec = load_mnist()
    x = X[0, :, :, :]

    x = x * 255

    plt.imshow(x, cmap='gray')
    plt.show()

