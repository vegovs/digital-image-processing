from imageio import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import signal


def _mean_filter(n):
    """
    Produces a kernel for average- or mean filtering.

    :param n: int
        Size of kernel.
    :return:
        n-sized average filter
    """
    return np.full((n, n), 1) / n


def mean_convolution(image, n):
    """
    Performs a mean-filtering using convolution

    :param image:
        Image to be filtered
    :param n:
        Size of mean-kernel
    :return:
        Filtered image
    """
    kernel = _mean_filter(n)
    return signal.convolve2d(image, kernel)


def mean_frequency_domain(image, n):
    """
    Performs a mean-filtering using frequency domain

    :param image:
        Image to be filtered
    :param n:
        Size of mean-kernel
    :return:
        Filtered image
    """
    kernel = _mean_filter(n)

    x1 = x2 = (image.shape[0] - n) // 2
    y1 = y2 = (image.shape[1] - n) // 2

    if n % 2 == 0:
        if (image.shape[0]) % 2 == 1:
            x2 = x2 + 1

        if (image.shape[1]) % 2 == 1:
            y2 = y2 + 1

    if (n % 2) == 1:
        if (image.shape[0]) % 2 == 0:
            x2 = x2 + 1

        if (image.shape[1]) % 2 == 0:
            y2 = y2 + 1

    # Fourier transform
    fourier = np.fft.fft2(image)

    # Fourier transform for filter
    f_filter = np.fft.fft2(kernel, image.shape)

    # Multiply the filter with the image
    fourier = f_filter * fourier

    # Reverse fourier
    fourier = np.fft.ifft2(fourier)

    fourier = np.real(fourier)
    return fourier


def time_convolve(image, filter_sizes):
    """
    Performs timed mean-filtering using convolution
    :param image:
        Image to be filtered
    :param filter_sizes: int
        Kernel sizes
    :return: list
        Time
    """
    start = time.time()
    times = []
    for filter_sizes in filter_sizes:
        start = time.time()
        mean_convolution(image, filter_sizes)
        times.append((time.time() - start))

    return times


def time_freq(image, filter_sizes):
    """
    Performs timed mean-filtering using frequency domain
    :param image:
        Image to be filtered
    :param filter_sizes: int
        Kernel sizes
    :return: list
        Time
    """
    start = time.time()
    times = []
    for filter_sizes in filter_sizes:
        start = time.time()
        mean_frequency_domain(image, filter_sizes)
        times.append((time.time() - start))

    return times


if __name__ == "__main__":
    img = imread('cow.png', as_gray=True)

    mc = mean_convolution(img, 15)
    mf = mean_frequency_domain(img, 15)

    plot, (plot_img, plot_conv, plot_fouri) = plt.subplots(1, 3)

    plot_conv.imshow(mc, cmap='gray')
    plot_conv.set_axis_off()
    plot_conv.set_title('Romlig')

    plot_img.imshow(img, cmap='gray')
    plot_img.set_axis_off()
    plot_img.set_title('Orginal')

    plot_fouri.imshow(mf, cmap='gray')
    plot_fouri.set_axis_off()
    plot_fouri.set_title('Fourier')

    plt.savefig("1_1.png", dpi=300)
    plt.show()
    plt.close()

    kernels_sizes = (3, 5, 6, 9, 15, 19, 25, 30)

    conv2d = time_convolve(img, kernels_sizes)
    freq_d = time_freq(img, kernels_sizes)

    print("Convolve: ")
    print(conv2d)

    plt.plot(kernels_sizes, conv2d, color='blue', label='Convolve')
    plt.plot(kernels_sizes, freq_d, color='red', label='Freq Domain')
    plt.xlabel('Size of mean filter')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.savefig("1_2.png")
    plt.show()
    plt.close()
