from math import cos, pi, log2
import numpy as np
from imageio import imread, imsave
from numba import jit


def lossy_jpeg_compression(img_name, q):
    """
    A crude lossy-compression method
    :param img_name:
        Image to be compressed
    :param q:
        Compression scalar
    :return:
    """
    image = imread(img_name)
    image = np.subtract(image, 128.0)
    dct_img = dct_2d(image, q, decompress=False)
    H, CR = entropy_and_compression_rate(dct_img)
    return dct_img, H, CR


def lossy_jpeg_decompression(img, q):
    """
    An equally crude decompression method
    :param q:
        Compression scalar
    :param img:
        Image to be decompressed
    :return:
    """
    dctd_img = dct_2d(img, q, decompress=True)
    dctd_img = np.add(dctd_img, 128.0)
    return dctd_img


def dct_2d(image, q, decompress=False):
    """
    Splits the image into 8x8 blocks and performs a discrete cosine transformation
    :param q:
        Scalar for the quantification-matrix Q
    :param decompress: bool
        Run reverse DCT or not
    :param image:
        Image to be transformed
    :return:
        Transformed image
    """
    img_out = np.zeros(image.shape)
    Q = np.loadtxt('Q.txt', usecols=range(8))
    Q = np.flip(Q, axis=0)
    qQ = q * Q

    for i in np.r_[:image.shape[0]:8]:
        for j in np.r_[:image.shape[1]:8]:
            block = image[i: (i + 8), j:(j + 8)]
            if decompress is False:
                if q > 0:
                    img_out[i: (i + 8), j: (j + 8)] = np.round(_dct(block) / qQ, 0)
                else:
                    img_out[i: (i + 8), j: (j + 8)] = _dct(block)
            else:
                if q > 0:
                    img_out[i: (i + 8), j: (j + 8)] = np.round(_idct(np.round(block * qQ, 0)),0)
                else:
                    img_out[i: (i + 8), j: (j + 8)] = np.round(_idct(block), 0)

    return img_out


@jit(nopython=True)
def _dct(f):
    """
    Performs a simplified version of DCT based on 8x8 blocks
    :param f: 2d array
        8x8 block
    :return:
        Transformed block
    """
    F = np.zeros(f.shape)

    for u in range(f.shape[0]):
        for v in range(f.shape[1]):
            pre_sum = (1 / 4) * _c(u) * _c(v)
            sum = 0

            for x in range(f.shape[0]):
                for y in range(f.shape[1]):
                    sum = sum + f[x, y] * cos(((2 * x + 1) * u * pi) / 16) * cos(((2 * y + 1) * v * pi) / 16)
            F[u, v] = pre_sum * sum

    return F


@jit(nopython=True)
def _idct(F):
    """
    Performs a simplified version of DCT-inverse based on 8x8 blocks
    :param F: 2d array
        8x8 block
    :return:
        Transformed block
    """
    f = np.zeros(F.shape)

    for x in range(F.shape[0]):
        for y in range(F.shape[1]):
            pre_sum = (1 / 4)
            sum = 0

            for u in range(F.shape[0]):
                for v in range(F.shape[1]):
                    sum = sum + _c(u) * _c(v) * F[u, v] * cos(((2 * x + 1) * u * pi) / 16) \
                          * cos(((2 * y + 1) * v * pi) / 16)
            f[x, y] = pre_sum * sum

    return f


@jit(nopython=True)
def _c(a):
    """
    Helper function
    :param a:
    :return:
    """
    return (1 / np.sqrt(2)) if a == 0 else 1


def entropy_and_compression_rate(image):
    """
    Method for calculating entropy and compression rate
    :param image:
        Image
    :return: tuple
        H, CR
    """
    # Entropy
    low, high = np.floor(image.min()), np.ceil(image.max())
    bins = np.arange(low, high, 1)
    hist, bins = np.histogram(image, density=True, bins=bins)
    H = np.sum([-(p * log2(p)) if p > 0 else 0 for p in hist])
    # Compression rate(c=h in this task)
    CR = 8 / H
    return H, CR


if __name__ == '__main__':

    # Test decompression
    compressed_image, H, CR = lossy_jpeg_compression('uio.png', 0)
    decompressed_image = lossy_jpeg_decompression(compressed_image, 0)
    image = imread('uio.png', as_gray=True)
    assert (np.equal(image, decompressed_image).all()), "Decompression failed!"
    print("Decompression successful = ", np.equal(image, decompressed_image).all())

    # Test different scalars of q
    i = 0
    for _q in np.array([0.1, 0.5, 2, 8, 32]):
        i += 1
        compressed_image, H, CR = lossy_jpeg_compression('uio.png', _q)
        decompressed_image = lossy_jpeg_decompression(compressed_image, _q)
        img_name = "2_%d.png" % i
        imsave(img_name, decompressed_image)
        print("H = ", H)
        print("CR = ", CR)
