from math import floor, ceil

import numpy as np
from imageio import imread, imsave
import matplotlib.pyplot as plt


def transformation_fw(img_in, a, b):
    """
    Performs affine transformation using forward mapping
    Args:
        img_in: Image to be transformed
        a: x-coefficients
        b: y-coefficients

    Returns:
        img_out: Transformed image

    """
    rows, cols = img_in.shape
    img_out = np.zeros((600, 512), dtype='uint8')

    _rows, _cols = img_out.shape
    for y in range(rows):
        for x in range(cols):
            _y, _x = int(np.around(a[0] * y + a[1] * x + a[2])), \
                     int(np.around(b[0] * y + b[1] * x + b[2]))
            if _y in range(0, _rows) and _x in range(0, _cols):
                img_out[_y, _x] = img_in[y, x]

    return img_out


def transformation_cni(img_in, a, b):
    """
    Performs affine transformation using closest neighbour interpolation
    Args:
        img_in: Image to be transformed
        a: x-coefficients
        b: y-coefficients

    Returns:
        img_out: Transformed image

    """
    rows, cols = img_in.shape
    img_out = np.zeros((600, 512), dtype='uint8')
    # Get inverse of the transformation matrix
    t = np.row_stack((a, b, np.array((0, 0, 1))))
    ti = np.linalg.inv(t)
    a = ti[0]
    b = ti[1]

    _rows, _cols = img_out.shape
    for _y in range(_rows):
        for _x in range(_cols):
            y, x = int(np.around(a[0] * _y + a[1] * _x + a[2])), \
                   int(np.around(b[0] * _y + b[1] * _x + b[2]))
            if y in range(0, rows) and x in range(0, cols):
                img_out[_y, _x] = img_in[y, x]

    return img_out


def transformation_bli(img_in, a, b):
    """
    Performs affine transformation using bilinear interpolation
    Args:
        img_in: Image to be transformed
        a: x-coefficients
        b: y-coefficients

    Returns:
        img_out: Transformed image

    """
    rows, cols = img_in.shape
    img_out = np.zeros((600, 512), dtype='uint8')
    # Get inverse of the transformation matrix
    t = np.row_stack((a, b, np.array((0, 0, 1))))
    ti = np.linalg.inv(t)
    a = ti[0]
    b = ti[1]

    _rows, _cols = img_out.shape
    for _y in range(_rows):
        for _x in range(_cols):
            y, x = (a[0] * _y + a[1] * _x + a[2]), \
                   (b[0] * _y + b[1] * _x + b[2])

            # Algorithm
            y_0, x_0, y_1, x_1 = floor(y), \
                                 floor(x), \
                                 ceil(y), \
                                 ceil(x)
            d_y, d_x = y - y_0, x - x_0
            p = img_in[y_0, x_0] + (img_in[y_1, x_0] - img_in[y_0, x_1]) * d_y
            q = img_in[y_0, x_1] + (img_in[y_1, x_1] - img_in[y_0, x_1]) * d_y
            img_out[_y, _x] = np.clip(p + (q - p) * d_x, 0, 255)

    return img_out


def std_contrast(image):
    """
    Standardizes contrast using linear greyscale transformation
    Args:
        image: Image to be standardized

    Returns:
        img_out: Standardized image

    """

    # Mean
    mu = 127
    # Standard deviation
    sigma = 64
    a = sigma / np.std(image)
    b = mu - np.mean(image) * a
    img_out = image * a + b
    rows, cols = image.shape
    for y in range(rows):
        for x in range(cols):
            img_out[y, x] = np.clip(a * image[y, x] + b, 0, 255)
    return img_out


def coreg(G):
    """
    Calculates coefficients for the affine transform based on a facial mask using co-registration

    Args:
        G: coordinates to 4 registrations point of the face on the image
            - right eye
            - left eye
            - right mouth crevice
            - left mouth crevice

    Returns:
        a, b: Two arrays, the x and y coefficients

    """
    assert len(G) == 4, "Wrong number of control points"
    dx = np.array([257, 257, 439, 439])
    dy = np.array([340, 169, 316, 192])
    # (G^T*G)^-1*G*d
    gtg = np.matmul(np.transpose(G), G)
    gtgi = np.linalg.inv(gtg)
    gtgigt = np.matmul(gtgi, np.transpose(G))
    a = np.matmul(gtgigt, dx)
    b = np.matmul(gtgigt, dy)
    return a, b


if __name__ == "__main__":
    f = imread("portrett.png", as_gray=True)
    G = np.array([[67, 119, 1],
                  [88, 82, 1],
                  [100, 141, 1],
                  [115, 115, 1]])

    a, b = coreg(G)
    f_s = std_contrast(f)
    fw = transformation_fw(f_s, a, b)
    imsave('fw.png', fw)
    cni = transformation_cni(f_s, a, b)
    imsave('cni.png', cni)
    bli = transformation_bli(f_s, a, b)
    imsave('bli.png', cni)

    fig = plt.figure(figsize=(15, 10))

    fig.add_subplot(1, 2, 1)
    plt.imshow(f, cmap='gray')
    plt.title('Orginal')
    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    plt.imshow(f_s, cmap='gray')
    plt.title('Gray scale transformation')
    plt.axis('off')

    plt.savefig('grayscale.png')

    plt.close()

    fig = plt.figure(figsize=(30, 10))

    fig.add_subplot(1, 4, 1)
    plt.imshow(f_s, cmap='gray')
    plt.title('Orginal')
    plt.axis('off')

    fig.add_subplot(1, 4, 2)
    plt.imshow(fw, cmap='gray')
    plt.title('Forward mapping')
    plt.axis('off')

    fig.add_subplot(1, 4, 3)
    plt.imshow(cni, cmap='gray')
    plt.title('Closest neighbour interpolation')
    plt.axis('off')

    fig.add_subplot(1, 4, 4)
    plt.imshow(bli, cmap='gray')
    plt.title('Bilinear interpolation')
    plt.axis('off')

    plt.savefig('processed.png')
