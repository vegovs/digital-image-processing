import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave


def hysteresis_threshold(g_n, t_h, t_l):
    """
    Performs an hysteresis threshold to sample out the "good" edges.
    Args:
        g_n: Normalized gradient magnitude
        t_h: High threshold
        t_l: Low Threshold

    Returns:
        img_out: Detected edges

    """
    T_h = g_n.max() * t_h
    T_l = T_h * t_l

    g_nh = np.where(g_n >= T_h, g_n, 0)
    g_nl = np.where(g_n >= T_l, g_n, 0)

    g_nl = g_nl - g_nh

    weak = 1
    strong = 255

    # Valid edge pixels
    g_nh[g_nh > 0] = strong
    g_nl[g_nl > 0] = weak

    cols, rows = g_nl.shape

    for x in range(cols):
        for y in range(rows):
            if g_nl[x, y] == weak:
                try:
                    if ((g_nh[x + 1, y] == strong) or
                            (g_nh[x + 1, y + 1] == strong) or
                            (g_nh[x + 1, y - 1] == strong) or
                            (g_nh[x, y + 1] == strong) or
                            (g_nh[x, y - 1] == strong) or
                            (g_nh[x - 1, y] == strong) or
                            (g_nh[x - 1, y + 1] == strong) or
                            (g_nh[x - 1, y - 1] == strong)):
                        g_nl[x, y] = strong
                    else:
                        g_nl[x, y] = 0
                except IndexError as e:
                    pass

    img_out = g_nh + g_nl

    return img_out


def nonmaxima_suppression(g, theta):
    """
    Performs nonmaxima suppression on the gradient magnitude
    Args:
        g: Gradient magnitude of the image
        theta: Gradient direction scaled to 0, 45, 90 and 135 degrees.

    Returns:
        g_n: Nonmaxima suppressed image

    """
    cols, rows = g.shape
    g_n = np.zeros_like(g)

    for x in range(1, cols - 1):
        for y in range(1, rows - 1):
            try:
                # 0 degrees = south
                if theta[x, y] == 0.:
                    p = g[x, y - 1]
                    n = g[x, y + 1]
                # 45 degrees = south-east
                elif theta[x, y] == 45.:
                    p = g[x + 1, y - 1]
                    n = g[x - 1, y + 1]
                # 45 degrees = east
                elif theta[x, y] == 90.:
                    p = g[x + 1, y]
                    n = g[x - 1, y]
                # 135 degrees = north-east
                elif theta[x, y] == 135.:
                    p = g[x + 1, y + 1]
                    n = g[x - 1, y - 1]

                # If g(x,y) less than at least on neighbour, suppress
                if g[x, y] < p or g[x, y] < n:
                    g_n[x, y] = 0
                else:
                    g_n[x, y] = g[x, y]
            # Skip edges
            except IndexError as e:
                pass

    return g_n


def gradient_convolution(img_in):
    """
    Calculates gradient using a symmetrical 1-dimensional gradient operator
    Args:
        img_in: Image to calculate gradient from

    Returns:
        g: Gradient magnitude
        theta: Gradient direction i two first quadrants
    """

    rows, cols = img_in.shape
    g_x = np.zeros_like(img_in)
    g_y = np.zeros_like(img_in)
    for i in range(rows):
        for j in range(cols):
            try:
                # Vertical gradient
                g_x[i, j] = img_in[i + 1, j] - img_in[i - 1, j]
                # Horizontal gradient
                g_y[i, j] = img_in[i, j - 1] - img_in[i, j + 1]
            except IndexError as e:
                pass

    # Gradient direction
    theta = np.arctan2(g_x, g_y)
    # Scale from radians to degrees and round angles to the two first quadrants
    theta = np.round(theta * 180 / np.pi * 1 / 45) * 45 % 180

    # Gradient magnitude
    g = np.hypot(g_x, g_y)
    # Scale gradient to (0-255)
    g = g / g.max()

    return g, theta


def gaussian_convolution(img_in, sigma):
    gk = _gaussian_kernel(sigma=sigma)
    return _general_convolution(img_in=img_in, kernel=gk)


def _gaussian_kernel(sigma):
    """
    Produces a gaussian kernel
    Args:
        sigma: Standard deviation

    Returns:
        kernel_norm: Gaussian Kernel

    """

    # Adapt filter size to sigma
    size = int(np.ceil(1 + 8 * sigma))
    # Create base filter mesh
    size = size // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    # Calculate kernel
    kernel = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    A = 1 / kernel.sum()
    kernel_norm = A * kernel

    return kernel_norm


def _general_convolution(img_in, kernel):
    """
    Performs a general convolution with an odd 1D or rectangular kernel.
    Adds mirror indexing as padding and keeps the size of the image input.
    Args:
        img_in: Image to be convoluted
        kernel: Kernel to convolve with

    Returns:
        img_out: Image convoluted with kernel

    """

    h, w = kernel.shape
    assert h % 2 != 0 and h == w, "Kernel not odd, or rectangular"

    # Rotate kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Offset due to padding
    x_off = h // 2
    y_off = w // 2

    img_out = np.zeros_like(img_in)
    rows, cols = img_out.shape

    # Add mirror padding to image
    padded = _mirror_indexing(img_in=img_in, kernel_size=h)

    # Calculate response on all pixels
    for x in range(cols):
        for y in range(rows):
            x_p = x + x_off
            y_p = y + y_off
            img_out[y, x] = (kernel * padded[y_p:y_p + h, x_p:x_p + w]).sum()

    return img_out


def _mirror_indexing(img_in, kernel_size):
    """
    Pads the image with the closest values.
    NB! To add 1 pixel padding use kernel_size 2.
    Args:
        img_in: Image to be padded
        kernel_size: Size of the kernel

    Returns:
        img_in: Padded image

    """
    for _ in range(kernel_size - 1):
        img_in = np.column_stack((img_in[:, 0], img_in))
        img_in = np.column_stack((img_in, img_in[:, img_in.shape[1] - 1]))
        img_in = np.row_stack((img_in[0, :], img_in))
        img_in = np.row_stack((img_in, img_in[img_in.shape[0] - 1, :]))

    return img_in


if __name__ == "__main__":
    f = imread("cellekjerner.png", as_gray=True)

    sigma = 6
    blurred = gaussian_convolution(img_in=f, sigma=sigma)
    gradient_magnitude, gradient_direction = gradient_convolution(img_in=blurred)
    suppressed = nonmaxima_suppression(g=gradient_magnitude, theta=gradient_direction)
    detected_edges = hysteresis_threshold(g_n=suppressed, t_h=0.28, t_l=0.11)

    imsave('blurred.png', blurred)
    imsave('gradient_magnitude.png', gradient_magnitude)
    imsave('gradient_direction.png', gradient_direction)
    imsave('suppressed.png', suppressed)
    imsave('detected_edges.png', detected_edges)

    fig = plt.figure(figsize=(30, 15))

    fig.add_subplot(3, 3, 1)
    plt.imshow(f, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    fig.add_subplot(3, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Gaussian blur with standard deviation = %.2f' % sigma)
    plt.axis('off')

    fig.add_subplot(3, 3, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient magnitude using symmetric 1D gradient operator')
    plt.axis('off')

    fig.add_subplot(3, 3, 4)
    plt.imshow(gradient_direction, cmap='gray')
    plt.title('Gradient direction using symmetric 1D gradient operator(Degrees[0,45,90,135])')
    plt.axis('off')

    fig.add_subplot(3, 3, 5)
    plt.imshow(suppressed, cmap='gray')
    plt.title('Nonmaxima suppressed image')
    plt.axis('off')

    fig.add_subplot(3, 3, 6)
    plt.imshow(detected_edges, cmap='gray')
    plt.title('Detected edges')
    plt.axis('off')

    fig.savefig("steps.png")
