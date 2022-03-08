from webbrowser import get
import cv2
import numpy as np
import random
from scipy import signal
from scipy import fftpack

my_img = cv2.imread('./dog.jpg')


def getGrayImage(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    cv2.imwrite("dog-gray.jpg", gray_img)
    return gray_img
# noises


def salt_pepper_noise(img):
    row, col = img.shape
    selected_pixel = random.randint(100, 5000)
    for i in range(selected_pixel):
        # set these pixel to white
        x = random.randint(0, col-1)
        y = random.randint(0, row-1)
        img[y][x] = 255
    for i in range(selected_pixel):
        # set these pixel to black
        x = random.randint(0, col-1)
        y = random.randint(0, row-1)
        img[y][x] = 0

    return img


def gussian_noise(img):
    row, col = img.shape
    mean = 0.0
    std = 15.0
    noise = np.random.normal(mean, std, size=(row, col))
    img_noisy = np.add(img, noise)
    img_noisy = img_noisy.astype(np.uint8)
    return img_noisy


def uniform_noise(img):
    row, col = img.shape
    noise = np.random.uniform(-20, 20, size=(row, col))
    img_noisy = np.add(img, noise)
    img_noisy = img_noisy.astype(np.uint8)
    return img_noisy


def putNoize(noise, gray_img):
    if noise == 'Salt&pepper':
        noisy_img = salt_pepper_noise(gray_img)
        cv2.imwrite("salt&pepper-noise.jpg", noisy_img)
    elif noise == 'Gaussian':
        noisy_img = gussian_noise(gray_img)
        cv2.imwrite("gaussian-noise.jpg", noisy_img)
    else:
        noisy_img = uniform_noise(gray_img)
        cv2.imwrite("uniform-noise.jpg", noisy_img)
    return noisy_img


# filters
# all filters are of size 3x3

def apply_mask(img, mask):
    img_masked = signal.convolve2d(img, mask)
    img_masked = img_masked.astype(np.uint8)

    return img_masked

# 3x3


def convolution(img, mask):
    row, col = img.shape
    img_masked = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] + img[
                i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[i + 1, j - 1] * mask[
                2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

            img_masked[i, j] = temp

    img_masked = img_masked.astype(np.uint8)
    return img_masked


def ave_filter(img):
    # row, col = img.shape
    mask = np.ones([3, 3], dtype=int)
    mask = mask/9
    return convolution(img, mask)


def gaussian_filter(img, shape):
    sigma = 2.6

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return apply_mask(img, h)


def median_filter(img):
    row, col = img.shape
    img_masked = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            median_array = [img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                            img[i, j - 1], img[i, j], img[i, j + 1],
                            img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]]

            img_masked[i, j] = np.median(median_array)

    img_masked = img_masked.astype(np.uint8)
    return img_masked


def laplacian_filter(img):
    mask = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

    row, col = img.shape
    masked_img = np.zeros([row, col])
    for i in range(1, row - 2):
        for j in range(1, col - 2):
            Ix = np.sum(np.multiply(mask, img[i:i + 3, j:j + 3]))
            masked_img[i + 1, j + 1] = Ix
    masked_img = masked_img.astype(np.uint8)
    return np.uint8(masked_img)


def freq_domain_filter(img, filter_type):
    if filter_type == 'lpf':
        mask = np.ones([9, 9], dtype=int)
        mask = mask / 81

    if filter_type == 'hpf':
        mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # width of padding
    width = (img.shape[0] - mask.shape[0], img.shape[1] - mask.shape[1])
    mask = np.pad(mask, (((width[0] + 1) // 2, width[0] // 2), ((width[1] + 1) // 2, width[1] // 2)),
                  'constant')

    mask = fftpack.ifftshift(mask)

    filtered = np.real(fftpack.ifft2(fftpack.fft2(img) * fftpack.fft2(mask)))

    filtered = np.maximum(0, np.minimum(filtered, 255))
    filtered = filtered.astype(np.uint8)
    return filtered


def get_filter_image(filter, noisy_img):
    if filter == 'average filter':
        ave_img = ave_filter(noisy_img)
        cv2.imwrite("ave_img.jpg", ave_img)
    elif filter == 'Gaussian filter':
        gauss_img = gaussian_filter(noisy_img, (9, 9))
        cv2.imwrite("gauss_img.jpg", gauss_img)
    elif filter == 'Median filter':
        median_img = median_filter(noisy_img)
        cv2.imwrite("median_img.jpg", median_img)
    elif filter == 'Freq. low pass filter':
        freq_LPF = freq_domain_filter(noisy_img, 'lpf')
        cv2.imwrite("freq_LPF.jpg", freq_LPF)
    else:
        freq_HPF = freq_domain_filter(noisy_img, 'hpf')
        cv2.imwrite("freq_HPF.jpg", freq_HPF)


get_filter_image("average filter", putNoize(
    "Salt&pepper", getGrayImage(my_img)))
