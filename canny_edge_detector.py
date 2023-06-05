import tkinter as tk
from tkinter import filedialog
import argparse

import numpy as np
from matplotlib import image as im
from matplotlib import pyplot as plt

# Define the color for strong and weak pixels
STRONG_GREY = 255
WEAK_GREY = 100

def get_image_gui():
    """
    Launches a GUI to select your image
    """
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title='Select the image to run the Canny Edge Detector on',
        # filetypes=[('Images', '.jpg')]
    )
    return image_path

def get_image():
    """
    Gets the image path and then reads in the image to a numpy array
    """
    image_path = get_image_gui()
    return im.imread(image_path)

def image_to_grayscale(image):
    """
    Converts the image from RGB to greyscale

    :param image: numpy array with RGB channels
    :return image: 2D numpy array representing greyscale image
    """
    # based off of https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def convolve(image, kernel):
    """
    Apply a kernel to the image

    :param image: 2D numpy array representing greyscale image
    :returns output: 2D numpy array representing the result of the kernel convolved with the image
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros_like(image)
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode='constant')
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

def gaussian_filter(image, filter_size, filter_sigma):
    """
    Create a Gaussian kernel and the convolve it with the image.

    :param image: 2D numpy array representing greyscale image
    :param filter_size: size of the Gaussian kernel
    :param filter_sigma: standard deviation of the Gaussian filter
    :return: Gaussian filter convolved with the greyscale image
    """
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*filter_sigma**2)) * np.exp(-((x-(filter_size//2))**2 + (y-(filter_size//2))**2)/(2*filter_sigma**2)), (filter_size, filter_size))
    kernel = kernel / np.sum(kernel)

    return convolve(image, kernel)

def calculate_intesity_gradient(image, operator="Sobel"):
    """
    Calculate the intensity gradient and direction of the Gaussian filtered image. The direction is further mapped to horizontal (0), vertical (90), positive diagonal (45) and negative diagonal (135).

    :param image: 2D numpy array representing the Gaussian filtered image
    :param operator: Optional parameter for which kernel to use ('Sobel', 'Prewitt' or 'Roberts cross'). Defaults to 'Sobel'.
    :returns: Tuple of the Gradient of each pixel and the direction of each pixel
    """
    kernel_horizontal, kernel_vertical = None, None
    if operator == "Sobel":
        # Sobel Operator kernels as described by https://en.wikipedia.org/wiki/Sobel_operator
        kernel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kernel_vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    elif operator == "Prewitt":
        # Prewitt Operator kernels as described by https://en.wikipedia.org/wiki/Prewitt_operator
        kernel_horizontal = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
        kernel_vertical = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
    else:
        # Roberts cross Operator kernels as described by https://en.wikipedia.org/wiki/Prewitt_operator
        kernel_horizontal = np.array([[1, 0], [0, -1]], np.float32)
        kernel_vertical = np.array([[0, 1], [-1, 0]], np.float32)

    horizontal_first_derivative = convolve(image, kernel_horizontal)
    vertical_first_derivative = convolve(image, kernel_vertical)

    image_height, image_width = image.shape
    edge_gradient = np.zeros_like(image)
    edge_direction = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            edge_gradient[i, j] = np.hypot(horizontal_first_derivative[i,j], vertical_first_derivative[i,j])
            if operator == "Sobel" or operator == "Prewitt":
                edge_direction[i, j] = np.arctan2(vertical_first_derivative[i,j], horizontal_first_derivative[i,j])
            else:
                edge_direction[i, j] = np.arctan(vertical_first_derivative[i,j]/horizontal_first_derivative[i,j]) - ((3*np.pi)/4)
            
            if edge_direction[i,j] < 0:
                edge_direction[i,j] += np.pi

            if 0 <= edge_direction[i, j] < 22.5 or  157.5 <= edge_direction[i, j] <= 180:
                edge_direction[i, j] = 0
            elif 22.5 <= edge_direction[i, j] < 67.5:
                edge_direction[i, j] = 45
            elif 67.5 <= edge_direction[i, j] < 112.5:
                edge_direction[i, j] = 90
            elif 112.5 <= edge_direction[i, j] < 157.5:
                edge_direction[i, j] = 135
            
    return (edge_gradient, edge_direction)

def lower_bound_cut_off_supression(gradient, direction):
    """
    For each pixel we look at the neighboring pixels in the same direction a keep the gradient if the current pixel is the greatest of the three and otherwise suprress it to 0.

    :param gradient: 2D numpy array representing the gradient of each pixel in the image
    :param direciton: 2D numpy array representing the direction of each pixel in the image
    :returns: image made up of gradients with 'sharpest change of intensity value' and all other values are 0
    """
    image_height, image_width = gradient.shape
    output = np.zeros_like(gradient)

    padded_gradient = np.pad(gradient, (1,), mode='constant')
    
    for i in range(image_height):
        for j in range(image_width):
            a, b = None, None
            x, y = i + 1, j + 1
            if direction[i,j] == 0:
                a = padded_gradient[x, y-1]
                b = padded_gradient[x, y+1]
            elif direction[i,j] == 45:
                a = padded_gradient[x-1, y+1]
                b = padded_gradient[x+1, y-1]
            elif direction[i,j] == 90:
                a = padded_gradient[x-1, y]
                b = padded_gradient[x+1, y]
            elif direction[i,j] == 135:
                a = padded_gradient[x-1, y-1]
                b = padded_gradient[x+1, y+1]
            
            if a <= gradient[i, j]  and b <= gradient[i, j]:
                output[i,j] = gradient[i, j]
            else:
                output[i, j] = 0
    
    return output

def double_threshold(image, low_threshold, high_threshold):
    """
    Marks pixels above the highthreshold as strong pixels, pixels between the low and high threshold as weak and suppresses to 0 all other pixels.

    :param image: 2D numpy array made up of gradients with 'sharpest change of intensity value' and all other values are 0
    :param low_threshold: gradient value below which pixels are suppressed
    :param high_threshold: gradient value above which pixels are kept
    :return output: 2D numpy array made up of strong pixels, and weak pixels and 0 for all other pixels
    """
    image_height, image_width = image.shape
    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            if image[i,j] >= high_threshold:
                output[i,j] = STRONG_GREY
            elif image[i,j] >= low_threshold:
                output[i,j] = WEAK_GREY

    return output

def check_neighbors(i,j, padded_image):
    """
    A helper function that checks the surrounding neighbors of a pixel and returns True if one is a strong pixel otherwise False.

    :param i: the row of the cell neighborhood
    :param j: the column of the cell neighborhood
    :param padded_image: 2D numpy array made up of strong pixels, and weak pixels and 0 for all other pixels padded by a row and column of 0s on all sides
    :returns: A boolean, True if there is a heigboring strong pixel and False otherwise
    """
    for x in range(-1, 2):
        for y in range(-1, 2):
            if padded_image[i+x, j+y] == STRONG_GREY:
                return True
    return False

def edge_tracking(image):
    """
    Looks at every pixel, keeps it if it is a strong pixel as well as if it is weak with a neighboring strong pixel.

    :param image: 2D numpy array made up of strong pixels, and weak pixels and 0 for all other pixels
    :returns output: A 2D numpy array made up of strong pixels and 0 for all other pixels where the strong pixels represent an edge in the image
    """
    image_height, image_width = image.shape
    output = np.zeros_like(image)
    padded_image = np.pad(image, (1,), mode='constant')

    for i in range(image_height):
        for j in range(image_width):
            if (image[i,j] == WEAK_GREY and check_neighbors(i+1,j+1, padded_image)) or image[i,j] == STRONG_GREY:
                output[i,j] = STRONG_GREY

    return output

def add_subplot(image, index, cmap='gray'):
    """
    Helper method to plot each stage of the algorithm
    """
    plt.subplot(2, 3, index)
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

def canny_edge_detector(
    image,
    low_threshold=100,
    high_threshold=200,
    filter_size=5,
    filter_sigma=1
):
    """
    canny_edge_detector runs the Canny Edge Detector algorithm on the image to return an image made up of the edges of the image

    :param image: 2D numpy array representing the image to be processed
    :param low_threshold: gradient value below which pixels are suppressed
    :param high_threshold: gradient value above which pixels are kept
    :param filter_size: size of the Gaussian kernel
    :param filter_sigma: standard deviation of the Gaussian filter
    :returns: Nothing, put will plot show the image at each step in the algorithm as well as the final result
    """
    image_grayscale = image_to_grayscale(image)
    add_subplot(image_grayscale, 1)

    image_gaussian_filter = gaussian_filter(image_grayscale, filter_size, filter_sigma)
    add_subplot(image_gaussian_filter, 2)

    image_intensity_gradient, image_gradient_direction = calculate_intesity_gradient(image_gaussian_filter)
    add_subplot(image_intensity_gradient, 3)

    image_supressed = lower_bound_cut_off_supression(image_intensity_gradient, image_gradient_direction)
    add_subplot(image_supressed, 4)

    image_threshold = double_threshold(image_supressed, low_threshold, high_threshold)
    add_subplot(image_threshold, 5)

    image_edges = edge_tracking(image_threshold)
    add_subplot(image_edges, 6)    

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an image into its edges by the Canny Edge Detector.")
    parser.add_argument("--low-threshold", type=float, help="Optional tuning parameter for the gradient value below which pixels are suppressed", default=100, required=False)
    parser.add_argument("--high-threshold", type=float, help="Optional tuning parameter for the gradient value below which pixels are suppressed", default=200, required=False)
    parser.add_argument("--filter-size", type=lambda s: s.isdigit() and int(s)%2 != 0, help="Optional tuning parameter for the size of the Gaussian kernel (odd positive integer)", default=5, required=False)
    parser.add_argument("--filter-sigma", type=float, help="Optional tuning parameter for the standard deviation of the Gaussian kernel", default=1, required=False)
    args = parser.parse_args()
    image = get_image()

    canny_edge_detector(
        image,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        filter_size=int(args.filter_size),
        filter_sigma=args.filter_sigma
    )