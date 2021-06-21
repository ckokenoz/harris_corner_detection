import os
import cv2
import numpy as np
import scipy
from scipy import signal
import math



def read_img(path):
    return cv2.imread(path, 0)


def save_img(img, path):
    cv2.imwrite(path,img)
    print(path, "is saved!")

def convolve(image, kernel):
    convolved = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')
    return convolved

def edge_detection(image):
    kx = [[-1, 0, 1]]
    ky = [[-1],[0],[1]]

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = Ix * Iy

    return Ix2, Iy2, IxIy


def harris_corner_detector(image, x_offset=5, y_offset=5, window_size=(5,5)):
    filterWindow = np.zeros((window_size[0],window_size[1]))
    sigmaSq = 1 / (2 * math.log(2))
    constant = 1 / (2 * math.pi * sigmaSq)
    for i in range(0, window_size[0]):
        for j in range(0, window_size[1]):
            filterWindow[i][j] = constant * math.exp((i**2 + j**2)/(-2*sigmaSq))

    gaus_image = convolve(image, filterWindow)
    Ix2, Iy2, IxIy = edge_detection(image)

    filterWindow = np.ones((window_size[0], window_size[1]))

    convolveIx2 = convolve(Ix2, filterWindow)
    convolveIy2 = convolve(Iy2, filterWindow)
    convolveIxIy = convolve(IxIy, filterWindow)

    output = np.zeros((convolveIx2.shape[0], convolveIx2.shape[1]))

    M = [[0, 0], [0, 0]]

    for i in range(convolveIx2.shape[0]):
        for j in range(convolveIx2.shape[1]):
            M[0][0] = convolveIx2[i][j]
            M[0][1] = convolveIxIy[i][j]
            M[1][0] = convolveIxIy[i][j]
            M[1][1] = convolveIy2[i][j]
            eig, v = np.linalg.eig(M)
            l1 = eig[0]
            l2 = eig[1]
            output[i][j] = l1 * l2 - 0.05 * ((l1+l2)*(l1+l2))

    return output


def main():

    img = read_img('./images/flower.png')

    if not os.path.exists("./result"):
        os.makedirs("./result")

    harris_corner_image = harris_corner_detector(img)
    save_img(harris_corner_image, "./result/img_corners.png")



if __name__ == "__main__":
    main()