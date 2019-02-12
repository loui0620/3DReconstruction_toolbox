import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

import pairwise_feature
import toolbox
import structure

def dinosaur():
    # recontruct dino dataset
    imgPath1 = 'C:/imgs/dino/viff.003.ppm'
    imgPath2 = 'C:/imgs/dino/viff.001.ppm'
    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)
    pts1, pts2 = pairwise_feature.find_pairwise_features(img1, img2)
    points1 = toolbox.cart2homo(pts1)
    points2 = toolbox.cart2homo(pts2)


    height, width, ch = img1.shape
    intrinsic = np.array([[2360, 0, width/2],
                          [0, 2360, height/2],
                          [0, 0, 1]])

    return points1, points2, intrinsic


points1, points2, intrinsic = dinosaur()

# 1. normalize matches points
# 2. compute essential matrix by 2d matches
points1n = np.dot(np.linalg.inv(intrinsic), points1)
points2n = np.dot(np.linalg.inv(intrinsic), points2)
E = structure.compute_essential_normalized(points1n, points2n)
print('Essential Matrix: \n', -(E/E[0][1]))

Project1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
Project2s = structure.compute_P_from_essential(E)

ind = -1