# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


   
if __name__ == "__main__":
    
    img1 = np.asarray(Image.open("C:/Users/spy/Desktop/MatLab/img/灰度图.jpg").convert("L"))
    img2 = np.asarray(Image.open("C:/Users/spy/Desktop/MatLab/img/11.jpg").convert("L"))
    
    mse_score = mse(img1, img2)
    ssim_score = ssim(img1, img2)
    print(mse_score)
    print(ssim_score)