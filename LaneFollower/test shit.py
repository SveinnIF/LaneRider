import cv2
import numpy as np
import picamera
import matplotlib.pyplot as plt
from easygopigo3 import EasyGoPiGo3
from sklearn.metrics import mean_squared_error
GPG = EasyGoPiGo3()

def best_pol_ord(x, y):
    pol1 = np.polyfit(y, x, 1)
    pred1 = pol_calc(pol1, y)
    mse1 = mean_squared_error(x, pred1)
    if mse1 < DEV_POL:
        return pol1, mse1
    pol2 = np.polyfit(y, x, 2)
    pred2 = pol_calc(pol2, y)
    mse2 = mean_squared_error(x, pred2)
    if mse2 < DEV_POL or mse1 / mse2 < MSE_DEV:
            return pol2, mse2
    else:
        pol3 = np.polyfit(y, x, 3)
        pred3 = pol_calc(pol3, y)
        mse3 = mean_squared_error(x, pred3)
        if mse2 / mse3 < MSE_DEV:
            return pol2, mse2
        else:
            return pol3, mse3