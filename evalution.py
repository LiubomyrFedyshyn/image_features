import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
def equal_error_rate(y,y_pred,**kwargs):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    return brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)