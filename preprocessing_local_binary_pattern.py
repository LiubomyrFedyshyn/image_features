from multiprocessing import Pool, TimeoutError
from tqdm import tqdm_notebook

import numpy as np
import pandas as pd
import cv2

from skimage.feature import local_binary_pattern
from mahotas.features import haralick


def process_image(image_path):
    image = cv2.imread(image_path)
    return extract_features(image)

def extract_features(image):
    hsv_image, ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV), cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(hsv_image) + cv2.split(ycrcb_image)
    lbp_features = [local_binary_pattern(ch,18,2,method='uniform') for ch in channels]
    hist_lbp_features = [np.histogram(lf,bins=19,density=True)[0] for lf in lbp_features]
    haralick_features  = np.append(haralick(hsv_image).mean(axis=0),
                              haralick(ycrcb_image).mean(axis=0))
    return np.append(hist_lbp_features,haralick_features)

def process_data(filelist):
    def __impl(files):
        pool = Pool(2) 
        features = list(tqdm_notebook(pool.imap(process_image, files), total=len(files)))
        pool.close()
        return features
    feature_list = __impl(filelist) 
    return feature_list