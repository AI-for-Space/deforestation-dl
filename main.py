from src.data_loading import DataLoader
from src.segmentation import Segmentator
from src.deforestation_model import DeforestationModel
from src.utils import *

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def main():

    # SENTINEL-HUB PARAMETERS
    INSTACE_ID = '7be9d07b-1941-4229-923b-d8d5b5142fbb'
    SH_CLIENT_ID = 'f61de7f1-e839-40ca-a88b-2f84f3757287'
    SH_CLIENT_SECRET = '+RZq[zhn!@w,p:>C]LX&XT0*FIqoje4;*+V5WQ]D'

    # DATA LOADING CONFIGURATION PARAMETERS
    SIZE_FRAGMENTS = 128
    START_YEAR = 2017
    END_YEAR = 2023
    MONTHS = [6,7]
    TYPE_IMAGES = ['rgb','nvdi','nvdi_edited']
    CLOUD_COVERAGE = 0.1

    # MODEL PARAMETERS
    YEAR_1 = 2018
    YEAR_2 = 2019

    loader = DataLoader(INSTACE_ID,SH_CLIENT_ID,SH_CLIENT_SECRET)
    segmentator = Segmentator()
    
    dataset_site_A = loader.get_image(-3.764226,-52.128754, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)
    dataset_site_B = loader.get_image(-7.009882, -59.907156, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)
    dataset_site_C = loader.get_image(-7.337594, -55.309489, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)
    dataset_site_D = loader.get_image(-7.849816, -72.399943, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)

    X_train, X_test, Y_train, Y_test, X_year_1, X_year_2, X, Y = get_train_test_data([dataset_site_A,dataset_site_B,dataset_site_C,dataset_site_D],YEAR_1,YEAR_2,0.2)
    
    # Training 
    """
    model = DeforestationModel('u_net',X_train[0].shape)
    model.display_random_samples_years_mask(X_year_1, X_year_2, Y, 3)
    model.train(X_train,Y_train)

    # Predict
    predictions_deforestation = model.predict(X_test,Y_test)
    
    NUMBER_TEST_SAMPLES = 5
    model.display_predictions(X_test,predictions_deforestation,NUMBER_TEST_SAMPLES)

    """

    loader.display_random_samples_dataset(dataset_site_A,5,'rgb')
    loader.display_region_along_years(dataset_site_A,356,'rgb')
    #segmentator.segment(dataset_site_A[2021]['nvdi_edited'][356],dataset_site_A[2021]['rgb'][356],True)
    deforestation = segmentator.get_ground_truth(dataset_site_A[2017]['rgb'][356],dataset_site_A[2021]['rgb'][356],True)
    
    #conts,_ = cv2.findContours(deforestation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos en una máscara 
    #image_with_conts = cv2.drawContours(np.clip(dataset_site_A[2019]['rgb'][24] * 2.5 / 255, 0, 1), conts, -1, (124,47,135), 1) # Dibujamos los contornos          
    #plt.imshow(image_with_conts, cmap='gray')
    #plt.title('Areas detected')
    #plt.show()



if __name__ == "__main__":
    main()
