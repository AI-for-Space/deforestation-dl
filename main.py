from src.data_loading import DataLoader
from src.segmentation import Segmentator
from src.deforestation_model import DeforestationModel

import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():

    # SENTINEL-HUB PARAMETERS
    INSTACE_ID = '98e90c95-04e9-4aa8-a105-688da74595be'
    SH_CLIENT_ID = '1c7168a0-37b5-444f-a7e6-826ae6c19d90'
    SH_CLIENT_SECRET = 'f&}UR;bV(I)fx?r|:hlNZ0sK1utD4ny_4V0WsQzJ'

    loader = DataLoader(INSTACE_ID,SH_CLIENT_ID,SH_CLIENT_SECRET)
    segmentator = Segmentator()
    model = DeforestationModel()
    dataset_site_A = loader.get_image(-3.764226,-52.128754, 150, 2017, 2023, [6,7],['rgb','nvdi'],0.1)
    loader.display_random_samples(dataset_site_A,3,'rgb')
    loader.display_region_along_years(dataset_site_A,24,'rgb')
    segmentator.segment(dataset_site_A[2020]['rgb'][24],dataset_site_A[2020]['rgb'][24],True)
    deforestation = segmentator.get_ground_truth(dataset_site_A[2018]['rgb'][24],dataset_site_A[2019]['rgb'][24],True)
    
    conts,_ = cv2.findContours(deforestation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos en una máscara 
    image_with_conts = cv2.drawContours(np.clip(dataset_site_A[2019]['rgb'][24] * 2.5 / 255, 0, 1), conts, -1, (124,47,135), 1) # Dibujamos los contornos          
    plt.imshow(image_with_conts, cmap='gray')
    plt.title('Areas detected')
    plt.show()



if __name__ == "__main__":
    main()
