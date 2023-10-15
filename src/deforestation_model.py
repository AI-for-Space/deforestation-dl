import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from skimage import morphology
from scipy.ndimage import binary_fill_holes
from src.segmentation import Segmentator

from models.FastSAM.fastsam import FastSAM, FastSAMPrompt

from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

class DeforestationModel:
    def __init__(self):
        self.segmentator = Segmentator()
   
    def get_ground_truth(self, image_reference_1, image_reference_2):
        # Call the 'segment' function to obtain the segmented masks for each image
        mask_reference_1 = self.segmentator.segment(image_reference_1)
        mask_reference_2 = self.segmentator.segment(image_reference_2)

        # Calculate the ground truth based on the comparison of reference masks
        mask_difference = cv2.absdiff(mask_reference_1,mask_reference_2)

        # Eliminamos pequeños huecos en blanco
        img_mask_clean = morphology.remove_small_objects(mask_difference.astype('bool'),min_size=50).astype('uint8')
        ground_truth = morphology.remove_small_holes(img_mask_clean.astype('bool'), area_threshold=50).astype('uint8')

        plt.imshow(mask_reference_1, cmap='gray')
        plt.title('First Year')
        plt.show()
        plt.imshow(mask_reference_2, cmap='gray')
        plt.title('Second Year')
        plt.show()
        plt.imshow(ground_truth, cmap='gray')
        plt.title('GT')
        plt.show()

        return ground_truth
