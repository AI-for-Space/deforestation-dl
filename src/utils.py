import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import random
from src.segmentation import Segmentator
from sklearn.model_selection import train_test_split
import os
import cv2
segmentator = Segmentator()

def get_train_test_data(datasets, year_1, year_2, test_size):

        X_year_1 = []
        X_year_2 = [] # These are RGB Images from the Year 2 
        Y = [] # These are the masks GT
        
        for dataset in datasets:
            X_year_1 += dataset[year_1]['rgb']
            X_year_2 += dataset[year_2]['rgb']
            Y += (map(segmentator.get_ground_truth, dataset[year_1]['rgb'], dataset[year_2]['rgb']))

        # Convert to numpy arrays
        X_year_1 = np.array(X_year_1)
        X_year_2 = np.array(X_year_2)
        Y = np.array(Y)

        # Early Fusion: Concatenating the two dates.
        X = np.concatenate((X_year_1, X_year_2), axis = 3)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        #get number of samples with masks
        #num_samples_with_masks = np.any(y_train > 0, axis=(1, 2, 3)).sum()

        # Display stats
        print("=============================================================================")
        print (f"Train shape {X_train.shape}")
        print (f"Y_train shape {y_train.shape}")
        print (f"Number of samples with deforestation in Train {np.any(y_train, axis=(1, 2, 3)).sum()}")
        print (f"Test shape {X_test.shape}")
        print("=============================================================================")
        

        return X_train, X_test, y_train, y_test, X_year_1, X_year_2, X, Y


def display_images_deforestation(image_year1,image_year2,mask):
        # Create a figure with two subplots
        plt.figure(figsize=(12, 5))

        # Plot accuracy on the first subplot
        plt.subplot(1, 3, 1)
        plt.imshow(image_year1,cmap='gray')
        plt.title('Year 1')

        # Plot accuracy on the first subplot
        plt.subplot(1, 3, 2)
        plt.imshow(image_year2,cmap='gray')
        plt.title('Year 2')

        # Plot accuracy on the first subplot
        plt.subplot(1, 3, 3)
        plt.imshow(mask,cmap='gray')
        plt.title('Mask')

        plt.tight_layout()  # Ensure that the subplots don't overlap
        plt.show()


def display_picked_areas():
        imageA = cv2.imread(os.path.join(os.getcwd(),"dataset/2020/Latitude_-3.76_Longitude_-52.13_Type_rgb.png"))
        imageB = cv2.imread(os.path.join(os.getcwd(),"dataset/2020/Latitude_-7.01_Longitude_-59.91_Type_rgb.png"))
        imageC = cv2.imread(os.path.join(os.getcwd(),"dataset/2020/Latitude_-7.34_Longitude_-55.31_Type_rgb.png"))
        imageD = cv2.imread(os.path.join(os.getcwd(),"dataset/2020/Latitude_-7.85_Longitude_-72.4_Type_rgb.png"))

        # Plot accuracy on the first subplot
        plt.subplot(1, 4, 1)
        plt.imshow(np.clip(imageA * 2.5 / 255, 0, 1),cmap='gray')
        plt.axis('off')  # Turn off axis values

        # Plot accuracy on the first subplot
        plt.subplot(1, 4, 2)
        plt.imshow(np.clip(imageB * 2.5 / 255, 0, 1),cmap='gray')
        plt.axis('off')  # Turn off axis values

        # Plot accuracy on the first subplot
        plt.subplot(1, 4, 3)
        plt.imshow(np.clip(imageC * 2.5 / 255, 0, 1),cmap='gray')
        plt.axis('off')  # Turn off axis values

        # Plot accuracy on the first subplot
        plt.subplot(1, 4, 4)
        plt.imshow(np.clip(imageD * 2.5 / 255, 0, 1),cmap='gray')
        plt.axis('off')  # Turn off axis values

        plt.tight_layout()  # Ensure that the subplots don't overlap
        plt.show()
