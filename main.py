from src.data_loading import DataLoader
from src.segmentation import Segmentator
from src.deforestation_model import DeforestationModel

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
    model = DeforestationModel()

    dataset_site_A = loader.get_image(-3.764226,-52.128754, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)
    dataset_site_B = loader.get_image(-7.009882, -59.907156, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)
    dataset_site_C = loader.get_image(-7.337594, -55.309489, SIZE_FRAGMENTS, START_YEAR, END_YEAR, MONTHS, TYPE_IMAGES, CLOUD_COVERAGE)

    """
    #X_train, X_test, Y_train, Y_test, X_year_1, X_year_2, X, Y = model.get_train_test_data([dataset_site_A,dataset_site_B,dataset_site_C],YEAR_1,YEAR_2,0.2)
    #model.display_random_samples(X_year_1, X_year_2, Y, 3)
    
    weights_file = "u_net_model_weights.h5"

    # Check if the weights file exists
    if os.path.exists(weights_file):
        # If the weights file exists, load the model with the pre-trained weights
        u_net_model = model.unet_model(X_train[0].shape)
        u_net_model.load_weights(weights_file)
        print("Loaded pre-trained weights.")
    else:
        # If the weights file doesn't exist, train the model and save the weights
        u_net_model = model.unet_model(X_train[0].shape)
        u_net_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_history = u_net_model.fit(X_train, Y_train, validation_split=0.1, epochs=10, batch_size=32)
        
        # Save the model weights
        u_net_model.save_weights(weights_file)
        print("Trained the model and saved weights.")

        accuracy = model_history.history['accuracy']
        val_accuracy = model_history.history['val_accuracy']
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

        epochs = range(1, len(accuracy) + 1)

        # Create a figure with two subplots
        plt.figure(figsize=(12, 5))

        # Plot accuracy on the first subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss on the second subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()  # Ensure that the subplots don't overlap
        plt.show()

    print("Shape of X_test: ", X_test.shape)
    # Predict
    predicted_deforestation = u_net_model.predict(X_test)
    

    NUMBER_TEST_SAMPLES = 5

    # Create a figure with two subplots
    fig, axs = plt.subplots(ncols=3, nrows=NUMBER_TEST_SAMPLES, figsize=(5 * 3 , 5 * NUMBER_TEST_SAMPLES))
    
    for i in range(NUMBER_TEST_SAMPLES):

        index_random_image = random.randint(0, X_test.shape[0])
        ax = axs[i, 0]
        ax.imshow(X_test[index_random_image][:,:,:3], cmap='gray')
        ax.set_title('First year')

        # Plot loss on the second subplot
        ax = axs[i, 1]
        ax.imshow(X_test[index_random_image][:,:,3:], cmap='gray')
        ax.set_title('Second year')

        # Plot loss on the second subplot
        ax = axs[i, 2]
        ax.imshow(predicted_deforestation[index_random_image], cmap='gray')
        ax.set_title('Deforestation')

    plt.tight_layout()  # Ensure that the subplots don't overlap
    plt.show()

    """

    loader.display_random_samples(dataset_site_A,5,'rgb')
    #loader.display_region_along_years(dataset_site_A,356,'nvdi_edited')
    #segmentator.segment(dataset_site_A[2021]['nvdi_edited'][356],dataset_site_A[2021]['rgb'][356],True)
    deforestation = segmentator.get_ground_truth(dataset_site_A[2019]['rgb'][356],dataset_site_A[2021]['rgb'][356],True)
    
    #conts,_ = cv2.findContours(deforestation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos en una máscara 
    #image_with_conts = cv2.drawContours(np.clip(dataset_site_A[2019]['rgb'][24] * 2.5 / 255, 0, 1), conts, -1, (124,47,135), 1) # Dibujamos los contornos          
    #plt.imshow(image_with_conts, cmap='gray')
    #plt.title('Areas detected')
    #plt.show()



if __name__ == "__main__":
    main()
