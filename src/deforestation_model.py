import matplotlib.pyplot as plt
import numpy as np
import random
import os

from src.segmentation import Segmentator
from src.utils import *
from sklearn.model_selection import train_test_split
from src.models.u_net import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import Model, load_model


class DeforestationModel:
    def __init__(self, model_architecture,input_shape):
        self.segmentator = Segmentator()
        self.selected_architecture = model_architecture

        if model_architecture == 'u_net':
            self.model = unet_model(input_shape)
        else:
            self.model = unet_model(input_shape)

    
    def train(self,X_train, Y_train):
        weights_file = self.selected_architecture+"_model_weights.h5"

        # Check if the weights file exists
        if os.path.exists(weights_file):
            # If the weights file exists, load the model with the pre-trained weights
            self.model.load_weights(weights_file)
            print(f"Loaded pre-trained weights fromm file {weights_file}.")
        else:
            # If the weights file doesn't exist, train the model and save the weights
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model_history = self.model.fit(X_train, Y_train, validation_split=0.1, epochs=10, batch_size=32)
            
            # Save the model weights
            self.model.save_weights(weights_file)
            print("Trained the model and saved weights.")


            # Print Results Training
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
    
    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def display_random_samples_years_mask(self, dataset_1, dataset_2, Y, number_of_samples):
        ncols = 3
        nrows = number_of_samples  # Number of rows of samples

        fig, axs = plt.subplots(ncols=3, nrows=nrows, figsize=(5 * ncols , 5 * nrows))

        for i in range(nrows):
            index_random_image = random.randint(0, len(dataset_1))
            
            ax = axs[i, 0]
            image = dataset_1[index_random_image]
            ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
            ax.set_title(f"Image First Year", fontsize=10)
            ax.axis('off')

            ax = axs[i, 1]
            image = dataset_2[index_random_image]
            ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
            ax.set_title(f"Image Second Year", fontsize=10)
            ax.axis('off')

            ax = axs[i, 2]
            image = Y[index_random_image]
            ax.imshow(image)
            ax.set_title(f"Image Second Year", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_predictions(self, X_test, predicted_deforestation, number_of_samples):
        # Create a figure with two subplots
        fig, axs = plt.subplots(ncols=3, nrows=number_of_samples, figsize=(5 * 3 , 5 * number_of_samples))
        
        for i in range(number_of_samples):

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
