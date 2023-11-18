import matplotlib.pyplot as plt
import numpy as np
import random
import os

from src.segmentation import Segmentator
from src.utils import *
from src.models.u_net import *
from src.models.res_u_net import *

import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class DeforestationModel:
    def __init__(self, model_architecture,input_shape):
        self.segmentator = Segmentator()
        self.selected_architecture = model_architecture

        if model_architecture == 'u_net':
            self.model = unet_model(input_shape)
        elif model_architecture == 'res_u_net':
            self.model = res_unet_model(input_shape)
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
            model_history = self.model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=32)
            
            # Save the model weights
            self.model.save_weights(weights_file)
            print("Trained the model and saved weights.")


            # Print Results Training
            accuracy = model_history.history['accuracy']
            val_accuracy = model_history.history['val_accuracy']
            loss = model_history.history['loss']
            val_loss = model_history.history['val_loss']

            epochs = range(0, len(accuracy))

            # Gráficas
            plt.style.use("ggplot")
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.plot(epochs, model_history.history["loss"], label="train_loss")
            plt.plot(epochs, model_history.history["val_loss"], label="val_loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(epochs, model_history.history["accuracy"], label="train_acc")
            plt.plot(epochs, model_history.history["val_accuracy"], label="val_acc")
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    
    def predict(self,X_test,Y_test):
        deforestation_predictions = self.model.predict(X_test)
        y_predictions_evaluation = deforestation_predictions.reshape(-1)
        y_test_evaluation = Y_test.reshape(-1)
        y_predictions_evaluation = (y_predictions_evaluation > 0.1).astype(np.uint8)

        tn, fp, fn, tp = metrics.confusion_matrix(y_test_evaluation, y_predictions_evaluation).ravel()
        # print(tn, fp, fn, tp)
        
        intersection = np.sum(np.logical_and(y_predictions_evaluation, y_test_evaluation)) # Logical AND  
        union = np.sum(np.logical_or(y_predictions_evaluation, y_test_evaluation)) # Logical OR 
            
        accu = (tp + tn)/(tn + fp + fn + tp)
        Prec = tp/(tp + fp)
        R = tp/(tp + fn)
        F1 = 2 * Prec* R/(Prec + R)
        Iou = intersection/union
        Alarm_Area = (tp + fp)/(tn + fp + fn + tp)

        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_evaluation, y_predictions_evaluation)

        # Calculate Jaccard (IoU)
        jaccard = jaccard_score(y_test_evaluation, y_predictions_evaluation)

        f1 = f1_score(y_test_evaluation, y_predictions_evaluation)
        precision = precision_score(y_test_evaluation, y_predictions_evaluation)
        recall = recall_score(y_test_evaluation, y_predictions_evaluation)

        print("Accuracy:", accuracy, "-",accu)
        print("Jaccard (IoU):", jaccard,"-",Iou)
        print("Dice Score F1:", f1,"-",F1)
        print("Precision:", precision,"-",Prec)
        print("Recall:", recall,"-",R)
        print("Alarm area:, ", Alarm_Area,"-")


        return deforestation_predictions
    
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
