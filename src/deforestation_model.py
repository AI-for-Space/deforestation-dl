import matplotlib.pyplot as plt
import numpy as np
import random

from src.segmentation import Segmentator
from src.utils import *
from sklearn.model_selection import train_test_split

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
    def __init__(self):
        self.segmentator = Segmentator()

    def get_train_test_data(self, datasets, year_1, year_2, test_size):

        X_year_1 = []
        X_year_2 = [] # These are RGB Images from the Year 2 
        Y = [] # These are the masks GT
        
        for dataset in datasets:
            X_year_1 += dataset[year_1]['rgb']
            X_year_2 += dataset[year_2]['rgb']
            Y += (map(self.segmentator.get_ground_truth, dataset[year_1]['rgb'], dataset[year_2]['rgb']))

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
    
    def display_random_samples(self, dataset_1, dataset_2, Y, number_of_samples):
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
        

    def double_conv_block(self, x, n_filters):
        # Conv2D then ReLU activation
        conv = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        # Conv2D then ReLU activation
        conv = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(conv)
        return conv

    def downsample_block(self, x, n_filters):
        conv_block = self.double_conv_block(x, n_filters)
        pool = MaxPooling2D(pool_size=(2, 2),strides=2, padding='same')(conv_block)
        output = Dropout(0.3)(pool)
        return conv_block, output

    def upsample_block(self, x, conv_features, n_filters):
        # upsample
        x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        # concatenate
        x = Concatenate(axis=3)([x, conv_features])
        # dropout
        x = Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = self.double_conv_block(x, n_filters)
        return x

    def unet_model(self,input_shape):

        inputs = keras.Input(shape = input_shape)
        # encoder: contracting path - downsample
        # 1 - downsample
        conv1, output_conv1 = self.downsample_block(inputs, 64)
        # 2 - downsample
        conv2, output_conv2 = self.downsample_block(output_conv1, 128)
        # 3 - downsample
        conv3, output_conv3 = self.downsample_block(output_conv2, 256)
        # 4 - downsample
        conv4, output_conv4 = self.downsample_block(output_conv3, 512)
        # 5 - bottleneck
        bottleneck = self.double_conv_block(output_conv4, 1024)
        # decoder: expanding path - upsample
        # 6 - upsample
        up_conv6 = self.upsample_block(bottleneck, conv4, 512)
        # 7 - upsample
        up_conv7 = self.upsample_block(up_conv6, conv3, 256)
        # 8 - upsample
        up_conv8 = self.upsample_block(up_conv7, conv2, 128)
        # 9 - upsample
        up_conv9 = self.upsample_block(up_conv8, conv1, 64)
        # outputs
        outputs = Conv2D(1, 1, padding="same", activation = "sigmoid")(up_conv9)
        # unet model with Keras Functional API
        unet_model = keras.Model(inputs, outputs, name="U-Net")
        unet_model.summary()

        return unet_model


