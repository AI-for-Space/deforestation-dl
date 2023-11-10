import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from skimage import morphology
from scipy.ndimage import binary_fill_holes

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

class Segmentator:
   
    def segment_fast_sam(self, year, index):
        # Define an inference source
        #source = self.year_data[year][index]
        source = "./wood2_true_color.png"
        # Create a FastSAM model
        model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt

        # Run inference on an image
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Everything prompt
        ann = prompt_process.text_prompt(text='deforestation')
        prompt_process.plot(annotations=ann, output_path='./output/image1.jpg')

        # Calculate the area of each mask
        mask_areas = []
        for mask in ann:
            mask_area = mask.sum().item()  # Calculate the sum of non-zero pixels and convert to a Python float
            mask_areas.append(mask_area)

        # Print the areas of the masks
        for i, area in enumerate(mask_areas):
            print(f"Mask {i + 1} Area: {area}")



        # ==============DISPLAY IMAGE==============

        # Convert the PyTorch tensor to a NumPy array
        masks = ann.numpy()

        image = cv2.imread(source)

        # Create an empty mask image with the same dimensions as the original image
        mask_image = np.zeros_like(image)

        # Define a list of colors for the text
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(masks))]


        # Iterate through the masks and draw contours on the mask image
        for i, (mask, color) in enumerate(zip(masks, colors)):
            # Find contours in the binary mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contour on the mask image with the assigned color
            cv2.drawContours(mask_image, contours, -1, color, thickness=cv2.FILLED)

            # Calculate the area of the mask
            mask_area = mask.sum()
            mask_areas.append(mask_area)

            # Calculate the centroid of the mask
            M = cv2.moments(contours[0])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Display the mask area on the image (in white color for visibility)
            mask_area_text = f"Area {i + 1}: {mask_area}"
            cv2.putText(mask_image, mask_area_text, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Overlay the mask image on the original image
        result_image = cv2.addWeighted(image, 0.7, mask_image, 0.3, 0)

        # Display the result image
        cv2.imshow('Image with Masks', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def segment(self, image_to_segment, image_rgb=None, show_plots = False):

        image = None

        if show_plots is True and image_rgb is not None:
            plt.imshow(np.clip(image_rgb * 2.5 / 255, 0, 1), cmap='gray')
            plt.title('Image in RGB (Second image parameter)')
            plt.show()

        if show_plots is True:
            plt.imshow(np.clip(image_to_segment * 2.5 / 255, 0, 1), cmap='gray')
            plt.title('Image to segment')
            plt.axis('off')
            plt.show()
        
        # Otra forma de mostrar el histograma (solo visualización)
        if show_plots is True:
            plt.hist(image_to_segment.ravel(), bins=50) # .ravel convierte un array multidimensional en una dimension
            plt.grid(True)
            plt.title('Histogram of colors in image to segment')
            plt.show()

        green_img = image_to_segment[:,:,1]

        if show_plots is True:
            plt.imshow(green_img, cmap='gray')
            plt.title('Image with green Channel')
            plt.axis('off')
            plt.show()

        # Aplicamos un filtro gaussiano para emborronar las altas frecuencias
        image_gaus = cv2.GaussianBlur(green_img, (5,5), 0) # (5x5) es el tamaño del filtro y 0 es la desviación estándar

        if show_plots is True:
            plt.imshow(image_gaus, cmap='gray')
            plt.title('Image Gauss')
            plt.show()

        # Otra forma de mostrar el histograma (solo visualización)
        if show_plots is True:
            plt.hist(green_img.ravel(), bins=50) # .ravel convierte un array multidimensional en una dimension
            plt.title('Histogram of colors in image in green channel')
            plt.grid(True)
            plt.show()

        # Fijamos el umbral en base al histograma anterior
        t = 34

        # Extreaemos la máscara binaria
        maxim = 255
        _, final_mask = cv2.threshold(green_img, t, maxim, cv2.THRESH_BINARY)

        # Fijamos el umbral con el método de OTSU
        #t, final_mask = cv2.threshold(image_gaus,0,1,cv2.THRESH_OTSU) # 0 es por defecto y 1 es el valor máximo de la máscara
        
        if show_plots is True:
            print(np.unique(final_mask))

        # Visualizamos para corroborar que se obtiene el mismo resultado
        if show_plots is True:
            plt.imshow(final_mask, cmap='gray')
            plt.title('Mask with threshold t=' + str(t))
            plt.axis('off')
            plt.show()

        # Eliminamos pequeños huecos en blanco
        img_mask_clean = morphology.remove_small_objects(final_mask.astype('bool'),min_size=300).astype('uint8')
        img_mask_clean = morphology.remove_small_holes(img_mask_clean.astype('bool'), area_threshold=300).astype('uint8')

        # Visualizamos resultado final
        if show_plots is True:
            plt.imshow(img_mask_clean, cmap='gray')
            plt.title('Deleted holes')
            plt.axis('off')
            plt.show()


        # Visualizar la máscara resultante
        image_filled = binary_fill_holes(img_mask_clean).astype('uint8')

        if show_plots is True:
            plt.imshow(image_filled, cmap='gray')
            plt.title('Filled small holes')
            plt.axis('off')
            plt.show()


        # Dibujar los contornos de los lúmenes en color verde sobre la imagen original RGB. Nota: Utilizar los flags necesarios
        # para que los contornos en verde sean perfectamente visibles. 
        # Visualizar la imagen superpuesta
        if image_rgb is not None and show_plots is True:
            conts,_ = cv2.findContours(image_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos en una máscara 
            image_with_conts = cv2.drawContours(np.clip(image_rgb.copy() * 2.5 / 255, 0, 1), conts, -1, (124,47,135), 1) # Dibujamos los contornos          
            plt.imshow(image_with_conts, cmap='gray')
            plt.title('Ground Thruth')
            plt.axis('off')
            plt.show()

        return image_filled

    def get_ground_truth(self, image_reference_1, image_reference_2, show_plots = False):
        # Call the 'segment' function to obtain the segmented masks for each image
        mask_reference_1 = self.segment(image_reference_1)
        mask_reference_2 = self.segment(image_reference_2)

        # Calculate the ground truth based on the comparison of reference masks
        mask_difference = cv2.absdiff(mask_reference_1,mask_reference_2)

        # Eliminamos pequeños huecos en blanco
        img_mask_clean = morphology.remove_small_objects(mask_difference.astype('bool'),min_size=50).astype('uint8')
        ground_truth = morphology.remove_small_holes(img_mask_clean.astype('bool'), area_threshold=50).astype('uint8')

        ground_truth = ground_truth[:, :, np.newaxis]

        if show_plots is True:
            # Create a figure with two subplots
            plt.figure(figsize=(12, 5))

            plt.subplot(2, 3, 1)
            plt.imshow(np.clip(image_reference_1 * 2.5 / 255, 0, 1), cmap='gray')
            plt.axis('off')
            plt.title('Image RGB Year 1')

            plt.subplot(2, 3, 2)
            plt.imshow(np.clip(image_reference_2 * 2.5 / 255, 0, 1), cmap='gray')
            plt.axis('off')
            plt.title('Image RGB Year 2')

            plt.subplot(2, 3, 3)
            conts,_ = cv2.findContours(ground_truth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos en una máscara 
            image_with_conts = cv2.drawContours(np.clip(image_reference_2 * 2.5 / 255, 0, 1), conts, -1, (124,47,135), 1) # Dibujamos los contornos          
            plt.imshow(image_with_conts, cmap='gray')
            plt.axis('off')
            plt.title('Deforestation')

            plt.subplot(2, 3, 4)
            plt.imshow(mask_reference_1, cmap='gray')
            plt.axis('off')
            plt.title('Mask Year 1')

            plt.subplot(2, 3, 5)
            plt.imshow(mask_reference_2, cmap='gray')
            plt.axis('off')
            plt.title('Mask Year 2')

            plt.subplot(2, 3, 6)         
            plt.imshow(ground_truth, cmap='gray')
            plt.axis('off')
            plt.title('Deforestation Mask')
            
            plt.tight_layout()
            plt.show()

        return ground_truth

