import matplotlib.pyplot as plt



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
