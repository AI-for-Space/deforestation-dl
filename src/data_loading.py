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

class DataLoader:
    def __init__(self, instance_id, sh_client_id, sh_client_secret, sh_base_url='https://services.sentinel-hub.com'):
        
        self.sentinel_hub_config = SHConfig(
            instance_id = instance_id,
            sh_client_id = sh_client_id,
            sh_client_secret = sh_client_secret,
            sh_base_url = sh_base_url,
        )

        self.evalscript_true_color = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04", "B08"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """
        self.year_data={}

    def get_true_color_request(self,time_interval):
        return SentinelHubRequest(
            evalscript=self.evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                    maxcc = 0.3, #cloud coverage
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=self.image_bbox,
            size=self.image_size,
            config=self.sentinel_hub_config,
        )

    def get_image(self, latitude, longitude, size_fragments):
        # Define the desired width and height of the bounding box in degrees
        bbox_width = 0.3  # Example: 0.1 degrees (approximately 11.1 km at the equator)
        bbox_height = 0.3

        # Calculate the bounding box coordinates
        lower_left_lat = latitude - bbox_height / 2
        lower_left_lon = longitude - bbox_width / 2
        upper_right_lat = latitude + bbox_height / 2
        upper_right_lon = longitude + bbox_width / 2

        # Create the bounding box
        self.image_coords_wgs84 = (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
        self.resolution = 20
        self.image_bbox = BBox(bbox=self.image_coords_wgs84, crs=CRS.WGS84)
        self.image_size = bbox_to_dimensions(self.image_bbox, resolution=self.resolution) # Size of the final image
        
        # Create Time Intervals
        start_year = 2017
        end_year = 2023
        month = [7,8]

        data = []
    
        for year in range(start_year, end_year + 1):

            # Generate filename for image
            filedir = f"dataset/{year}"
            filename = f"Latitude_{round(latitude,2)}_Longitude_{round(longitude,2)}.png"
            filepath = os.path.join(os.getcwd(),filedir, filename)

            # Create the output directory if it doesn't exist
            os.makedirs(filedir, exist_ok=True)

            # Check if image exists
            if os.path.exists(filepath):
                print(f"Image for year {year} already stored")
                data.append(cv2.imread(filepath))
            
            else:
                start_date = datetime.datetime(year, month[0], 1)
                end_date = datetime.datetime(year, month[1], 31)
                time_interval = (start_date.date().isoformat(), end_date.date().isoformat())
                image =  self.get_true_color_request(time_interval).get_data()[0]
                print(f"Getting image from Sentinel Hub from year {year}")
                data.append(image)
                cv2.imwrite(filepath, image)
                

        # create a list of requests
        #list_of_requests = [self.get_true_color_request(slot) for slot in time_intervals]
        #list_of_requests = [request.download_list[0] for request in list_of_requests]

        # download data with multiple threads
        #data = SentinelHubDownloadClient(config=self.sentinel_hub_config).download(list_of_requests, max_threads=5)

        # Loop through each time interval
        year = start_year
        for image in data:
            fragment_list = []
            for y in range(0, self.image_size[1], size_fragments):
                for x in range(0, self.image_size[0], size_fragments):
                    fragment = image[y:y+size_fragments, x:x+size_fragments, :]
                    if fragment.shape[0] == size_fragments and fragment.shape[1] == size_fragments:
                        fragment_list.append(fragment)
            # Store fragments in arrays by year
            self.year_data[year] = fragment_list
            year = year + 1
    
    def display_random_samples(self, number_of_samples):
        ncols = len(self.year_data)
        nrows = number_of_samples  # Number of rows of samples

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols , 5 * nrows))

        for i in range(nrows):
            number_of_fragments = len(next(iter(self.year_data.values()))) # Number of sample of array of the first year
            index_random_image = random.randint(0, number_of_fragments)
            for j,year in enumerate(self.year_data.keys()):
                # Get a random image from the selected year
                ax = axs[i, j]
                image = self.year_data[year][index_random_image]
                ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
                ax.set_title(f"Year {year}", fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()