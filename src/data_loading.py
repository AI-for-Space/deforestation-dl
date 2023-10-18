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

        self.evalscript_true_color_rgb = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
        """

        self.evalscript_nvdi = """
            //VERSION=3

            function setup() {
            return {
                input: [{
                bands:["B04", "B08"],
                }],
                output: {
                id: "default",
                bands: 3,
                }
            }
            }

            function evaluatePixel(sample) {
                let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

                if (val<-0.5) return [0.05,0.05,0.05]
                else if (val<-0.2) return [0.75,0.75,0.75]
                else if (val<-0.1) return [0.86,0.86,0.86]
                else if (val<0) return [0.92,0.92,0.92]
                else if (val<0.025) return [1,0.98,0.8]
                else if (val<0.05) return [0.93,0.91,0.71]
                else if (val<0.075) return [0.87,0.85,0.61]
                else if (val<0.1) return [0.8,0.78,0.51]
                else if (val<0.125) return [0.74,0.72,0.42]
                else if (val<0.15) return [0.69,0.76,0.38]
                else if (val<0.175) return [0.64,0.8,0.35]
                else if (val<0.2) return [0.57,0.75,0.32]
                else if (val<0.25) return [0.5,0.7,0.28]
                else if (val<0.3) return [0.44,0.64,0.25]
                else if (val<0.35) return [0.38,0.59,0.21]
                else if (val<0.4) return [0.31,0.54,0.18]
                else if (val<0.45) return [0.25,0.49,0.14]
                else if (val<0.5) return [0.19,0.43,0.11]
                else if (val<0.55) return [0.13,0.38,0.07]
                else if (val<0.6) return [0.06,0.33,0.04]
                else return [0,0.27,0];  
            }
        """

        self.evalscript_functions = {
            'rgb': """
                    //VERSION=3

                    function setup() {
                        return {
                            input: [{
                                bands: ["B02", "B03", "B04"]
                            }],
                            output: {
                                bands: 3
                            }
                        };
                    }

                    function evaluatePixel(sample) {
                        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                    }
                """,
            'nvdi': """
                    //VERSION=3

                    function setup() {
                    return {
                        input: [{
                        bands:["B04", "B08"],
                        }],
                        output: {
                        id: "default",
                        bands: 3,
                        }
                    }
                    }

                    function evaluatePixel(sample) {
                        let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

                        if (val<-0.5) return [0.05,0.05,0.05]
                        else if (val<-0.2) return [0.75,0.75,0.75]
                        else if (val<-0.1) return [0.86,0.86,0.86]
                        else if (val<0) return [0.92,0.92,0.92]
                        else if (val<0.025) return [1,0.98,0.8]
                        else if (val<0.05) return [0.93,0.91,0.71]
                        else if (val<0.075) return [0.87,0.85,0.61]
                        else if (val<0.1) return [0.8,0.78,0.51]
                        else if (val<0.125) return [0.74,0.72,0.42]
                        else if (val<0.15) return [0.69,0.76,0.38]
                        else if (val<0.175) return [0.64,0.8,0.35]
                        else if (val<0.2) return [0.57,0.75,0.32]
                        else if (val<0.25) return [0.5,0.7,0.28]
                        else if (val<0.3) return [0.44,0.64,0.25]
                        else if (val<0.35) return [0.38,0.59,0.21]
                        else if (val<0.4) return [0.31,0.54,0.18]
                        else if (val<0.45) return [0.25,0.49,0.14]
                        else if (val<0.5) return [0.19,0.43,0.11]
                        else if (val<0.55) return [0.13,0.38,0.07]
                        else if (val<0.6) return [0.06,0.33,0.04]
                        else return [0,0.27,0];  
                    }
                """
        }

        self.evalscript_nvdi = """
            //VERSION=3

            function setup() {
            return {
                input: [{
                bands:["B04", "B08"],
                }],
                output: {
                id: "default",
                bands: 3,
                }
            }
            }

            function evaluatePixel(sample) {
                let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

                if (val<-0.5) return [0.05,0.05,0.05]
                else if (val<-0.2) return [0.75,0.75,0.75]
                else if (val<-0.1) return [0.86,0.86,0.86]
                else if (val<0) return [0.92,0.92,0.92]
                else if (val<0.025) return [1,0.98,0.8]
                else if (val<0.05) return [0.93,0.91,0.71]
                else if (val<0.075) return [0.87,0.85,0.61]
                else if (val<0.1) return [0.8,0.78,0.51]
                else if (val<0.125) return [0.74,0.72,0.42]
                else if (val<0.15) return [0.69,0.76,0.38]
                else if (val<0.175) return [0.64,0.8,0.35]
                else if (val<0.2) return [0.57,0.75,0.32]
                else if (val<0.25) return [0.5,0.7,0.28]
                else if (val<0.3) return [0.44,0.64,0.25]
                else if (val<0.35) return [0.38,0.59,0.21]
                else if (val<0.4) return [0.31,0.54,0.18]
                else if (val<0.45) return [0.25,0.49,0.14]
                else if (val<0.5) return [0.19,0.43,0.11]
                else if (val<0.55) return [0.13,0.38,0.07]
                else if (val<0.6) return [0.06,0.33,0.04]
                else return [0,0.27,0];  
            }
        """

        self.evalscript_functions = {
            'rgb': """
                    //VERSION=3

                    function setup() {
                        return {
                            input: [{
                                bands: ["B02", "B03", "B04"]
                            }],
                            output: {
                                bands: 3
                            }
                        };
                    }

                    function evaluatePixel(sample) {
                        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                    }
                """,
            'nvdi': """
                    //VERSION=3

                    function setup() {
                    return {
                        input: [{
                        bands:["B04", "B08"],
                        }],
                        output: {
                        id: "default",
                        bands: 3,
                        }
                    }
                    }

                    function evaluatePixel(sample) {
                        let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

                        if (val<-0.5) return [0.05,0.05,0.05]
                        else if (val<-0.2) return [0.75,0.75,0.75]
                        else if (val<-0.1) return [0.86,0.86,0.86]
                        else if (val<0) return [0.92,0.92,0.92]
                        else if (val<0.025) return [1,0.98,0.8]
                        else if (val<0.05) return [0.93,0.91,0.71]
                        else if (val<0.075) return [0.87,0.85,0.61]
                        else if (val<0.1) return [0.8,0.78,0.51]
                        else if (val<0.125) return [0.74,0.72,0.42]
                        else if (val<0.15) return [0.69,0.76,0.38]
                        else if (val<0.175) return [0.64,0.8,0.35]
                        else if (val<0.2) return [0.57,0.75,0.32]
                        else if (val<0.25) return [0.5,0.7,0.28]
                        else if (val<0.3) return [0.44,0.64,0.25]
                        else if (val<0.35) return [0.38,0.59,0.21]
                        else if (val<0.4) return [0.31,0.54,0.18]
                        else if (val<0.45) return [0.25,0.49,0.14]
                        else if (val<0.5) return [0.19,0.43,0.11]
                        else if (val<0.55) return [0.13,0.38,0.07]
                        else if (val<0.6) return [0.06,0.33,0.04]
                        else return [0,0.27,0];  
                    }
                """,
                'nvdi_edited': """
                    //VERSION=3

                    function setup() {
                    return {
                        input: [{
                        bands:["B04", "B08"],
                        }],
                        output: {
                        id: "default",
                        bands: 3,
                        }
                    }
                    }

                    function evaluatePixel(sample) {
                        let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

                        if (val<-0.5) return [0.05,0.05,0.05]
                        else if (val<-0.2) return [0.75,0.75,0.75]
                        else if (val<-0.1) return [0.86,0.86,0.86]
                        else if (val<0) return [0.92,0.92,0.92]
                        else if (val<0.025) return [1,0.98,0.8]
                        else if (val<0.05) return [0.93,0.91,0.71]
                        else if (val<0.075) return [0.87,0.85,0.61]
                        else if (val<0.1) return [0.8,0.78,0.51]
                        else if (val<0.125) return [0.74,0.72,0.42]
                        else if (val<0.15) return [0.69,0.76,0.38]
                        else if (val<0.175) return [0.64,0.8,0.35]
                        else if (val<0.2) return [0.57,0.75,0.32]
                        else if (val<0.25) return [0.5,0.7,0.28]
                        else if (val<0.3) return [0.44,0.64,0.25]
                        else if (val<0.35) return [0.38,0.59,0.21]
                        else if (val<0.4) return [0.31,0.54,0.18]
                        else if (val<0.45) return [0.25,0.49,0.14]
                        else if (val<0.5) return [0.19,0.43,0.11]
                        else if (val<0.55) return [0.13,0.38,0.07]
                        else if (val<0.9) return [0.06,0.33,0.04]
                        else return [0,0.27,0];  
                    }
                """
        }

    def get_request(self,time_interval, evalscript,cloud_coverage):
        return SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                    maxcc = cloud_coverage, #cloud coverage
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=self.image_bbox,
            size=self.image_size,
            config=self.sentinel_hub_config,
        )    

    def get_image(self, latitude, longitude, size_fragments, start_year, end_year, month, types_image, cloud_coverage):
        # Where we are going to save all fragments of images for a coordinate
        dataset={}

        # Define the desired width and height of the bounding box in degrees
        bbox_width = 0.2  # Example: 0.1 degrees (approximately 11.1 km at the equator)
        bbox_height = 0.2

        # Calculate the bounding box coordinates
        lower_left_lat = latitude - bbox_height / 2
        lower_left_lon = longitude - bbox_width / 2
        upper_right_lat = latitude + bbox_height / 2
        upper_right_lon = longitude + bbox_width / 2

        # Create the bounding box
        self.image_coords_wgs84 = (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
        self.resolution = 9
        self.image_bbox = BBox(bbox=self.image_coords_wgs84, crs=CRS.WGS84)
        self.image_size = bbox_to_dimensions(self.image_bbox, resolution=self.resolution) # Size of the final image
        
        # Create dictionaries by years to save images
        for year in range(start_year, end_year + 1):
            dataset[year] = {}

        for type_image in types_image:

            data = []

            for year in range(start_year, end_year + 1):

                # Generate filename for image
                filedir = f"dataset/{year}"
                filename = f"Latitude_{round(latitude,2)}_Longitude_{round(longitude,2)}_Type_{type_image}.png"
                filepath = os.path.join(os.getcwd(),filedir, filename)

                # Create the output directory if it doesn't exist
                os.makedirs(filedir, exist_ok=True)

                # Check if image exists
                if os.path.exists(filepath):
                    print(f"Image for year {year} in {type_image} already stored")
                    data.append(cv2.imread(filepath))
                
                else:
                    start_date = datetime.datetime(year, month[0], 1)
                    end_date = datetime.datetime(year, month[1], 31)
                    time_interval = (start_date.date().isoformat(), end_date.date().isoformat())
                    image =  self.get_request(time_interval,self.evalscript_functions[type_image],cloud_coverage).get_data()[0]
                    print(f"Getting image from Sentinel Hub from year {year} in {type_image} with a resolution of {self.image_size[0]}x{self.image_size[1]}")
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
                dataset[year][type_image] = fragment_list
                year = year + 1
        
        print(f"Number of samples (fragments) for every year is ${len(dataset[start_year][types_image[0]])}")

        return dataset
    
    def display_random_samples(self, dataset, number_of_samples,type_image):
        ncols = len(dataset.keys())
        nrows = number_of_samples  # Number of rows of samples

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols , 5 * nrows))

        for i in range(nrows):
            number_of_fragments = len(dataset[2017]['rgb'])
            index_random_image = random.randint(0, number_of_fragments)
            for j,year in enumerate(dataset.keys()):
                # Get a random image from the selected year
                ax = axs[i, j]
                image = dataset[year][type_image][index_random_image]
                ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
                ax.set_title(f"Year {year} - Index {index_random_image}", fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def display_region_along_years(self, dataset, index, type_image):
        ncols = len(dataset)

        fig, axs = plt.subplots(ncols=ncols, figsize=(5 * ncols, 5))

        for j, year in enumerate(dataset.keys()):
            # Get a random image from the selected year
            ax = axs[j]  # Use a single index for the subplot
            image = dataset[year][type_image][index]
            ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
            ax.set_title(f"Year {year} - Index {index}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.show()