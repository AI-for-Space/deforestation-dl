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
                        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)

                        if (ndvi < 0) return [0.05, 0.05, 0.05];
                        else if (ndvi < 0.05) return [0.75, 0.75, 0.75];
                        else if (ndvi < 0.1) return [0.86, 0.86, 0.86];
                        else if (ndvi < 0.15) return [0.92, 0.92, 0.92];
                        else if (ndvi < 0.2) return [1, 0.98, 0.8];
                        else if (ndvi < 0.25) return [0.93, 0.91, 0.71];
                        else if (ndvi < 0.3) return [0.87, 0.85, 0.61];
                        else if (ndvi < 0.35) return [0.8, 0.78, 0.51];
                        else if (ndvi < 0.4) return [0.74, 0.72, 0.42];
                        else if (ndvi < 0.45) return [0.69, 0.76, 0.38];
                        else if (ndvi < 0.5) return [0.64, 0.8, 0.35];
                        else if (ndvi < 0.55) return [0.57, 0.75, 0.32];
                        else if (ndvi < 0.6) return [0.5, 0.7, 0.28];
                        else if (ndvi < 0.61) return [0.48, 0.69, 0.27];  // Additional intervals starting at 0.6
                        else if (ndvi < 0.62) return [0.46, 0.68, 0.26];
                        else if (ndvi < 0.63) return [0.44, 0.67, 0.25];
                        else if (ndvi < 0.64) return [0.42, 0.66, 0.24];
                        else if (ndvi < 0.65) return [0.4, 0.65, 0.23];
                        else if (ndvi < 0.66) return [0.38, 0.64, 0.22];
                        else if (ndvi < 0.67) return [0.36, 0.63, 0.21];
                        else if (ndvi < 0.68) return [0.34, 0.62, 0.2];
                        else if (ndvi < 0.69) return [0.32, 0.61, 0.19];
                        else if (ndvi < 0.7) return [0.31, 0.6, 0.18];
                        else if (ndvi < 0.71) return [0.3, 0.59, 0.175];
                        else if (ndvi < 0.72) return [0.29, 0.58, 0.17];
                        else if (ndvi < 0.73) return [0.28, 0.57, 0.165];
                        else if (ndvi < 0.74) return [0.27, 0.56, 0.16];
                        else if (ndvi < 0.75) return [0.26, 0.55, 0.155];
                        else if (ndvi < 0.76) return [0.25, 0.54, 0.15];
                        else if (ndvi < 0.77) return [0.24, 0.53, 0.145];
                        else if (ndvi < 0.78) return [0.23, 0.52, 0.14];
                        else if (ndvi < 0.79) return [0.22, 0.51, 0.135];
                        else if (ndvi < 0.8) return [0.21, 0.5, 0.13];
                        else if (ndvi < 0.81) return [0.2, 0.49, 0.125];
                        else if (ndvi < 0.82) return [0.19, 0.48, 0.12];
                        else if (ndvi < 0.83) return [0.18, 0.47, 0.115];
                        else if (ndvi < 0.84) return [0.17, 0.46, 0.11];
                        else if (ndvi < 0.85) return [0.16, 0.45, 0.105];
                        else if (ndvi < 0.86) return [0.15, 0.44, 0.1];
                        else if (ndvi < 0.87) return [0.14, 0.43, 0.095];
                        else if (ndvi < 0.88) return [0.13, 0.42, 0.09];
                        else if (ndvi < 0.89) return [0.12, 0.41, 0.085];
                        else if (ndvi < 0.9) return [0.11, 0.4, 0.08];
                        else if (ndvi < 0.91) return [0.1, 0.39, 0.075];
                        else if (ndvi < 0.92) return [0.095, 0.38, 0.07];
                        else if (ndvi < 0.93) return [0.09, 0.37, 0.065];
                        else if (ndvi < 0.94) return [0.085, 0.36, 0.06];
                        else if (ndvi < 0.95) return [0.08, 0.35, 0.055];
                        else if (ndvi < 0.96) return [0.075, 0.34, 0.05];
                        else if (ndvi < 0.97) return [0.07, 0.33, 0.045];
                        else if (ndvi < 0.98) return [0.065, 0.32, 0.04];
                        else if (ndvi < 0.99) return [0.06, 0.31, 0.035];
                        else return [0.055, 0.3, 0.03];
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

        print(f"===========================================================================================================================================")
        print(f"Getting images for coordinates [Latitude {round(latitude,2)}, Longitude {round(longitude,2)}]")

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
        
        print(f"Number of samples (fragments) for every year is {len(dataset[start_year][types_image[0]])}")

        return dataset
    
    def display_random_samples_dataset(self, dataset, number_of_samples,type_image):
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