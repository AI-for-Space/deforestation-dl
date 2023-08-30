import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

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
                        bands: ["B02", "B03", "B04"]
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

    def get_true_color_request(self,time_interval):

        print(f"Image shape at {self.resolution} m resolution: {self.betsiboka_size} pixels")

        return SentinelHubRequest(
            evalscript=self.evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=self.betsiboka_bbox,
            size=self.betsiboka_size,
            config=self.sentinel_hub_config,
        )

    def get_image(self, latitude, longitude):

        # Define the desired width and height of the bounding box in degrees
        bbox_width = 0.6  # Example: 0.1 degrees (approximately 11.1 km at the equator)
        bbox_height = 0.6

        # Calculate the bounding box coordinates
        lower_left_lat = latitude - bbox_height / 2
        lower_left_lon = longitude - bbox_width / 2
        upper_right_lat = latitude + bbox_height / 2
        upper_right_lon = longitude + bbox_width / 2

        # Create the bounding box
        self.betsiboka_coords_wgs84 = (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
        self.resolution = 30
        self.betsiboka_bbox = BBox(bbox=self.betsiboka_coords_wgs84, crs=CRS.WGS84)
        self.betsiboka_size = bbox_to_dimensions(self.betsiboka_bbox, resolution=self.resolution)
        
        start_year = 2017
        end_year = 2023
        month = [7,8]
        time_intervals = []
    
        for year in range(start_year, end_year + 1):
            start_date = datetime.datetime(year, month[0], 1)
            end_date = datetime.datetime(year, month[1], 31)
                
            time_intervals.append((start_date.date().isoformat(), end_date.date().isoformat()))

        print(time_intervals)

        # create a list of requests
        list_of_requests = [self.get_true_color_request(slot) for slot in time_intervals]
        list_of_requests = [request.download_list[0] for request in list_of_requests]

        # download data with multiple threads
        data = SentinelHubDownloadClient(config=self.sentinel_hub_config).download(list_of_requests, max_threads=5)

        ncols = 4
        nrows = 2
        aspect_ratio = self.betsiboka_size[0] / self.betsiboka_size[1]
        subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows), subplot_kw=subplot_kw)

        for idx, image in enumerate(data):
            ax = axs[idx // ncols][idx % ncols]
            ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
            ax.set_title(f"{time_intervals[idx][0]}  -  {time_intervals[idx][1]}", fontsize=10)

        plt.tight_layout()
        plt.show()