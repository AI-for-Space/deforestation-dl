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
    def __init__(self):
        
        self.sentinel_hub_config = SHConfig(
            instance_id='3e8b80d9-d9eb-43e1-84c7-44a569e6ba83',
            sh_client_id='570d0ec5-4d9b-4852-8d30-45e1af205e89',
            sh_client_secret='rXF2Lz4|-%yHoe2dPBgp{e-10-[8s?X3*m:)&2r}',
            sh_base_url='https://services.sentinel-hub.com',
            sh_auth_base_url='https://services.sentinel-hub.com',
        )

    def get_image(self):
        betsiboka_coords_wgs84 = (46.16, -16.15, 46.51, -15.58)
        resolution = 60
        betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
        betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)
        print(f"Image shape at {resolution} m resolution: {betsiboka_size} pixels")


        evalscript_true_color = """
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

        request_true_color = SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=("2020-06-12", "2020-06-13"),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=betsiboka_bbox,
            size=betsiboka_size,
            config=self.sentinel_hub_config,
        )

        true_color_imgs = request_true_color.get_data()

        print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
        print(f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")

        image = true_color_imgs[0]
        print(f"Image type: {image.dtype}")
        
        # Adjust brightness
        brightness_factor = 3.5 / 255
        brightened_image = np.clip(image * brightness_factor, 0, 1)

        plt.imshow(brightened_image, aspect='auto')
        plt.title('True Color Image')
        plt.xlabel('Pixel Column')
        plt.ylabel('Pixel Row')
        plt.show()

        # plot function
        # factor 1/255 to scale between 0-1
        # factor 3.5 to increase brightness
        #plot_image(image, factor=3.5 / 255, clip_range=(0, 1))



