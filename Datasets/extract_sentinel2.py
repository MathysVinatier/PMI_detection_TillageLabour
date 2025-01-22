#EarthEngine
import ee

#Extra
import datetime
import os
import requests
import sys
from typing import Optional, Tuple
import threading
import time

class extract_sentinel2:

    project_name = "ee-sentinel-analysis"

    def __init__(self, roi, roi_name):

        ee.Authenticate()
        ee.Initialize(project=self.project_name)

        # Load the Sentinel-2 ImageCollection -> https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
        self.sentinel_2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        self.current_sent2 = self.sentinel_2

        # Create Region of Interest
        self.polygon_roi = ee.Geometry.Polygon(roi)
        self.path_to_image_folder = f"Sentinel2_Images_{roi_name}/"

        # Preparing new timestamp
        self.old_timestamp=(0,0,0)

    def mask_s2_clouds(self, image):
        """Masks clouds in a Sentinel-2 image using the QA band.

        Args:
            image (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: A cloud-masked Sentinel-2 image.
        """
        qa = image.select('QA60')

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask).divide(10000)
    
    def __calculate_ndvi(self, image):
        """Calculates the NDVI for a Sentinel-2 image.

        Args:
            image (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: An image with an additional NDVI band.
        """
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    def __filter_date(self, timestamp):

        # Update timestamp
        self.old_timestamp = timestamp

        # Convert tuple to time
        timestamp_begin = datetime.date(timestamp[2], timestamp[1], timestamp[0])
        timestamp_end   = timestamp_begin + datetime.timedelta(days=2)

        # Get timestamped GEE
        self.period = ee.Filter.date(timestamp_begin.strftime('%Y-%m-%d'), timestamp_end.strftime('%Y-%m-%d'))

        # Filter the Sentinel-2 collection by metadata properties.
        image_sent2 = (
            self.sentinel_2.filter(self.period)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(self.mask_s2_clouds)
        )

        return image_sent2

    def __write_log(self, message, context=None):
        """
        Logs an error message to `extraction.log` with a timestamp and optional context.

        Parameters:
            message (str): The error message to log.
            context (str): Additional context about the error (e.g., input values).
        """
        log_file = "./extraction.log"
        
        # Check if the log file exists
        if not os.path.exists(log_file):
            # Create the file if it doesn't exist
            with open(log_file, 'w') as f:
                f.write("Error Log\n")
                f.write("="*40 + "\n")
        
        # Append the new error to the log file
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] ERROR: {message} ")
            if context:
                f.write(f"{context}\n")

        return

    def __loading_bar(self, step_day, date_end):

        script_begin = datetime.datetime.now() # Used to get elapsed time
        start_date = self.current_date

        # Creation of the progress bar
        total_days = (date_end + datetime.timedelta(days=step_day) - self.current_date).days
        print(f"\nSaving images in \033[34m{os.getcwd()}/{self.path_to_image_folder}\033[0m")

        longueur_barre = 30 
        i = 0

        while self.current_date <= date_end:
        # Progress bar print out
            i = (self.current_date - start_date).days
            pourcentage = int((i / total_days) * 100)
            fill  = '=' * int((i / total_days) * longueur_barre)
            blank = '-' * (longueur_barre - len(fill))

            elapsed_time = datetime.datetime.now() - script_begin
            sys.stdout.write(f'\r[{fill}{blank}] {pourcentage}% - Date : {self.current_date} (elapsed time : {int(elapsed_time.total_seconds())} s)')
            sys.stdout.flush()
            time.sleep(1)


    def save(self, start : tuple , end : Optional[Tuple] = None, step_day=1, is_RGB = True, is_NDVI = True):
        '''
        Save images based on the provided date range.

        This method saves data from a specified start date to an optional end date, 
        allowing for daily steps. It can also include options to save additional 
        vertical and horizontal data based on the boolean flags provided.

        Parameters:
        ----------
        start : tuple (day, month, year)
            
        end : tuple (day, month, year) or None

        step_day (default = 1) : int

        is_RGB : boolean

        is_NDVI : boolean

        Returns:
        -------
        None

        Usage:
        ------
        To save data from a start date to an end date, call the method as follows:

        >>> instance.save(start=(17,3,2017), end=(20,3,2017), step_day=1, is_RGB = False, is_NDVI = True)

        This will save the data from March 17, 2017, to March 20, 2017, with daily increments for just NDVI bands
    '''

        # Create path if it does not exist
        if not os.path.exists(self.path_to_image_folder):
            os.mkdir(self.path_to_image_folder)

        # Get the first datetime of the timestamp
        self.current_date = datetime.date(start[2], start[1], start[0])

        # Handle the whether it is a unique timestamp or not
        if end == None:
            date_end = self.current_date

        else:
            date_end = datetime.date(end[2], end[1], end[0])

        # Handle timestamps order issue
        if self.current_date > date_end :
            print("\n/!\\ Ending Date is BEFORE Starting Date /!\\")
            return

        threading.Thread(target=lambda : self.__loading_bar(step_day, date_end)).start()

        while self.current_date <= date_end:
                
            # Get different dtype for current timestamp
            string_date = self.current_date.strftime("%d_%m_%y")
            tuple_date  = (self.current_date.day, self.current_date.month, self.current_date.year)

            try :
                sent2_image = self.__filter_date(tuple_date)
                
                pixel_count = sent2_image.mean().reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=self.polygon_roi,
                    scale=30
                ).get('B4').getInfo()

                if pixel_count is None or pixel_count == 0:
                    raise ValueError("Empty ROI")
                
                if is_RGB :
                    image_sent2_RGB = sent2_image.mean().clip(self.polygon_roi)
                    url_RGB = image_sent2_RGB.getThumbURL({'min': 0.0, 'max': 0.3, 'dimensions': 512,'region': self.polygon_roi, 'bands': ['B4', 'B3', 'B2'], 'format': 'png'})
                    response_RGB = requests.get(url_RGB, stream=True)

                    tif_path_RGB = self.path_to_image_folder + string_date+"_RGB.png"
                    with open(tif_path_RGB, 'wb') as file:
                        file.write(response_RGB.content)
                
                if is_NDVI:
                    sent2_NDVI = sent2_image.map(self.__calculate_ndvi)
                    image_sent2_NDVI = sent2_NDVI.mean().clip(self.polygon_roi)
                    url_NDVI = image_sent2_NDVI.getThumbURL({'min': 0.0, 'max': 1.0, 'dimensions': 512,'region': self.polygon_roi, 'bands': ['NDVI'], 'format': 'png'})
                    response_NDVI = requests.get(url_NDVI, stream=True)

                    tif_path_NDVI = self.path_to_image_folder + string_date+"_NDVI.png"
                    with open(tif_path_NDVI, 'wb') as file:
                        file.write(response_NDVI.content)

            except Exception as e:
                self.__write_log(e, context=f'({self.current_date})')

            # Increment current_date by the given step
            self.current_date += datetime.timedelta(days=step_day)

        print("\nDone !\n")



if __name__ == '__main__':
    """
    +---------------------------------------------------------------------+
    | This is an usage example to save Sentinel2 images from Beauvais ROI |
    | from 10th of March 2017 to 14th of March 2017                       |
    +---------------------------------------------------------------------+
    -> ROI : https://geojson.io/
    """
    catillon_roi = [[ 2.3703110742853255, 49.51292627390305 ],
                    [ 2.370971146551426 , 49.512947460881065],
                    [ 2.3717096053418345, 49.51387540856808 ],
                    [ 2.3725654276000228, 49.51394410794967 ],
                    [ 2.3735336940326306, 49.514016556731804],
                    [ 2.3751526501201994, 49.514142515077   ],
                    [ 2.3758244449259394, 49.51546420864673 ],
                    [ 2.376769892661855 , 49.5152595998436  ],
                    [ 2.3769904971330504, 49.51489130184069 ],
                    [ 2.377116556831197 , 49.51464576829835 ],
                    [ 2.3789759373771346, 49.513868237281514],
                    [ 2.379259571697787 , 49.51290653708338 ],
                    [ 2.380394108901271 , 49.511944817954486],
                    [ 2.3820959148246175, 49.511105856057725],
                    [ 2.3703093330577474, 49.50963252229451 ],
                    [ 2.3710026613964885, 49.51212897839622 ],
                    [ 2.3703093330577474, 49.51292699896456 ]]

    roi_name = "Catillon"

    time_start = (1,1,2018)
    time_stop  = (1,1,2019)

    data = extract_sentinel2(catillon_roi, roi_name)

    data.save(time_start, time_stop, is_RGB = True, is_NDVI = False)

