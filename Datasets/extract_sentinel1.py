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

class extract_sentinel1:

    project_name = "ee-sentinel-analysis"

    def __init__(self, roi, roi_name):

        ee.Authenticate()
        ee.Initialize(project=self.project_name)

        # Load the Sentinel-1 ImageCollection
        self.sentinel_1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        self.current_sent1 = self.sentinel_1

        # Create Region of Interest
        self.polygon_roi = ee.Geometry.Polygon(roi)
        self.path_to_image_folder = f"Sentinel1_Images_{roi_name}/"

        # Preparing new timestamp
        self.old_timestamp=(0,0,0)

    def __filter_date(self, timestamp):

        # Update timestamp
        self.old_timestamp = timestamp

        # Convert tuple to time
        timestamp_begin = datetime.date(timestamp[2], timestamp[1], timestamp[0])
        timestamp_end   = timestamp_begin + datetime.timedelta(days=2)

        # Get timestamped GEE
        self.period = ee.Filter.date(timestamp_begin.strftime('%Y-%m-%d'), timestamp_end.strftime('%Y-%m-%d'))

        # Filter the Sentinel-1 collection by metadata properties.
        self.vv_vh_iw = (
            self.sentinel_1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(self.period)
        )


    def __get_vv_vh_iw_asc(self, timestamp):

        if timestamp!=self.old_timestamp:
            self.__filter_date(timestamp)
        vv_vh_iw_asc  = self.vv_vh_iw.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).mean()

        return vv_vh_iw_asc.clip(self.polygon_roi)
    
    def __get_vv_vh_iw_desc(self, timestamp):
        
        if timestamp!=self.old_timestamp:
            self.__filter_date(timestamp)
        vv_vh_iw_desc = self.vv_vh_iw.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).mean()

        return vv_vh_iw_desc.clip(self.polygon_roi)

    def __get_vv(self, timestamp):
        
        if timestamp!=self.old_timestamp:
            self.__filter_date(timestamp)
        vv_image = self.vv_vh_iw.select('VV').mean()  # Mean VV for the day

        return vv_image.clip(self.polygon_roi)

    def __get_vh(self, timestamp):

        if timestamp!=self.old_timestamp:
            self.__filter_date(timestamp)
        vh_image = self.vv_vh_iw.select('VH').mean()  # Mean VH for the day

        return vh_image.clip(self.polygon_roi)

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

        self.bar_running = True

        while (self.current_date <= date_end) and (self.bar_running==True):
        # Progress bar print out
            i = (self.current_date - start_date).days
            pourcentage = int((i / total_days) * 100)
            fill  = '=' * int((i / total_days) * longueur_barre)
            blank = '-' * (longueur_barre - len(fill))

            elapsed_time = datetime.datetime.now() - script_begin
            sys.stdout.write(f'\r[{fill}{blank}] {pourcentage}% - Date : {self.current_date} (elapsed time : {int(elapsed_time.total_seconds())} s)')
            sys.stdout.flush()
            time.sleep(1)


    def save(self, start : tuple , end : Optional[Tuple] = None, step_day=1, is_vv=True, is_vh=True):
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

        is_vv (default = True) : bool

        is_vh (default = True) : bool

        Returns:
        -------
        None

        Usage:
        ------
        To save data from a start date to an end date, call the method as follows:

        >>> instance.save(start=(17,3,2017), end=(20,3,2017), step_day=1, is_vv=True, is_vh=False)

        This will save the data from March 17, 2017, to March 20, 2017, with daily increments, 
        including vertical velocity data and excluding vertical height data.
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
            date_end     = datetime.date(end[2], end[1], end[0])

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

                # If user wants to save vv images
                if is_vv:
                    vv_image = self.__get_vv(tuple_date)

                    pixel_count = vv_image.reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=self.polygon_roi,
                        scale=30
                    ).get('VV').getInfo()

                    if pixel_count is None or pixel_count == 0:
                        raise ValueError("Empty ROI")

                    url = vv_image.getThumbURL({'min': -25, 'max': 0, 'dimensions': 512, 'region': self.polygon_roi, 'format': 'png'})
                    response = requests.get(url, stream=True)

                    vv_tif_path = self.path_to_image_folder + string_date+"_vv.png"
                    with open(vv_tif_path, 'wb') as file:
                        file.write(response.content)

                # If user wants to save vh images
                if is_vh:
                    vh_image = self.__get_vh(tuple_date)

                    pixel_count = vh_image.reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=self.polygon_roi,
                        scale=30
                    ).get('VH').getInfo()

                    if pixel_count is None or pixel_count == 0:
                        raise ValueError("Empty ROI")

                    url = vh_image.getThumbURL({'min': -20, 'max': -0, 'dimensions': 512, 'region': self.polygon_roi, 'format': 'png'})
                    response = requests.get(url, stream=True)

                    vh_tif_path = self.path_to_image_folder + string_date+"_vh.png"
                    with open(vh_tif_path, 'wb') as file:
                        file.write(response.content)

            except Exception as e:
                self.__write_log(e, context=f'({self.current_date})')

            # Increment current_date by the given step
            self.current_date += datetime.timedelta(days=step_day)
        self.bar_running = False
        print("\nDone !\n")



if __name__ == '__main__':
    """
    +---------------------------------------------------------------------+
    | This is an usage example to save Sentinel1 images from Beauvais ROI |
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

    data = extract_sentinel1(catillon_roi, roi_name)

    data.save(time_start)

