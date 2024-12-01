#EarthEngine
import ee

#Extra
import datetime
import os
import requests
import sys
from typing import Optional, Tuple

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
        timestamp_begin = datetime.date(timestamp[2], timestamp[1], timestamp[0]).strftime('%y-%m-%d')
        timestamp_end   = datetime.date(timestamp[2], timestamp[1], timestamp[0]+1).strftime('%y-%m-%d')

        # Get timestamped GEE
        self.current_sent1 = self.current_sent1.filterDate(timestamp_begin, timestamp_end)

        # Filter the Sentinel-1 collection by metadata properties.
        self.vv_vh_iw = (
            self.sentinel_1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
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
        current_date = datetime.date(start[2], start[1], start[0])

        # Handle the whether it is a unique timestamp or not
        if end == None:
            date_end = current_date

        else:
            date_end     = datetime.date(end[2], end[1], end[0])

        # Handle timestamps order issue
        if current_date > date_end :
            print("\n/!\\ Ending Date is BEFORE Starting Date /!\\")
            return
        
        script_begin = datetime.datetime.now() # Used to get elapsed time
        
        # Creation of the progress bar
        total_days = (date_end + datetime.timedelta(days=step_day) - current_date).days
        print(f"\nSaving images in \033[34m{os.getcwd()}/{self.path_to_image_folder}\033[0m")
        longueur_barre = 30 
        i = 0
        pourcentage = int((i / total_days) * 100)
        fill  = '#' * int((i / total_days) * longueur_barre)
        blank = '-' * (longueur_barre - len(fill))
        
        # Get current elapsed time
        elapsed_time = datetime.datetime.now() - script_begin

        # Print out the progress bar
        sys.stdout.write(f'\r[{fill}{blank}] {pourcentage}% - Date : {current_date} (elapsed time : {round(elapsed_time.total_seconds(), 3)} s)')
        sys.stdout.flush()

        while current_date <= date_end:
                
            # Get different dtype for current timestamp
            string_date = current_date.strftime("%d_%m_%y")
            tuple_date  = (current_date.day, current_date.month, current_date.year)

            try :
                # If user wants to save vv images
                if is_vv:
                    vv_image = self.__get_vv(tuple_date)
                    url = vv_image.getThumbURL({'min': -20, 'max': -0, 'dimensions': 512, 'region': self.polygon_roi, 'format': 'png'})
                    response = requests.get(url, stream=True)

                    vv_tif_path = self.path_to_image_folder + string_date+"_vv.png"
                    with open(vv_tif_path, 'wb') as file:
                        file.write(response.content)

                # If user wants to save vh images
                if is_vh:
                    vh_image = self.__get_vh(tuple_date)
                    url = vh_image.getThumbURL({'min': -20, 'max': -0, 'dimensions': 512, 'region': self.polygon_roi, 'format': 'png'})
                    response = requests.get(url, stream=True)

                    vh_tif_path = self.path_to_image_folder + string_date+"_vh.png"
                    with open(vh_tif_path, 'wb') as file:
                        file.write(response.content)

            except Exception as e:
                self.__write_log(e, context=f'({current_date})')

            # Increment current_date by the given step
            current_date += datetime.timedelta(days=step_day)

            # Progress bar print out
            i += 1
            pourcentage = int((i / total_days) * 100)
            fill  = '#' * int((i / total_days) * longueur_barre)
            blank = '-' * (longueur_barre - len(fill))

            elapsed_time = datetime.datetime.now() - script_begin
            sys.stdout.write(f'\r[{fill}{blank}] {pourcentage}% - Date : {current_date} (elapsed time : {round(elapsed_time.total_seconds(), 3)} s)')
            sys.stdout.flush()

        print("\nDone !\n")



if __name__ == '__main__':
    """
    +---------------------------------------------------------------------+
    | This is an usage example to save Sentinel1 images from Beauvais ROI |
    | from 10th of March 2017 to 14th of March 2017                       |
    +---------------------------------------------------------------------+
    -> ROI : https://geojson.io/
    """
    beauvais_roi = [
        [2.4717675770780545, 48.49202395041567],
        [2.4735192968839783, 48.49453601542879],
        [2.47749656234393, 48.492794710870555],
        [2.4757161258208384, 48.49032062341078]
    ]
    roi_name = "Beauvais"

    time_start = (1,1,2017)
    time_stop  = (31,12,2017)

    data = extract_sentinel1(beauvais_roi, roi_name)

    data.save(time_start, time_stop, 15)

