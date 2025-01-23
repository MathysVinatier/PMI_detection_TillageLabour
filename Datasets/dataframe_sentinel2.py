import pandas as pd
import os
import datetime
import cv2

class sentinelTwo2df:
    def __init__(self, dir_path):
        self.dir_path        = dir_path
        self.files           = os.listdir(dir_path)
        self.df              = pd.DataFrame()
        self.dates           = list()
        self.density_ndvi_list = list()


    def get_density(self, image):
        return image[image!=0].mean()

    def make_df(self):

        for f in self.files:
            date        = "/".join(f.split("_")[:3])
            date_format = datetime.datetime.strptime(date, '%d/%m/%y')
            self.dates.append(date_format)

        self.df["Date"]  = self.dates.copy()
        self.df          = self.df.drop_duplicates()
        self.df          = self.df.sort_values("Date")

        for d in self.df["Date"]:
            try :
                string_date = d.strftime('%d_%m_%y')

                image_ndvi    = f'{self.dir_path}/{string_date}_NDVI.png'

                density_ndvi  = self.get_density(cv2.imread(image_ndvi, cv2.COLOR_BGR2GRAY))

                self.density_ndvi_list.append(density_ndvi)

            except Exception as e:
                print(f"For {d}\t> {e}")
                self.density_ndvi_list.append(None)

        self.df["NDVI"] = self.density_ndvi_list
        self.df = self.df.dropna()


        return self.df

if __name__ == '__main__':
    print(sentinelTwo2df("./Sentinel2_Images_Catillon/2018").make_df())