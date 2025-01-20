import pandas as pd
import os
import datetime
import cv2

class sentinel2df:
    def __init__(self, dir_path):
        self.dir_path        = dir_path
        self.files           = os.listdir(dir_path)
        self.df              = pd.DataFrame()
        self.dates           = list()
        self.density_vv_list = list()
        self.density_vh_list = list()


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

                image_vv    = f'{self.dir_path}/{string_date}_vv.png'
                image_vh    = f'{self.dir_path}/{string_date}_vh.png'

                density_vv  = self.get_density(cv2.imread(image_vv, cv2.COLOR_BGR2GRAY))
                density_vh  = self.get_density(cv2.imread(image_vh, cv2.COLOR_BGR2GRAY))

                self.density_vv_list.append(density_vv)
                self.density_vh_list.append(density_vh)

            except Exception as e:
                print(f"For {d}\t> {e}")
                self.density_vv_list.append(None)
                self.density_vh_list.append(None)

        self.df["VV"] = self.density_vv_list
        self.df["VH"] = self.density_vh_list
        self.df = self.df.dropna()
        self.df["CR"] = self.df["VV"].values/self.df["VH"].values


        return self.df

if __name__ == '__main__':
    print(sentinel2df("./Sentinel1_Images_Beauvais/2017").make_df())