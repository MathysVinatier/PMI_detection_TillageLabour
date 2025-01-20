import sys

if sys.path[0].split("/")[-1] == "Datasets" :
    from dataframe_sentinel1 import *
    from extract_sentinel1   import *

else:        
    from Datasets.dataframe_sentinel1 import *
    from Datasets.extract_sentinel1   import *

__all__ = ['sentinel2df', 'extract_sentinel1']