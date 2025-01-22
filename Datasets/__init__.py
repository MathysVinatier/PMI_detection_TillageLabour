import sys

if sys.path[0].split("/")[-1] == "Datasets" :
    from dataframe_sentinel1 import *
    from dataframe_sentinel2 import *
    from extract_sentinel1   import *
    from extract_sentinel2   import *

else:        
    from Datasets.dataframe_sentinel1 import *
    from Datasets.dataframe_sentinel2 import *
    from Datasets.extract_sentinel1   import *
    from Datasets.extract_sentinel2   import *

__all__ = ['sentinelOne2df', 'sentinelTwo2df', 'extract_sentinel1', 'extract_sentinel2']