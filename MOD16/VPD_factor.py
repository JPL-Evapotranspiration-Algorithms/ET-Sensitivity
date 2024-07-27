from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

def calculate_VPD_factor(
        VPD_open: Union[Raster, np.ndarray], 
        VPD_close: Union[Raster, np.ndarray], 
        VPD: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Calculate the VPD factor based on the open and closed vapor pressure deficit

    Parameters:
        VPD_open (Union[Raster, np.ndarray]): open vapor pressure deficit
        VPD_close (Union[Raster, np.ndarray]): closed vapor pressure deficit
        VPD (Union[Raster, np.ndarray]): vapor pressure deficit
    
    Returns:
        Union[Raster, np.ndarray]: VPD factor
    """
    # calculate VPD factor using queried open and closed VPD
    mVPD = rt.where(VPD <= VPD_open, 1.0, np.nan)
    mVPD = rt.where(rt.logical_and(VPD_open < VPD, VPD < VPD_close), (VPD_close - VPD) / (VPD_close - VPD_open), mVPD)
    mVPD = rt.where(VPD >= VPD_close, 0.0, mVPD)

    return mVPD