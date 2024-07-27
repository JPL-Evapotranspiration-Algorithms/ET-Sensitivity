import numpy as np
from typing import Union
from rasters import Raster

from .constants import RH_THRESHOLD, MIN_FWET

def calculate_fwet(
        RH: Union[Raster, np.ndarray], 
        RH_threshold: float = RH_THRESHOLD, 
        min_fwet: float = MIN_FWET) -> Union[Raster, np.ndarray]:
    """
    calculates relative surface wetness
    :param RH: relative humdity from 0.0 to 1.0
    :return: relative surface wetness from 0.0 to 1.0
    """
    fwet = np.float32(np.clip(RH ** 4.0, min_fwet, None))

    if RH_threshold is not None:
        fwet = np.where(RH < RH_threshold, min_fwet, fwet)

    return fwet
