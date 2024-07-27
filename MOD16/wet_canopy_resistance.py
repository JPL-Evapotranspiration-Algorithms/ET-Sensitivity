from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

from .constants import MIN_RESISTANCE, MAX_RESISTANCE

def calculate_wet_canopy_resistance(
        conductance: Union[Raster, np.ndarray], 
        LAI: Union[Raster, np.ndarray], 
        fwet: Union[Raster, np.ndarray],
        min_resistance: float = MIN_RESISTANCE,
        max_resistance: float = MAX_RESISTANCE) -> Union[Raster, np.ndarray]:
    """
    calculates wet canopy resistance.
    :param conductance: leaf conductance to evaporated water vapor
    :param LAI: leaf-area index
    :param fwet: relative surface wetness
    :return: wet canopy resistance
    """
    return rt.clip(1.0 / rt.clip(conductance * LAI * fwet, 1.0 / max_resistance, None), min_resistance, max_resistance)
