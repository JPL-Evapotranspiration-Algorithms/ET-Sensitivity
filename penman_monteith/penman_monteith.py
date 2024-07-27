from typing import Union
import numpy as np
from evapotranspiration_conversion.evapotranspiration_conversion import lambda_Jkg_from_Ta_C
from meteorology_conversion.meteorology_conversion import celcius_to_kelvin

from rasters import Raster

SPECIFIC_HEAT_CAPACITY_AIR = 1013 # J kg-1 K-1, Monteith & Unsworth (2001)
MOL_WEIGHT_WET_DRY_RATIO_AIR = 0.622

def calculate_gamma(
        Ta_C: Union[Raster, np.ndarray],
        Ps_Pa: Union[Raster, np.ndarray],
        Cp_Jkg: Union[Raster, np.ndarray, float] = SPECIFIC_HEAT_CAPACITY_AIR,
        RMW: Union[Raster, np.ndarray, float] = MOL_WEIGHT_WET_DRY_RATIO_AIR) -> Union[Raster, np.ndarray]:
    # calculate latent heat of vaporization (J kg-1)
    lambda_Jkg = lambda_Jkg_from_Ta_C(Ta_C)
    gamma =  (Cp_Jkg * Ps_Pa) / (lambda_Jkg * RMW)
    
    return gamma