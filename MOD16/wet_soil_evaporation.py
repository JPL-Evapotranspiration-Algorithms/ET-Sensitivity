from typing import Union
import numpy as np
from rasters import Raster

from .constants import GAMMA_PA

def calculate_wet_soil_evaporation(
        delta_Pa: Union[Raster, np.ndarray], 
        Asoil: Union[Raster, np.ndarray], 
        rho: Union[Raster, np.ndarray], 
        Cp: Union[Raster, np.ndarray], 
        FVC: Union[Raster, np.ndarray], 
        VPD: Union[Raster, np.ndarray], 
        ras: Union[Raster, np.ndarray], 
        fwet: Union[Raster, np.ndarray], 
        rtot: Union[Raster, np.ndarray],
        gamma_Pa: float = GAMMA_PA) -> Union[Raster, np.ndarray]:
    """
    MOD16 method for calculating wet soil evaporation
    :param s: slope of the saturation to vapor pressure curve
    :param Asoil: available radiation to the soil (soil partition of net radiation)
    :param rho: air density in kilograms per cubic meter
    :param Cp: specific humidity
    :param fc: vegetation fraction
    :param VPD: vapor pressure deficit
    :param ras: aerodynamic resistance at the soil surface
    :param fwet: relative surface wetness
    :param rtot: total aerodynamic resistance
    :param gamme: gamma constant (default: GAMMA)
    :return: wet soil evaporation in watts per square meter
    """
    numerator = (delta_Pa * Asoil + rho * Cp * (1.0 - FVC) * VPD / ras) * fwet
    denominator = delta_Pa + gamma_Pa * rtot / ras
    LE_soil_wet = numerator / denominator

    return LE_soil_wet
