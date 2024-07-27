from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

from .constants import GAMMA_PA

def calculate_interception(
        delta_Pa: Union[Raster, np.ndarray],
        Ac: Union[Raster, np.ndarray],
        rho: Union[Raster, np.ndarray],
        Cp: Union[Raster, np.ndarray],
        VPD_Pa: Union[Raster, np.ndarray],
        FVC: Union[Raster, np.ndarray],
        rhrc: Union[Raster, np.ndarray],
        fwet: Union[Raster, np.ndarray],
        rvc: Union[Raster, np.ndarray],
        water: Union[Raster, np.ndarray],
        gamma_Pa: Union[Raster, np.ndarray, float] = GAMMA_PA) -> Union[Raster, np.ndarray]:
    """
    Calculates the wet evaporation partition of the latent heat flux using the MOD16 method.

    :param delta_Pa: slope of saturation to vapor pressure curve in Pascal per degree Celsius
    :param Ac: available radiation to the canopy in watts per square meter
    :param rho: air density in kilograms per cubic meter
    :param Cp: specific heat capacity of the air in joules per kilogram per kelvin
    :param VPD: vapor pressure deficit in Pascal
    :param FVC: fraction of vegetation cover
    :param rhrc: aerodynamic resistance in seconds per meter
    :param fwet: relative surface wetness
    :param rvc: wet canopy resistance
    :param water: water content in the canopy
    :param gamma_Pa: psychrometric constant for atmospheric pressure in Pascal (default: GAMMA_PA)

    :return: wet evaporation in watts per square meter
    """
    numerator = (delta_Pa * Ac + (rho * Cp * VPD_Pa * FVC / rhrc)) * fwet
    denominator = delta_Pa + gamma_Pa * (rvc / rhrc)
    LEi = numerator / denominator

    return LEi
