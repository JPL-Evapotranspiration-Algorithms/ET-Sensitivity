from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

from .constants import GAMMA_PA

def calculate_transpiration(
        delta_Pa: Union[Raster, np.ndarray],
        Ac: Union[Raster, np.ndarray],
        rho: Union[Raster, np.ndarray], 
        Cp: Union[Raster, np.ndarray], 
        VPD_Pa: Union[Raster, np.ndarray], 
        FVC: Union[Raster, np.ndarray], 
        ra: Union[Raster, np.ndarray], 
        fwet: Union[Raster, np.ndarray], 
        rs: Union[Raster, np.ndarray], 
        gamma_Pa: Union[Raster, np.ndarray, float] = GAMMA_PA) -> Union[Raster, np.ndarray]:
    """
    Calculates the transpiration (LEc) using the Penman-Monteith equation.

    Parameters:
        delta_Pa (Union[Raster, np.ndarray]): slope of saturation vapor pressure curve in Pascal per degree Celsius
        Ac (Union[Raster, np.ndarray]): available radiation to the canopy in watts per square meter
        rho (Union[Raster, np.ndarray]): calculate air density in kilograms per cubic meter
        Cp (Union[Raster, np.ndarray]): specific heat capacity of the air in joules per kilogram per kelvin
        VPD_Pa (Union[Raster, np.ndarray]): vapor pressure deficit in Pascal
        FVC (Union[Raster, np.ndarray]): fraction of vegetation cover
        ra (Union[Raster, np.ndarray]): aerodynamic resistance (ra) in s/m.
        fwet (Union[Raster, np.ndarray]): fraction of wet area (fwet).
        rs (Union[Raster, np.ndarray]): surface resistance (rs) in s/m.
        gamma_Pa (Union[Raster, np.ndarray, float], optional): psychrometric constant (gamma) in Pascal per degree Celsius

    Returns:
        Union[Raster, np.ndarray]: transpiration (LEc) in mm/day.
    """
    numerator = (delta_Pa * Ac + (rho * Cp * FVC * VPD_Pa / ra)) * (1.0 - fwet)
    denominator = delta_Pa + gamma_Pa * (1.0 + (rs / ra))
    LEc = numerator / denominator

    # fill transpiration with zero
    LEc = rt.where(rt.isnan(LEc), 0.0, LEc)

    return LEc