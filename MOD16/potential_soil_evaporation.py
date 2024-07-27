from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

from .constants import GAMMA_PA

def calculate_potential_soil_evaporation(
        delta_Pa: float,
        Asoil: float,
        rho: float,
        Cp_Jkg: float,
        FVC: float,
        VPD: float,
        ras: float,
        fwet: float,
        rtot: float,
        gamma_Pa: float = GAMMA_PA) -> Union[Raster, np.ndarray]:
    """
    MOD16 method for calculating potential soil evaporation using Penman-Monteith

    Parameters:
    s (float): Slope of the saturation to vapor pressure curve.
    Asoil (float): Available radiation at the soil.
    rho (float): Density of air.
    Cp (float): Specific heat capacity of air.
    fc (float): Fraction of soil covered by vegetation.
    VPD (float): Vapor pressure deficit.
    ras (float): Aerodynamic resistance.
    fwet (float): Fraction of soil surface wetted.
    rtot (float): Total resistance.
    gamma (float, optional): Psychrometric constant. Defaults to GAMMA.

    Returns:
    Union[Raster, np.ndarray]: The potential soil evaporation.

    Notes:
    - The potential soil evaporation is calculated using the Penman-Monteith equation.
    - The Penman-Monteith equation takes into account various factors such as radiation, air density, heat capacity, vegetation cover, vapor pressure deficit, aerodynamic resistance, soil wetness, and total resistance.
    - The function returns the potential soil evaporation as either a Raster object or a NumPy array, depending on the input data type.
    """
    numerator = (delta_Pa * Asoil + rho * Cp_Jkg * (1.0 - FVC) * VPD / ras) * (1.0 - fwet)
    denominator = delta_Pa + gamma_Pa * rtot / ras
    LE_soil_pot = numerator / denominator

    LE_soil_pot = rt.clip(LE_soil_pot, 0.0, None)

    return LE_soil_pot