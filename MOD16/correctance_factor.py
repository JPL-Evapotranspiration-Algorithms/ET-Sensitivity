from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

def calculate_rcorr(
        Ps_Pa: Union[Raster, np.ndarray], 
        Ta_K: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    calculates correctance factor (rcorr)
    for stomatal and cuticular conductances
    from surface pressure and air temperature.
    :param Ps_Pa: surface pressure in Pascal
    :param Ta_K: near-surface air temperature in kelvin
    :return: correctance factor (rcorr)
    """
    return 1.0 / ((101300.0 / Ps_Pa) * (Ta_K / 293.15) ** 1.75)
