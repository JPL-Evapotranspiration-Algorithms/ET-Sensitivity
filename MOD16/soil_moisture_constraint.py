import numpy as np
from typing import Union
from rasters import Raster

from .constants import BETA

def calculate_fSM(
        RH: Union[Raster, np.ndarray],
        VPD: Union[Raster, np.ndarray],
        beta: float = BETA) -> Union[Raster, np.ndarray]:
    """
    This function calculates the soil moisture constraint for the JPL adaptation
    of the Penman-Monteith MOD16 algorithm.
    :param RH: relative humidity between zero and one
    :param VPD: vapor pressure deficit in Pascal
    :param beta: VPD factor in Pascal (MOD16 uses 200, but PT-JPL uses 1000)
    :return: soil moisture constraint between zero and one
    """
    return RH ** (VPD / beta)
