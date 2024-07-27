import numpy as np
from typing import Union
from .constants import MAX_RESISTANCE
import rasters as rt
from rasters import Raster

def calculate_canopy_conductance(
    LAI: Union[Raster, np.ndarray],
    fwet: Union[Raster, np.ndarray],
    gl_sh: Union[Raster, np.ndarray],
    gs1: Union[Raster, np.ndarray],
    Gcu: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    calculate canopy conductance (Cc)
    Canopy conductance (Cc) to transpired water vapor per unit LAI is derived from stomatal
    and cuticular conductances in parallel with each other, and both in series with leaf boundary layer
    conductance (Thornton, 1998; Running & Kimball, 2005). In the case of plant transpiration,
    surface conductance is equivalent to the canopy conductance (Cc), and hence surface resistance
    (rs) is the inverse of canopy conductance (Cc).
    :param LAI: leaf-area index
    :param fwet: relative surface wetness
    :param gl_sh: leaf boundary layer conductance
    :param gs1: stomatal conductance
    :param Gcu: cuticular conductance
    :return: canopy conductance
    """
    # noinspection PyTypeChecker
    Cc = rt.where(
        rt.logical_and(LAI > 0.0, (1.0 - fwet) > 0.0),
        gl_sh * (gs1 + Gcu) / (gs1 + gl_sh + Gcu) * LAI * (1.0 - fwet),
        0.0
    )

    Cc = rt.clip(Cc, 1.0 / MAX_RESISTANCE, None)

    return Cc
