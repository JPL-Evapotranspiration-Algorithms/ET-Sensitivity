from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

def calculate_rtotc(
    VPD: Union[Raster, np.ndarray],
    vpd_open: Union[Raster, np.ndarray],
    vpd_close: Union[Raster, np.ndarray],
    rbl_max: Union[Raster, np.ndarray],
    rbl_min: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    calculates total aerodynamic resistance to the canopy
    from vapor pressure deficit and biome-specific constraints.
    :param VPD: vapor pressure deficit in Pascal
    :param vpd_open: vapor pressure deficit when stomata are open in Pascal
    :param vpd_close: vapor pressure deficit when stomata are closed in Pascal
    :param rbl_max: maximum boundary layer resistance in seconds per meter
    :param rbl_min: minimum boundary layer resistance in seconds per meter
    :return: aerodynamic resistance to the canopy in seconds per meter
    """
    rtotc = rt.where(VPD <= vpd_open, rbl_max, np.nan)
    rtotc = rt.where(VPD >= vpd_close, rbl_min, rtotc)

    rtotc = rt.where(
        rt.logical_and(vpd_open < VPD, VPD < vpd_close),
        rbl_min + (rbl_max - rbl_min) * (vpd_close - VPD) / (vpd_close - vpd_open),
        rtotc
    )

    return rtotc